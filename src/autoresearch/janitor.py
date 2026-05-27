"""Passive triage for orphan processes — counterpart to gpu_monitor's active triage.

Detects PPID=1 processes that no longer have a productive owner:
- Python procs from worktrees that have been removed
- Game servers (SC2, pyboy, gym, 2048) whose parent runner is dead
- Stale multiprocessing pool forks > 24h old with no active sweep

Dry-run by default (prints a table). `--apply` kills (SIGTERM, 4s grace, then
SIGKILL stragglers).

Usage:

    uv run autoresearch-janitor                # dry-run, table output
    uv run autoresearch-janitor --apply        # actually kill
    uv run autoresearch-janitor --json         # machine-readable
    uv run autoresearch-janitor --apply --json # both

Detection rules are pure functions in `find_orphans` so they can be unit-tested
without spawning real procs (see tests/test_janitor.py).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# ── Constants ───────────────────────────────────────────────────────────────

# Stale multiprocessing pool: kill only if older than this. Active sweeps will
# typically have forks much younger than this; older forks are almost always
# leftover from crashes / kills that didn't propagate.
STALE_MULTIPROCESSING_AGE_S = 24 * 3600

SIGTERM_WAIT_S = 4

# Procs we will NEVER kill, even if they appear orphan. These are infra/system
# services that legitimately run with PPID=1 forever.
KEEP_ALWAYS_PATTERNS = (
    re.compile(r"\bvllm\.entrypoints"),
    re.compile(r"\bhttp\.server\b"),
    re.compile(r"jupyter-notebook"),
    re.compile(r"\bsyncthing\b"),
    re.compile(r"tensorboard_data_server"),
)

# Captures the absolute path of any `/.venv/` directory's parent from a python
# invocation. Path-agnostic by design — `/workspace/<name>`, `/home/<user>/code`,
# `/tmp/build` all work. The existence check on the captured path is the actual
# safety gate (a live venv's parent dir exists; only deleted worktrees miss).
_WORKTREE_PYTHON_RE = re.compile(r"(/[A-Za-z0-9_./-]+?)/\.venv/[^\s]*\bpython")

# Game binaries that run independent of the python parent process. If a python
# parent dies, the OS reparents these to init (PPID=1) and they leak ports/GPU.
# Word boundaries prevent substring false-positives like "--game pyboy_env"
# misclassifying a live training proc as an orphan game binary.
DEFAULT_GAME_BINARY_TOKENS = ("SC2_x64", "pyboy", "gym-super-mario-bros", "burnysc2")
_GAME_BINARY_RE = re.compile(rf"\b(?:{'|'.join(DEFAULT_GAME_BINARY_TOKENS)})\b")

# Multiprocessing pool forks (resource_tracker + spawn_main).
_MULTIPROCESSING_RE = re.compile(r"multiprocessing\.(spawn|resource_tracker)")


# ── Types ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JanitorConfig:
    """Injectable knobs for the janitor's detection rules.

    Defaults match the module-level constants. Pass to `find_orphans` or
    `_kill_targets` to override per-call without monkey-patching globals —
    intended use is sweep_runner integration or per-environment policy:

        config = JanitorConfig(
            extra_game_binary_tokens=("my_emulator",),
            stale_multiprocessing_age_s=48 * 3600,  # 48h sweeps
            extra_keep_patterns=(re.compile(r"\\bdvc daemon\\b"),),
        )
        find_orphans(procs, current_pid=..., config=config)

    `extra_*` fields APPEND to defaults — callers don't have to re-declare
    the built-in tokens / patterns.
    """

    extra_keep_patterns: tuple[re.Pattern[str], ...] = ()
    extra_game_binary_tokens: tuple[str, ...] = ()
    stale_multiprocessing_age_s: int = STALE_MULTIPROCESSING_AGE_S
    sigterm_wait_s: int = SIGTERM_WAIT_S


DEFAULT_CONFIG = JanitorConfig()


class KillRule(StrEnum):
    REMOVED_WORKTREE_PYTHON = "removed_worktree_python"
    ORPHAN_GAME_BINARY = "orphan_game_binary"
    STALE_MULTIPROCESSING_POOL = "stale_multiprocessing_pool"


@dataclass
class ProcInfo:
    pid: int
    ppid: int
    etime_s: int
    cmd: str

    @property
    def etime_hours(self) -> float:
        return self.etime_s / 3600.0

    @classmethod
    def from_ps_line(cls, line: str) -> ProcInfo:
        """Parse one row of `ps -eo pid,ppid,etime,cmd --no-headers`.

        etime format: `[[DD-]HH:]MM:SS`. Examples:
          `41:32`            → 41 min 32 sec
          `15:23:11`         → 15 h 23 m 11 s
          `3-12:00:00`       → 3 days 12 h
        """
        parts = line.strip().split(None, 3)
        pid, ppid, etime, cmd = int(parts[0]), int(parts[1]), parts[2], parts[3]
        return cls(pid=pid, ppid=ppid, etime_s=_parse_etime(etime), cmd=cmd)


def _parse_etime(s: str) -> int:
    """Parse `[[DD-]HH:]MM:SS` → seconds. Raises ValueError on degenerate input."""
    days = "0"
    if "-" in s:
        days, s = s.split("-", 1)
    fields = s.split(":")
    if len(fields) == 2:  # MM:SS
        fields = ["0", *fields]
    h, m, sec = fields  # ValueError if not exactly 3 parts now
    return int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(sec)


@dataclass
class KillTarget:
    proc: ProcInfo
    rule: KillRule
    reason: str


# ── Detection (pure, testable) ──────────────────────────────────────────────


def find_orphans(
    procs: Iterable[ProcInfo],
    *,
    current_pid: int,
    worktree_exists: Callable[[Path], bool] | None = None,
    config: JanitorConfig = DEFAULT_CONFIG,
) -> list[KillTarget]:
    """Identify kill targets. Pure function — no side effects, no `ps` calls.

    Rule precedence (first match wins): removed-worktree python →
    orphan game binary → stale multiprocessing pool.

    `worktree_exists` lets tests inject a deterministic dir-present check.
    `config` overrides thresholds + extends KEEP_ALWAYS / game-binary lists.
    """
    exists = Path.exists if worktree_exists is None else worktree_exists
    keep_patterns = KEEP_ALWAYS_PATTERNS + config.extra_keep_patterns
    game_re = (
        _GAME_BINARY_RE
        if not config.extra_game_binary_tokens
        else re.compile(
            rf"\b(?:{'|'.join(DEFAULT_GAME_BINARY_TOKENS + config.extra_game_binary_tokens)})\b"
        )
    )
    stale_age = config.stale_multiprocessing_age_s

    targets: list[KillTarget] = []
    for p in procs:
        if p.pid == current_pid or p.ppid != 1 or any(pat.search(p.cmd) for pat in keep_patterns):
            continue
        if (m := _WORKTREE_PYTHON_RE.search(p.cmd)) and not exists(Path(m.group(1))):
            targets.append(
                KillTarget(p, KillRule.REMOVED_WORKTREE_PYTHON, f"worktree {m.group(1)} is gone")
            )
        elif game_re.search(p.cmd):
            targets.append(
                KillTarget(p, KillRule.ORPHAN_GAME_BINARY, "game binary reparented to PPID=1")
            )
        elif _MULTIPROCESSING_RE.search(p.cmd) and p.etime_s > stale_age:
            targets.append(
                KillTarget(
                    p,
                    KillRule.STALE_MULTIPROCESSING_POOL,
                    f"multiprocessing fork, age {p.etime_hours:.1f}h > {stale_age / 3600:.0f}h",
                )
            )
    return targets


# ── Real `ps` collection + killing ──────────────────────────────────────────


def _collect_procs() -> list[ProcInfo]:
    """Run `ps -eo pid,ppid,etime,cmd --no-headers` and parse."""
    out = subprocess.run(
        ["ps", "-eo", "pid,ppid,etime,cmd", "--no-headers"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    procs: list[ProcInfo] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            procs.append(ProcInfo.from_ps_line(line))
        except (ValueError, IndexError):
            continue
    return procs


def _kill_targets(
    targets: Iterable[KillTarget], *, config: JanitorConfig = DEFAULT_CONFIG
) -> dict[int, str]:
    """Send SIGTERM, wait, SIGKILL survivors.

    Returns `{pid: "term"|"kill"|"gone"|"denied"}`:
    - `term`: SIGTERM accepted and proc exited (or SIGTERM was enough)
    - `kill`: SIGTERM didn't take; SIGKILL fired
    - `gone`: proc was already dead when SIGTERM tried
    - `denied`: EPERM (e.g. non-root janitor against root-owned proc)
    """
    results: dict[int, str] = {}
    pids = [t.proc.pid for t in targets]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            results[pid] = "term"
        except ProcessLookupError:
            results[pid] = "gone"
        except PermissionError:
            results[pid] = "denied"
    time.sleep(config.sigterm_wait_s)
    for pid in pids:
        if results.get(pid) in ("gone", "denied"):
            continue
        try:
            os.kill(pid, 0)  # alive?
            os.kill(pid, signal.SIGKILL)
            results[pid] = "kill"
        except ProcessLookupError:
            results[pid] = "term"  # SIGTERM was enough (or proc exited mid-escalation)
        except PermissionError:
            results[pid] = "denied"
    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, help="Orphan-process janitor for autoresearch worktrees.")


@app.command()
def janitor(
    apply: bool = typer.Option(False, "--apply", help="Actually kill targets (default: dry-run)."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Audit PPID=1 procs and report (or kill) orphans."""
    procs = _collect_procs()
    targets = find_orphans(procs, current_pid=os.getpid())

    if json_output:
        payload = [
            {
                "pid": t.proc.pid,
                "ppid": t.proc.ppid,
                "etime_h": round(t.proc.etime_hours, 2),
                "rule": t.rule.value,
                "reason": t.reason,
                "cmd": t.proc.cmd[:200],
            }
            for t in targets
        ]
        if apply:
            results = _kill_targets(targets)
            for entry in payload:
                entry["action"] = results.get(entry["pid"], "skipped")
        print(json.dumps(payload, indent=2))
        return

    console = Console()
    if not targets:
        console.print("[green]No orphans detected.[/green]")
        return

    table = Table(title=f"Kill candidates ({len(targets)})", show_lines=False)
    table.add_column("PID", justify="right")
    table.add_column("Age (h)", justify="right")
    table.add_column("Rule")
    table.add_column("Reason")
    table.add_column("Cmd (truncated)")
    for t in targets:
        table.add_row(
            str(t.proc.pid),
            f"{t.proc.etime_hours:.1f}",
            t.rule.value,
            t.reason,
            t.proc.cmd[:80],
        )
    console.print(table)

    if not apply:
        console.print("\n[yellow]Dry-run.[/yellow] Re-run with [bold]--apply[/bold] to kill.")
        return

    results = _kill_targets(targets)
    counts = Counter(results.values())
    summary = " ".join(f"{k}={counts[k]}" for k in ("term", "kill", "gone", "denied") if counts[k])
    console.print(f"\n[green]Reaped:[/green] {summary or '(none)'}")


if __name__ == "__main__":
    app()
