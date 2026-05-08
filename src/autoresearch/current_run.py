"""In-flight RUNNING-dot daemon for the autoresearch chart.

Watches the latest `logs/autoresearch_*T*Z.log` and keeps
`experiments/<TAG>[/<config>]/current_run.json` in sync with whichever
iteration is currently in flight, so a chart rendered against
`results.jsonl` can show an extra RUNNING dot for the in-flight iter
without waiting for the iter to finish.

Behavior:
- Find the most recent iter-start line matching the active log format's
  `iter_start_re`. If any `iter_done_re` match appears positionally
  AFTER it, the iter is done — drop the sidecar. Otherwise — write it.

The sidecar payload includes:
  experiment       — index (count of existing results.jsonl rows)
  config_name      — passed through
  description      — parsed via the format's `desc_re` if set, else the
                     iter line's `rest` capture, else `"iter N/M"`
  notes            — same as description (chart compatibility)
  started_at       — timestamp from the iter line's `ts` capture (default
                     format only — omitted for formats without timestamps)
  log_path         — path of the autoresearch log being watched
  iter_marker      — "Iter N/M" string
  wandb_url        — last wandb.ai URL spotted in this iter's log chunk

## Log formats

- `default` — gemma4-style `[YYYY-MM-DDTHH:MM:SSZ] Iter N/M: rest` with
  per-iter `Iter N/M finished` end markers and `$ ... -d "<desc>" ...`
  command-line descriptions.
- `untimed` — orak-style `# Iteration N/M` (no timestamp wrapper) with a
  sweep-wide `Autoresearch complete after N iterations` end marker and
  dedicated `Description: <desc>` lines.

Pass via `--log-format` on the CLI.

Designed to run detached (`setsid + nohup + disown`) so it survives
SSH / coding-agent session death — verify `PPID=1` after launch. Use
`python -u` or `flush=True` on prints so the live log isn't silently
buffered.

Usage:
    autoresearch-current-run --tag <task> [--config <name>] \\
        [--logs-dir logs] [--log-format default|untimed]
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from autoresearch.results import tag_dir


@dataclass(frozen=True)
class LogFormat:
    """Regex patterns for parsing iter starts, completion, and descriptions.

    `iter_start_re` MUST capture named groups `n` and `m`. `ts` and `rest`
    are optional — `ts` populates `started_at`, `rest` is the description
    fallback when no `desc_re` is set.

    `iter_done_re` only needs to MATCH positionally after the latest
    `iter_start_re` match — captures are not used. This lets per-iter
    end markers (default format) and sweep-wide markers (untimed) share
    the same dispatch path.

    `desc_re`, when set, is searched in the chunk after the latest
    iter-start; its `desc` named group becomes the sidecar description.
    """

    iter_start_re: re.Pattern[str]
    iter_done_re: re.Pattern[str]
    desc_re: re.Pattern[str] | None = None


LOG_FORMATS: dict[str, LogFormat] = {
    "default": LogFormat(
        iter_start_re=re.compile(
            r"\[(?P<ts>[\d\-T:Z]+)\] Iter (?P<n>\d+)/(?P<m>\d+): (?P<rest>.*)"
        ),
        iter_done_re=re.compile(r"\[[\d\-T:Z]+\] Iter \d+/\d+ finished"),
        desc_re=re.compile(r"-d (?P<desc>\[autoresearch [^\]]+\][^-]+?)(?= --|$)"),
    ),
    "untimed": LogFormat(
        iter_start_re=re.compile(r"#\s*Iteration (?P<n>\d+)/(?P<m>\d+)"),
        iter_done_re=re.compile(r"Autoresearch complete after \d+ iterations"),
        desc_re=re.compile(r"Description: (?P<desc>.+)"),
    ),
}

WANDB_RE = re.compile(r"https://wandb\.ai/[\w\-./]+/runs/[\w\-]+")


def _latest_log(logs_dir: Path) -> Path | None:
    # Match training stdout logs only (timestamped name); skip the sidecar
    # `autoresearch_events.log` which has no step_time / `$ ...` lines.
    logs = sorted(logs_dir.glob("autoresearch_*T*Z.log"))
    return logs[-1] if logs else None


def _experiment_count(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    return sum(1 for line in results_path.read_text().splitlines() if line.strip())


def _resolve_description(
    text: str, last: re.Match[str], fmt: LogFormat, iter_n: int, iter_m: int
) -> str:
    """Extract description from `text` after the latest iter-start match.

    Tries `fmt.desc_re` first, then the iter line's `rest` capture, then
    falls back to a generic `iter N/M` label.
    """
    if fmt.desc_re is not None:
        m_desc = fmt.desc_re.search(text, pos=last.end())
        if m_desc is not None:
            return m_desc.group("desc").strip()
    try:
        rest = last.group("rest")
    except IndexError:  # `rest` not in pattern — formats may omit it
        rest = ""
    return rest.strip() if rest else f"iter {iter_n}/{iter_m}"


def _tick(
    logs_dir: Path,
    sidecar: Path,
    results_path: Path,
    config_name: str | None,
    fmt: LogFormat,
) -> None:
    log = _latest_log(logs_dir)
    if log is None:
        return
    text = log.read_text(errors="replace")

    starts = list(fmt.iter_start_re.finditer(text))
    if not starts:
        return

    last = starts[-1]
    iter_n = int(last.group("n"))
    iter_m = int(last.group("m"))

    # Iter is done if any done-marker (per-iter or sweep-wide) appears
    # positionally AFTER the latest iter-start match.
    if fmt.iter_done_re.search(text, pos=last.end()) is not None:
        if sidecar.exists():
            sidecar.unlink()
            rprint(
                f"\\[current_run] iter {iter_n}/{iter_m} [green]finished[/green] — sidecar removed"
            )
        return

    desc = _resolve_description(text, last, fmt, iter_n, iter_m)

    # Wandb URL inside this iter's chunk
    chunk = text[last.start() :]
    urls = WANDB_RE.findall(chunk)
    wandb_url = urls[-1] if urls else ""

    payload: dict[str, Any] = {
        "experiment": _experiment_count(results_path),
        "config_name": config_name or "",
        "description": desc,
        "notes": desc,
        "log_path": str(log),
        "iter_marker": f"Iter {iter_n}/{iter_m}",
        "wandb_url": wandb_url,
    }
    # Only include `started_at` when the format captures a timestamp.
    try:
        started_at = last.group("ts")
    except IndexError:
        started_at = None
    if started_at:
        if not started_at.endswith("Z") and "+" not in started_at:
            started_at += "Z"
        payload["started_at"] = started_at

    if sidecar.exists():
        try:
            cur = json.loads(sidecar.read_text())
            if cur == payload:
                return  # no change — avoid touching mtime every poll
        except json.JSONDecodeError:
            pass
    sidecar.write_text(json.dumps(payload, indent=2))
    wandb_label = wandb_url or "[dim]pending[/dim]"
    rprint(
        f"\\[current_run] sidecar → iter [bold]{iter_n}/{iter_m}[/bold] "
        f"(E{payload['experiment']}, wandb={wandb_label})"
    )


def main(
    tag: str = typer.Option(..., "--tag", help="Top-level task/sweep tag (e.g. 'dd_explainer')"),
    config_name: str | None = typer.Option(
        None, "--config", help="Per-config sub-dir for multi-sweep isolation"
    ),
    experiments_dir: Path = typer.Option(Path("experiments"), "--experiments-dir"),
    logs_dir: Path = typer.Option(Path("logs"), "--logs-dir"),
    log_format: str = typer.Option(
        "default",
        "--log-format",
        help=(
            "Log format preset: 'default' (timestamped `[ts] Iter N/M: ...` + "
            "per-iter `Iter N/M finished`) or 'untimed' (orak-style "
            "`# Iteration N/M` + sweep-wide `Autoresearch complete...`)"
        ),
    ),
    poll_s: int = typer.Option(
        15,
        "--poll-s",
        envvar="AUTORESEARCH_CURRENT_RUN_POLL_S",
        help="Seconds between ticks (default 15)",
    ),
) -> None:
    if log_format not in LOG_FORMATS:
        raise typer.BadParameter(
            f"unknown --log-format {log_format!r}; choose from {sorted(LOG_FORMATS)}"
        )
    fmt = LOG_FORMATS[log_format]

    logs_dir = logs_dir.resolve()
    target_dir = tag_dir(experiments_dir, tag, config_name)
    sidecar = target_dir / "current_run.json"
    results_path = target_dir / "results.jsonl"

    rprint(
        f"[bold cyan]\\[current_run][/bold cyan] starting — poll every {poll_s}s\n"
        f"  logs_dir={logs_dir}  sidecar={sidecar}  log_format={log_format}"
    )
    while True:
        try:
            _tick(logs_dir, sidecar, results_path, config_name, fmt)
        except Exception as e:
            rprint(f"[red]\\[current_run][/red] tick error: {e}")
        time.sleep(poll_s)


def cli() -> None:
    """Entry-point wrapper so the console script (`autoresearch-current-run`)
    runs `main` through typer's argument parser."""
    typer.run(main)


# ── In-loop sidecar writer (alternative to the log-watcher daemon above) ───
#
# The functions below let a sweep loop write/clear the `current_run.json`
# sidecar directly at known transition points (iter start / iter end /
# crash). They're the per-project equivalent of orak's `_write_sidecar` /
# `_clear_sidecar` helpers, extracted into the package.
#
# Use these instead of the log-watcher daemon when:
#   - The loop already knows what iter it's on (no need to infer from logs)
#   - You want guaranteed cleanup (the context manager handles unlink-on-exit)
#   - You don't want to run a separate daemon process
#
# Both writers produce the same JSON shape so chart renderers see a
# consistent payload either way.


def write_sidecar(
    payload: dict[str, Any],
    *,
    tag: str | None = None,
    config_name: str | None = None,
    experiments_dir: str | Path = "experiments",
) -> Path:
    """Write ``payload`` to ``experiments/<tag>[/<config_name>]/current_run.json``.

    Free-form payload — pass whatever fields your chart renderer expects.
    Common keys (used by both downstream consumers and the daemon above):
    ``experiment``, ``config_name``, ``description``, ``notes``,
    ``started_at``, ``log_path``, ``iter_marker``, ``wandb_url``.

    Returns the path written.
    """
    sidecar = tag_dir(experiments_dir, tag, config_name) / "current_run.json"
    sidecar.write_text(json.dumps(payload, indent=2))
    return sidecar


def clear_sidecar(
    *,
    tag: str | None = None,
    config_name: str | None = None,
    experiments_dir: str | Path = "experiments",
) -> bool:
    """Remove the sidecar if present. Returns True if a file was actually unlinked."""
    sidecar = tag_dir(experiments_dir, tag, config_name) / "current_run.json"
    if sidecar.exists():
        sidecar.unlink()
        return True
    return False


@contextmanager
def sidecar(
    payload: dict[str, Any],
    *,
    tag: str | None = None,
    config_name: str | None = None,
    experiments_dir: str | Path = "experiments",
) -> Iterator[Path]:
    """Write the sidecar on enter, unlink on exit (success OR exception).

    The intended in-loop usage:

        for plan in planner.plan_iters(history):
            with sidecar(
                {"experiment": i, "description": plan.description, ...},
                tag=tag, config_name=plan.config_name,
            ):
                proc = subprocess.Popen(plan.cmd, ...)
                ret, kill_reason = wait_with_timeout(proc, timeout_s=...)
            # sidecar unlinked here regardless of how the iter ended

    Without the context manager, projects routinely forget to clean up after
    KeyboardInterrupt or a crash mid-iter; stale sidecars then poison the
    chart's "RUNNING dot" indicator until manually deleted.
    """
    path = write_sidecar(
        payload,
        tag=tag,
        config_name=config_name,
        experiments_dir=experiments_dir,
    )
    try:
        yield path
    finally:
        clear_sidecar(tag=tag, config_name=config_name, experiments_dir=experiments_dir)


if __name__ == "__main__":
    cli()
