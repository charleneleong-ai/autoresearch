"""In-flight RUNNING-dot daemon for the autoresearch chart.

Watches the latest `logs/autoresearch_*T*Z.log` and keeps
`experiments/<TAG>[/<config>]/current_run.json` in sync with whichever
iteration is currently in flight, so a chart rendered against
`results.jsonl` can show an extra RUNNING dot for the in-flight iter
without waiting for the iter to finish.

Behavior:
- On the most recent `Iter N/M: ...` line with no matching `Iter N/M
  finished ...` below it → write the sidecar JSON.
- On `Iter N/M finished ...` → delete the sidecar.

The sidecar payload includes:
  experiment       — index (count of existing results.jsonl rows)
  config_name      — passed through
  description      — parsed from the `$ ... -d "<desc>" ...` line
  notes            — same as description (chart compatibility)
  started_at       — timestamp from the Iter log line
  log_path         — path of the autoresearch log being watched
  iter_marker      — "Iter N/M" string
  wandb_url        — last wandb.ai URL spotted in this iter's log chunk

Designed to run detached (`setsid + nohup + disown`) so it survives
SSH / coding-agent session death — verify `PPID=1` after launch. Use
`python -u` or `flush=True` on prints so the live log isn't silently
buffered.

Usage:
    autoresearch-current-run --tag <task> [--config <name>] [--logs-dir logs]
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from autoresearch.results import tag_dir

ITER_START_RE = re.compile(r"\[(?P<ts>[\d\-T:Z]+)\] Iter (?P<n>\d+)/(?P<m>\d+): (?P<rest>.*)")
ITER_END_RE = re.compile(r"\[[\d\-T:Z]+\] Iter (?P<n>\d+)/(?P<m>\d+) finished")
DESC_RE = re.compile(r"-d (\[autoresearch [^\]]+\][^-]+?)(?= --|$)")
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


def _tick(logs_dir: Path, sidecar: Path, results_path: Path, config_name: str | None) -> None:
    log = _latest_log(logs_dir)
    if log is None:
        return
    text = log.read_text(errors="replace")

    starts = list(ITER_START_RE.finditer(text))
    ends = {int(m.group("n")) for m in ITER_END_RE.finditer(text)}
    if not starts:
        return

    last = starts[-1]
    iter_n = int(last.group("n"))
    iter_m = int(last.group("m"))

    if iter_n in ends:
        # Latest started iter has finished — drop the sidecar.
        if sidecar.exists():
            sidecar.unlink()
            rprint(
                f"\\[current_run] iter {iter_n}/{iter_m} [green]finished[/green] — sidecar removed"
            )
        return

    # Pull description from the `$ ... -d <desc> --max-steps ...` line that
    # follows the Iter line.
    after_iter = text[last.end() :]
    cmd_line = next((ln for ln in after_iter.splitlines() if ln.startswith("$ ")), "")
    m_desc = DESC_RE.search(cmd_line)
    desc = m_desc.group(1).strip() if m_desc else last.group("rest").strip()

    # Wandb URL inside this iter's chunk
    chunk = text[last.start() :]
    urls = WANDB_RE.findall(chunk)
    wandb_url = urls[-1] if urls else ""

    # Iter timestamp from the log line
    started_at = last.group("ts")
    if not started_at.endswith("Z") and "+" not in started_at:
        started_at += "Z"

    payload = {
        "experiment": _experiment_count(results_path),
        "config_name": config_name or "",
        "description": desc,
        "notes": desc,
        "started_at": started_at,
        "log_path": str(log),
        "iter_marker": f"Iter {iter_n}/{iter_m}",
        "wandb_url": wandb_url,
    }

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
    poll_s: int = typer.Option(
        15,
        "--poll-s",
        envvar="AUTORESEARCH_CURRENT_RUN_POLL_S",
        help="Seconds between ticks (default 15)",
    ),
) -> None:
    logs_dir = logs_dir.resolve()
    target_dir = tag_dir(experiments_dir, tag, config_name)
    sidecar = target_dir / "current_run.json"
    results_path = target_dir / "results.jsonl"

    rprint(
        f"[bold cyan]\\[current_run][/bold cyan] starting — poll every {poll_s}s\n"
        f"  logs_dir={logs_dir}  sidecar={sidecar}"
    )
    while True:
        try:
            _tick(logs_dir, sidecar, results_path, config_name)
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
