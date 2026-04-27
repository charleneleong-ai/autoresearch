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

Designed to run detached (`setsid + nohup + disown`) so it survives SSH /
Claude Code session death — verify `PPID=1` after launch. Use `python -u`
or `flush=True` on prints so the live log isn't silently buffered.

Usage:
    autoresearch-current-run --tag <task> [--config <name>] [--logs-dir logs]
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import typer

from autoresearch.results import tag_dir

ITER_START_RE = re.compile(
    r"\[(?P<ts>[\d\-T:Z]+)\] Iter (?P<n>\d+)/(?P<m>\d+): (?P<rest>.*)"
)
ITER_END_RE = re.compile(r"\[[\d\-T:Z]+\] Iter (?P<n>\d+)/(?P<m>\d+) finished")
DESC_RE = re.compile(r"-d (\[autoresearch [^\]]+\][^-]+?)(?= --|$)")
WANDB_RE = re.compile(r"https://wandb\.ai/[\w\-./]+/runs/[\w\-]+")


def _latest_log(logs_dir: Path) -> Optional[Path]:
    # Match training stdout logs only (timestamped name); skip the sidecar
    # `autoresearch_events.log` which has no step_time / `$ ...` lines.
    logs = sorted(logs_dir.glob("autoresearch_*T*Z.log"))
    return logs[-1] if logs else None


def _experiment_count(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    return sum(1 for line in results_path.read_text().splitlines() if line.strip())


def _tick(logs_dir: Path, sidecar: Path, results_path: Path,
          config_name: Optional[str]) -> None:
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
            print(f"[current_run] iter {iter_n}/{iter_m} finished — sidecar removed",
                  flush=True)
        return

    # Pull description from the `$ ... -d <desc> --max-steps ...` line that
    # follows the Iter line.
    after_iter = text[last.end():]
    cmd_line = next((ln for ln in after_iter.splitlines() if ln.startswith("$ ")), "")
    m_desc = DESC_RE.search(cmd_line)
    desc = m_desc.group(1).strip() if m_desc else last.group("rest").strip()

    # Wandb URL inside this iter's chunk
    chunk = text[last.start():]
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
    print(
        f"[current_run] sidecar → iter {iter_n}/{iter_m} "
        f"(E{payload['experiment']}, wandb={wandb_url or 'pending'})",
        flush=True,
    )


def main(
    tag: str = typer.Option(..., "--tag", help="Top-level task/sweep tag (e.g. 'dd_explainer')"),
    config_name: Optional[str] = typer.Option(
        None, "--config", help="Per-config sub-dir for multi-sweep isolation"
    ),
    experiments_dir: Path = typer.Option(Path("experiments"), "--experiments-dir"),
    logs_dir: Path = typer.Option(Path("logs"), "--logs-dir"),
    poll_s: int = typer.Option(15, "--poll-s"),
) -> None:
    logs_dir = logs_dir.resolve()
    target_dir = tag_dir(experiments_dir, tag, config_name)
    sidecar = target_dir / "current_run.json"
    results_path = target_dir / "results.jsonl"

    print(
        f"[current_run] starting — poll every {poll_s}s\n"
        f"  logs_dir={logs_dir}  sidecar={sidecar}",
        flush=True,
    )
    while True:
        try:
            _tick(logs_dir, sidecar, results_path, config_name)
        except Exception as e:
            print(f"[current_run] tick error: {e}", flush=True)
        time.sleep(poll_s)


def cli() -> None:
    """Entry-point wrapper so the console script (`autoresearch-current-run`)
    runs `main` through typer's argument parser."""
    typer.run(main)


if __name__ == "__main__":
    cli()
