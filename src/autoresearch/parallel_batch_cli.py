"""CLI for ``autoresearch-parallel-batch``.

Reads a YAML batch spec and orchestrates parallel multi-slot launches via
:func:`autoresearch.parallel_batch.run_parallel_batch`. The YAML describes
slots (label, tag, command, n_runs, completion-file glob) and batch-wide
knobs (max_parallel, retries, poll interval).

A *completion file* convention is built in: each slot points at a glob that
matches its output directory; once a JSON file (default
``evaluation_summary.json``) appears inside that directory, it's parsed and
the score is extracted via a fallback chain of JSON keys.

For non-file-based completion (e.g. parsing stdout, checking a database
row), use the Python library directly — pass a custom ``completion_check``
callable to :func:`run_parallel_batch`.

Example schedule yaml (configs/schedules/repro_n3.yaml)::

    experiments_dir: ./experiments
    max_parallel: 8
    max_attempts_per_run: 3
    poll_interval_s: 60
    launch_stagger_s: 30
    slots:
      - label: baseline_mario
        tag: tgaer_pr1_baseline
        game: super_mario
        n_runs: 3
        cwd: /workspace/orak-master-baselines
        log_dir: /workspace/orak-master-baselines/logs
        command:
          - ./.venv/bin/python
          - run.py
          - -c
          - gemma_26b
          - --local
          - --games
          - super_mario
        completion:
          glob: "game_logs/super_mario/baseline_mario_*"
          file: evaluation_summary.json
          score_keys: [mean_score, score, evaluation_score]

Run with::

    autoresearch-parallel-batch run configs/schedules/repro_n3.yaml
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
import yaml

from .parallel_batch import SlotSpec, run_parallel_batch

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _build_completion_check(slot_cfg: dict[str, Any]):
    """Build a completion_check that looks for a JSON file inside a glob match."""
    completion = slot_cfg.get("completion") or {}
    if not completion:
        return None

    glob_pattern = completion["glob"]
    filename = completion.get("file", "evaluation_summary.json")
    score_keys = completion.get("score_keys", ["score", "evaluation_score", "mean_score"])
    cwd = Path(slot_cfg.get("cwd") or ".")

    def check(slot: SlotSpec, run_idx: int) -> dict[str, Any] | None:
        # Resolve the glob against cwd (the slot's working dir, where outputs land)
        matches = sorted(cwd.glob(glob_pattern))
        if not matches:
            return None
        # Pick the run_idx-th match if available (stable ordering by mtime)
        matches = sorted(matches, key=lambda p: p.stat().st_mtime)
        # Find an output dir whose summary file we haven't recorded yet — naive:
        # take the run_idx-th newest. For our needs this is good enough.
        if run_idx >= len(matches):
            return None
        output_dir = matches[run_idx]
        summary_file = output_dir / filename
        if not summary_file.exists():
            return None
        try:
            data = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            return None

        score = 0.0
        for key in score_keys:
            if key in data:
                score = float(data[key])
                break

        return {
            "score": score,
            "extra": {
                "run_id": output_dir.name,
                "evaluation_summary": data,
            },
            "description": f"{slot.label} r{run_idx} ({output_dir.name})",
        }

    return check


def _make_slot(slot_cfg: dict[str, Any]) -> SlotSpec:
    """Construct a SlotSpec from a YAML slot dict."""
    return SlotSpec(
        label=slot_cfg["label"],
        tag=slot_cfg["tag"],
        command=slot_cfg["command"],
        n_runs=slot_cfg.get("n_runs", 3),
        game=slot_cfg.get("game"),
        cwd=Path(slot_cfg["cwd"]) if slot_cfg.get("cwd") else None,
        env={**os.environ, **slot_cfg["env"]} if slot_cfg.get("env") else None,
        log_dir=Path(slot_cfg["log_dir"]) if slot_cfg.get("log_dir") else None,
        completion_check=_build_completion_check(slot_cfg),
        extra=slot_cfg.get("extra") or {},
    )


@app.command()
def run(
    spec: Path = typer.Argument(..., help="YAML batch spec"),
    experiments_dir: Path = typer.Option(
        None,
        help="Where per-tag results.jsonl files live. Defaults to spec.experiments_dir.",
    ),
    max_parallel: int = typer.Option(None, help="Override spec.max_parallel"),
    max_attempts_per_run: int = typer.Option(None, help="Override spec.max_attempts_per_run"),
    poll_interval_s: float = typer.Option(None, help="Override spec.poll_interval_s"),
    launch_stagger_s: float = typer.Option(None, help="Override spec.launch_stagger_s"),
    dry_run: bool = typer.Option(
        False, help="Parse the spec and print what would be launched; don't launch."
    ),
) -> None:
    """Run a parallel-batch from a YAML spec."""
    cfg = yaml.safe_load(spec.read_text())
    slots = [_make_slot(s) for s in cfg.get("slots", [])]
    if not slots:
        typer.echo("no slots in spec — exiting", err=True)
        raise typer.Exit(code=1)

    kwargs = {
        "experiments_dir": experiments_dir or Path(cfg.get("experiments_dir", "./experiments")),
        "max_parallel": max_parallel or cfg.get("max_parallel", 8),
        "max_attempts_per_run": max_attempts_per_run or cfg.get("max_attempts_per_run", 3),
        "poll_interval_s": poll_interval_s or cfg.get("poll_interval_s", 60.0),
        "launch_stagger_s": launch_stagger_s
        if launch_stagger_s is not None
        else cfg.get("launch_stagger_s", 10.0),
    }

    typer.echo(
        f"parallel-batch: {len(slots)} slots, total replicates {sum(s.n_runs for s in slots)}"
    )
    for s in slots:
        typer.echo(f"  • {s.label} → tag={s.tag} game={s.game} n_runs={s.n_runs}")
    typer.echo(
        f"  max_parallel={kwargs['max_parallel']}  "
        f"max_attempts={kwargs['max_attempts_per_run']}  "
        f"poll={kwargs['poll_interval_s']}s  stagger={kwargs['launch_stagger_s']}s"
    )
    typer.echo(f"  experiments_dir={kwargs['experiments_dir']}")

    if dry_run:
        typer.echo("\n[dry-run] not launching.")
        return

    result = run_parallel_batch(slots, **kwargs, log=lambda m: typer.echo(f"  {m}"))

    typer.echo(
        f"\ndone: completed={len(result.completed)} "
        f"gave_up={len(result.gave_up)} elapsed={result.elapsed_s:.0f}s"
    )
    if result.gave_up:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
