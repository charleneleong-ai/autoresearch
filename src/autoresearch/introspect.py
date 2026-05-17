"""CLI for post-hoc trajectory introspection across multiple run dirs.

Purpose
-------
**introspect** answers: *how did the agent actually behave?*
It reads ``game_states.jsonl`` (one row per env step) from each iter dir,
fires adapter-supplied milestone / dwell / action predicates against every
row, and produces per-iter behavioural metrics: when each milestone was first
reached, how many steps the agent spent in each map zone, ``move_to``
perseveration rate, and final zone.  Run it *post-hoc* once a stage is done
to compare agent quality across stages (L vs M vs N+O etc.).

Contrast with **retrospective** (``autoresearch-retrospective``), which answers:
*did the training process fail, and how?*  It reads ``results.jsonl`` + sweep
logs *per iter during* the sweep, detects failure modes (eval plateau, silent
kill, gradient collapse), and can block the sweep early.  Different data
source, different timing, different question.

Usage::

    uv run introspect \\
        --run "L:/tmp/orak-stage-l/pokemon_red:stage_l_map_aware_iter*" \\
        --run "M:/tmp/orak-stage-m/pokemon_red:stage_m_multi_signal_iter*" \\
        --adapter agents.pokemon_red.game_adapter

    # machine-readable output for downstream scripts
    uv run introspect \\
        --run "L:/tmp/stage-l:iter*" --run "M:/tmp/stage-m:iter*" \\
        --adapter agents.pokemon_red.game_adapter \\
        --format json | jq '.[].mean_score_pct'

The adapter module must expose these module-level names::

    TRAJECTORY_MILESTONES      : list[MilestoneSpec]
    TRAJECTORY_SCORE_EXTRACTOR : Callable[[dict], float]
    TRAJECTORY_ZONE_EXTRACTOR  : Callable[[dict], str]
    TRAJECTORY_SCORE_MAX       : float
    TRAJECTORY_DWELL_SPECS     : list[DwellSpec]          (optional)
    TRAJECTORY_ACTION_SPEC     : ActionSpec               (optional)
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path
from typing import Annotated

import typer

from autoresearch.trajectory import extract_iter_metrics

app = typer.Typer(pretty_exceptions_enable=False)


def _parse_run(spec: str) -> tuple[str, Path, str]:
    """Parse ``label:dir[:glob]`` → (label, dir_path, glob_pattern)."""
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise typer.BadParameter(f"--run must be label:dir[:glob], got: {spec!r}")
    label = parts[0]
    dir_path = Path(parts[1])
    glob = parts[2] if len(parts) > 2 else "*"
    return label, dir_path, glob


def _load_adapter(module_path: str) -> dict:
    """Dynamically import adapter module and return its TRAJECTORY_* attributes."""
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        typer.echo(f"[introspect] cannot import adapter {module_path!r}: {e}", err=True)
        raise typer.Exit(1) from e
    return {
        "milestone_specs": getattr(mod, "TRAJECTORY_MILESTONES", []),
        "dwell_specs": getattr(mod, "TRAJECTORY_DWELL_SPECS", None),
        "action_spec": getattr(mod, "TRAJECTORY_ACTION_SPEC", None),
        "score_extractor": getattr(mod, "TRAJECTORY_SCORE_EXTRACTOR", lambda _: 0.0),
        "zone_extractor": getattr(mod, "TRAJECTORY_ZONE_EXTRACTOR", lambda _: "?"),
        "score_max": getattr(mod, "TRAJECTORY_SCORE_MAX", 1.0),
    }


@app.command()
def main(
    runs: Annotated[
        list[str],
        typer.Option("--run", help="label:dir[:glob]  (repeatable)"),
    ],
    adapter: Annotated[
        str,
        typer.Option(help="Python module path exposing TRAJECTORY_* constants"),
    ],
    fmt: Annotated[
        str,
        typer.Option("--format", help="Output format: 'text' (default) or 'json'"),
    ] = "text",
) -> None:
    """Print a per-stage comparison table of iter-level trajectory metrics."""
    if fmt not in ("text", "json"):
        typer.echo(f"[introspect] unknown --format {fmt!r}; choose 'text' or 'json'", err=True)
        raise typer.Exit(1)

    kwargs = _load_adapter(adapter)
    output: list[dict] = []

    for spec in runs:
        label, base, glob = _parse_run(spec)
        iter_dirs = sorted(d for d in base.glob(glob) if d.is_dir())
        rows = [extract_iter_metrics(d, **kwargs) for d in iter_dirs]
        milestone_names = list(rows[0].first_milestone_step.keys()) if rows else []
        scores = [r.score_pct for r in rows if not r.error]

        if fmt == "json":
            output.append(
                {
                    "label": label,
                    "base": str(base),
                    "iters": [dataclasses.asdict(r) for r in rows],
                    "mean_score_pct": round(sum(scores) / len(scores), 2) if scores else None,
                }
            )
            continue

        # ── text output ──────────────────────────────────────────────────
        typer.echo(f"\n══════ {label} ({base}) ══════")
        if not iter_dirs:
            typer.echo("  (no iter dirs found)")
            continue

        for r in rows:
            if r.error:
                typer.echo(f"  {r.run_id}: {r.error}")
                continue

            ms_parts = " ".join(
                f"{n}@{r.first_milestone_step[n]}"
                if r.first_milestone_step[n] is not None
                else f"{n}@n/a"
                for n in milestone_names
            )
            dw_parts = " ".join(f"{k}={v:>3}" for k, v in r.dwell_counts.items())
            name = r.run_id
            iter_num = "?"
            for part in name.split("iter"):
                candidate = part.lstrip("_").split("_")[0].split("/")[0]
                if candidate.isdigit():
                    iter_num = candidate
                    break

            typer.echo(
                f"  iter {iter_num:>2}: "
                f"final={r.score_pct:5.2f}% "
                + (f"{ms_parts}  " if ms_parts else "")
                + (f"{dw_parts}  " if dw_parts else "")
                + f"actions={r.action_count:>3} "
                f"persev={r.perseveration_pct:>4.1f}% "
                f"zone={r.final_zone}"
            )

        if scores:
            typer.echo(f"\n  scores={scores}  mean={sum(scores) / len(scores):.2f}%")

    if fmt == "json":
        typer.echo(json.dumps(output, indent=2))


def cli() -> None:
    app()


if __name__ == "__main__":
    app()
