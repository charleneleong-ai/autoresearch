"""Per-sweep markdown writeup scaffolder.

Emits a skeleton for `docs/experiments/<config_name>/<schedule_name>.md` from
a schedule yaml + the matching `results.jsonl` rows. The author fills in the
hypothesis, verdict, and next-move sections; the boilerplate (schedule yaml
inline, per-iter results table, runtime accounting) is generated.

Convention:

    docs/experiments/
    └── <config_name>/
        └── <schedule_name>.md   # one writeup per configs/schedules/*.yaml

Mirrors the `experiments/<tag>/<config_name>/results.jsonl` layout from
`autoresearch.results.tag_dir`, so any artefact (results row, schedule yaml,
doc) maps cleanly to the other two.

Usage:

    uv run autoresearch-report \\
      --schedule configs/schedules/v2_granular_no_halluc.yaml \\
      --config train_v2_80gb \\
      --tag dd_explainer \\
      --experiments-dir experiments \\
      --out docs/experiments/train_v2_80gb/v2_granular_no_halluc.md

If `--out` exists the command refuses to overwrite (safe to re-run after
manual edits). Pass `--force` to regenerate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _format_iter_overrides(overrides: list[str]) -> str:
    """Pretty-print a flat list of `[--k, v, --k, v]` as `--k v --k v`."""
    return " ".join(str(x) for x in overrides) if overrides else "(none)"


def _format_results_row(row: dict) -> dict:
    """Pull the fields a sweep writeup needs from a results.jsonl row."""
    h = row.get("metrics", {}).get("heldout", {})
    return {
        "experiment": row.get("experiment", "?"),
        "status": row.get("status", "?"),
        "steps": row.get("steps", "?"),
        "runtime_min": row.get("runtime_min", 0.0),
        "score": row.get("score", "?"),
        "mean_total": h.get("mean_total", "—"),
        "f1": h.get("f1_triggers_mean", "—"),
        "no_halluc": h.get("no_hallucinated_facts_mean", "—"),
        "well_formed": h.get("well_formed_mean", "—"),
        "pass_all_pct": h.get("pass_all_pct", "—"),
        "description": row.get("description", "").lstrip("[early_stopped] "),
    }


def _render_skeleton(
    schedule_name: str,
    schedule_data: dict,
    config_name: str,
    results: list[dict],
) -> str:
    iters = schedule_data.get("iters", [])
    common = schedule_data.get("common_overrides", [])
    schedule_yaml = yaml.safe_dump(schedule_data, sort_keys=False, default_flow_style=False).rstrip()

    header_section = f"""# `{schedule_name}` — <one-line hypothesis here>

**Schedule:** [`configs/schedules/{schedule_name}.yaml`](../../../configs/schedules/{schedule_name}.yaml)
**Config:** `{config_name}`
**Iterations:** {len(iters)} iters{f' · {len(results)} rows logged' if results else ''}
**Started:** <UTC timestamp> · **Finished:** <UTC timestamp> (<duration>)

## Hypothesis

<What mechanism is this sweep testing? What outcome would falsify the hypothesis?>

## Schedule

```yaml
{schedule_yaml}
```

`common_overrides`: `{_format_iter_overrides(common)}`

## Pre-launch comparisons

<Reference rows from prior sweeps that this one is anchored against. Pull from
`docs/ceiling-diagnosis-*.md` or the `results.jsonl` directly.>

| anchor | mean_total | f1 | no_halluc |
|---|---|---|---|
| <prior exp> | — | — | — |
"""

    if results:
        rows = [_format_results_row(r) for r in results[-len(iters):]]
        results_table = "\n## Results\n\n| iter | exp | steps | runtime | mean_total | f1 | no_halluc | well_formed | pass_all |\n|---|---|---|---|---|---|---|---|---|\n"
        for i, r in enumerate(rows, 1):
            rt = f"{r['runtime_min']:.1f}m" if isinstance(r["runtime_min"], (int, float)) else r["runtime_min"]
            mt = f"{r['mean_total']:.3f}" if isinstance(r["mean_total"], (int, float)) else r["mean_total"]
            f1 = f"{r['f1']:.3f}" if isinstance(r["f1"], (int, float)) else r["f1"]
            nh = f"{r['no_halluc']:.3f}" if isinstance(r["no_halluc"], (int, float)) else r["no_halluc"]
            wf = f"{r['well_formed']:+.3f}" if isinstance(r["well_formed"], (int, float)) else r["well_formed"]
            pa = f"{r['pass_all_pct']}%" if isinstance(r["pass_all_pct"], (int, float)) else r["pass_all_pct"]
            results_table += f"| {i}/{len(iters)} | E{r['experiment']} | {r['steps']} | {rt} | {mt} | {f1} | {nh} | {wf} | {pa} |\n"
    else:
        results_table = "\n## Results\n\n_No results.jsonl rows found yet — fill in once the sweep finishes._\n"

    verdict_section = """
## Verdict

<Did the hypothesis hold? Reference specific iter numbers. Use ✓/✗ to mark
sub-claims so future readers can scan.>

## Next move

<Pointer to the next sweep yaml + writeup, or a "ceiling reached, pivot to X"
note. Cross-link to the diagnosis doc for the cross-sweep narrative.>
"""

    return header_section + results_table + verdict_section


@app.command()
def main(
    schedule: Path = typer.Option(..., help="Path to configs/schedules/<name>.yaml"),
    config: str = typer.Option(..., help="Training config name (e.g. train_v2_80gb)"),
    tag: str = typer.Option(..., help="Task tag (e.g. dd_explainer)"),
    experiments_dir: Path = typer.Option(Path("experiments"), help="Root experiments dir"),
    out: Path = typer.Option(..., help="Output markdown path"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing file"),
) -> None:
    """Emit a per-sweep writeup skeleton.

    The skeleton inlines the schedule yaml and the per-iter results table,
    leaves Hypothesis / Pre-launch comparisons / Verdict / Next move blank
    for the author to fill.
    """
    if not schedule.exists():
        raise typer.BadParameter(f"schedule not found: {schedule}")
    if out.exists() and not force:
        raise typer.BadParameter(f"{out} exists — pass --force to overwrite")

    schedule_data = yaml.safe_load(schedule.read_text())
    schedule_name = schedule.stem

    from autoresearch.results import load_results

    results = load_results(experiments_dir=experiments_dir, tag=tag, config_name=config)

    body = _render_skeleton(
        schedule_name=schedule_name,
        schedule_data=schedule_data,
        config_name=config,
        results=results,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    typer.echo(f"wrote {out}  ({len(body):,} chars, {len(results)} results rows)")


if __name__ == "__main__":
    app()
