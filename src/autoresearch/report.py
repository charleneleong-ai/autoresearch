"""Per-sweep markdown writeup scaffolder.

Emits a skeleton for `docs/experiments/<tag>/<config_name>/<schedule_name>.md`
from a schedule yaml + the matching `results.jsonl` rows. The author fills in
the hypothesis, verdict, and next-move sections; the boilerplate (schedule
yaml inline, per-iter results table, runtime accounting) is generated.

Convention:

    docs/experiments/
    └── <tag>/
        └── <config_name>/
            └── <schedule_name>.md   # one writeup per configs/schedules/*.yaml

Mirrors the `experiments/<tag>/<config_name>/results.jsonl` layout from
`autoresearch.results.tag_dir` 1:1 — `<tag>`, `<config_name>`, and
`<schedule_name>` are the three keys you can trace from any artefact (results
row, schedule yaml, runtime chart, doc) to the others.

Usage:

    uv run autoresearch-report \\
      --schedule configs/schedules/v2_granular_no_halluc.yaml \\
      --config train_v2_80gb \\
      --tag dd_explainer \\
      --experiments-dir experiments \\
      --out docs/experiments/dd_explainer/train_v2_80gb/v2_granular_no_halluc.md

If `--out` exists the command refuses to overwrite (safe to re-run after
manual edits). Pass `--force` to regenerate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer
import yaml

from autoresearch.results import load_results

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Chassis fields we look up from a training config + its Hydra defaults chain.
# Probed under both `train.<key>` (Hydra structured config) and the top level
# (flat config). Missing fields are silently dropped from the chassis line.
_CHASSIS_FIELDS = ("model_name", "lora_rank", "max_seq_length", "num_generations")


def _format_iter_overrides(overrides: list[str]) -> str:
    """Pretty-print a flat list of `[--k, v, --k, v]` as `--k v --k v`."""
    return " ".join(str(x) for x in overrides) if overrides else "(none)"


def _resolve_hydra_config(config_yaml: Path) -> dict[str, Any]:
    """Load `config_yaml` and shallow-merge any Hydra `defaults:` chain.

    Implements just enough of Hydra's defaults resolution to extract the
    chassis fields — child config overrides parent. Skips `_self_` and
    `???` entries; doesn't handle package overrides (`@pkg/foo`) since the
    chassis fields we look up are stable across that.
    """
    if not config_yaml.exists():
        return {}
    data = yaml.safe_load(config_yaml.read_text()) or {}
    defaults = data.get("defaults", []) or []
    merged: dict[str, Any] = {}
    for entry in defaults:
        if entry in (None, "_self_") or (isinstance(entry, str) and entry.startswith("???")):
            continue
        name = entry if isinstance(entry, str) else next(iter(entry.values()), None)
        if not name:
            continue
        parent_path = config_yaml.parent / f"{name}.yaml"
        if parent_path.exists():
            parent = _resolve_hydra_config(parent_path)
            for k, v in parent.items():
                if k != "defaults":
                    merged[k] = (
                        {**merged.get(k, {}), **v}
                        if isinstance(v, dict) and isinstance(merged.get(k), dict)
                        else v
                    )
    for k, v in data.items():
        if k == "defaults":
            continue
        merged[k] = (
            {**merged.get(k, {}), **v}
            if isinstance(v, dict) and isinstance(merged.get(k), dict)
            else v
        )
    return merged


def _autodetect_config_yaml(schedule: Path, config_name: str) -> Path | None:
    """Find `configs/.../<config_name>.yaml` near a schedule yaml.

    Tries in order:
      1. Sibling of schedules dir: `<schedule.parent.parent>/<config>.yaml`
         — matches gemma4-rl: configs/schedules/<sweep>.yaml + configs/<config>.yaml.
      2. Recursive `<schedule.parent.parent>/**/<config>.yaml` (first hit) —
         catches orak-style nested layouts: configs/<game>/agent/<config>.yaml.
      3. Fallback: walk up to 3 parents from schedule looking for any
         `<dir>/**/<config_name>.yaml` so schedules outside `configs/` work.

    Returns None if nothing matches; caller falls back to placeholder text.
    """
    candidate = schedule.parent.parent / f"{config_name}.yaml"
    if candidate.exists():
        return candidate
    search_root = schedule.parent.parent if schedule.parent.parent.exists() else schedule.parent
    matches = sorted(search_root.rglob(f"{config_name}.yaml"))
    if matches:
        return matches[0]
    for up in (schedule.parent, *schedule.parents[:3]):
        matches = sorted(up.rglob(f"{config_name}.yaml"))
        if matches:
            return matches[0]
    return None


def _extract_chassis(config_yaml: Path | None) -> dict[str, Any]:
    """Pull chassis fields (model_name, lora_rank, ...) from a training config.

    Returns an empty dict if `config_yaml` is None or unreadable. Looks under
    both `train.<key>` and top-level so it works with Hydra structured configs
    and flat configs.
    """
    if config_yaml is None:
        return {}
    merged = _resolve_hydra_config(config_yaml)
    train_block = merged.get("train", {}) if isinstance(merged.get("train"), dict) else {}
    out: dict[str, Any] = {}
    for field in _CHASSIS_FIELDS:
        if field in train_block:
            out[field] = train_block[field]
        elif field in merged:
            out[field] = merged[field]
    return out


def _format_chassis_line(chassis: dict[str, Any]) -> str:
    """Render the chassis dict as a bullet line for the writeup header."""
    if not chassis:
        return "**Chassis:** `<model_name>` · LoRA r=<rank> · max_seq=<n> · num_generations=<n>"
    parts = []
    if "model_name" in chassis:
        parts.append(f"`{chassis['model_name']}`")
    if "lora_rank" in chassis:
        parts.append(f"LoRA r={chassis['lora_rank']}")
    if "max_seq_length" in chassis:
        parts.append(f"max_seq={chassis['max_seq_length']}")
    if "num_generations" in chassis:
        parts.append(f"num_generations={chassis['num_generations']}")
    return "**Chassis:** " + " · ".join(parts)


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
        "description": row.get("description", "").removeprefix("[early_stopped] "),
    }


def _render_skeleton(
    schedule_name: str,
    schedule_data: dict,
    config_name: str,
    results: list[dict],
    chassis: dict[str, Any] | None = None,
    schedule_path: Path | None = None,
    out_path: Path | None = None,
) -> str:
    iters = schedule_data.get("iters", [])
    common = schedule_data.get("common_overrides", [])
    schedule_yaml = yaml.safe_dump(
        schedule_data, sort_keys=False, default_flow_style=False
    ).rstrip()
    chassis_line = _format_chassis_line(chassis or {})

    # Compute the schedule link as a relative path from the writeup's parent
    # to the schedule yaml. Adapts to whatever depth `--out` lands at — works
    # for two-level (<config>/<sweep>.md), three-level (<task>/<config>/<sweep>
    # .md), and any other layout the caller chooses. Falls back to the canonical
    # configs/schedules/<name>.yaml when paths can't be resolved.
    if schedule_path is not None and out_path is not None:
        rel = os.path.relpath(schedule_path.resolve(), out_path.parent.resolve())
        schedule_href = rel.replace(os.sep, "/")
    else:
        schedule_href = f"configs/schedules/{schedule_name}.yaml"
    schedule_link = f"[`configs/schedules/{schedule_name}.yaml`]({schedule_href})"
    iter_count = (
        f"{len(iters)} iters · {len(results)} rows logged" if results else f"{len(iters)} iters"
    )
    header_section = f"""# `{schedule_name}` — <one-line hypothesis here>

**Schedule:** {schedule_link}
**Config:** `{config_name}`
{chassis_line}
**Iterations:** {iter_count}
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
        rows = [_format_results_row(r) for r in results[-len(iters) :]]
        header = (
            "| iter | exp | steps | runtime | mean_total | f1 | no_halluc"
            " | well_formed | pass_all |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|"
        results_table = f"\n## Results\n\n{header}\n{sep}\n"
        for i, r in enumerate(rows, 1):
            rt = (
                f"{r['runtime_min']:.1f}m"
                if isinstance(r["runtime_min"], (int, float))
                else r["runtime_min"]
            )
            mt = (
                f"{r['mean_total']:.3f}"
                if isinstance(r["mean_total"], (int, float))
                else r["mean_total"]
            )
            f1 = f"{r['f1']:.3f}" if isinstance(r["f1"], (int, float)) else r["f1"]
            nh = (
                f"{r['no_halluc']:.3f}"
                if isinstance(r["no_halluc"], (int, float))
                else r["no_halluc"]
            )
            wf = (
                f"{r['well_formed']:+.3f}"
                if isinstance(r["well_formed"], (int, float))
                else r["well_formed"]
            )
            pa = (
                f"{r['pass_all_pct']}%"
                if isinstance(r["pass_all_pct"], (int, float))
                else r["pass_all_pct"]
            )
            results_table += (
                f"| {i}/{len(iters)} | E{r['experiment']} | {r['steps']} | {rt}"
                f" | {mt} | {f1} | {nh} | {wf} | {pa} |\n"
            )
    else:
        results_table = (
            "\n## Results\n\n_No results.jsonl rows found yet — fill in once the sweep finishes._\n"
        )

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
    config_yaml: Path | None = typer.Option(
        None,
        "--config-yaml",
        help="Path to configs/<config>.yaml — autodetected as configs/<config>.yaml "
        "next to the schedule if not given. Used to extract chassis (model_name, "
        "lora_rank, max_seq_length, num_generations) into the writeup header.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing file"),
) -> None:
    """Emit a per-sweep writeup skeleton.

    The skeleton inlines the schedule yaml + a per-iter results table from
    `results.jsonl`, and prefills a chassis line (model + LoRA + seq len)
    from `configs/<config>.yaml` when that file is auto-detectable. Author
    fills in Hypothesis / Pre-launch comparisons / Verdict / Next move.
    """
    if not schedule.exists():
        raise typer.BadParameter(f"schedule not found: {schedule}")
    if out.exists() and not force:
        raise typer.BadParameter(f"{out} exists — pass --force to overwrite")

    schedule_data = yaml.safe_load(schedule.read_text())
    schedule_name = schedule.stem

    if config_yaml is None:
        config_yaml = _autodetect_config_yaml(schedule, config)
    chassis = _extract_chassis(config_yaml)

    results = load_results(experiments_dir=experiments_dir, tag=tag, config_name=config)

    out.parent.mkdir(parents=True, exist_ok=True)
    body = _render_skeleton(
        schedule_name=schedule_name,
        schedule_data=schedule_data,
        config_name=config,
        results=results,
        chassis=chassis,
        schedule_path=schedule,
        out_path=out,
    )
    out.write_text(body)
    chassis_note = f" [chassis: {len(chassis)} fields]" if chassis else " [chassis: not found]"
    typer.echo(f"wrote {out}  ({len(body):,} chars, {len(results)} results rows){chassis_note}")


if __name__ == "__main__":
    app()
