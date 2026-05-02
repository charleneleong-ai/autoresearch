"""Cross-sweep comparison plots for autoresearch.

The per-tag plotly chart from ``autoresearch.render.render`` (and orak/
gemma4-rlvr's ``plot_progress``) is great for live monitoring but renders
*sparse* with only 2-4 iters — lots of whitespace, hard to see deltas at
a glance. This module generates after-the-fact static matplotlib charts
that consume the same ``results.jsonl`` files and overlay multiple
sweeps for direct comparison.

Three flavours, all available as functions and as typer CLI commands:

* :func:`plot_multi_tag_overlay` — overlay multiple sweeps on the same
  iter axis, with per-iter percentage delta annotations between the
  baseline (first sweep) and the rest. Best for "did adding feature X
  lift scores?" questions.
* :func:`plot_cross_game_scoreboard` — bar chart per game, one bar per
  sweep, best evaluation score. Best for "where does feature X help?"
  summary questions.
* :func:`plot_milestone_progression` — twin-axis line chart over a
  hand-curated sequence of milestones (``label``, ``metrics``-dict).
  Best for "where are we now across the project's checkpoints?" — the
  cross-PR / cross-experiment trajectory view that doesn't fit the
  per-tag results.jsonl shape because the data lives in scattered doc
  verdict tables, not a single results file.

Examples
--------

CLI overlay::

    python -m autoresearch.compare overlay \\
        --tag harness_check --tag cognitive_check --tag cognitive_check_v2 \\
        --label "Stage A baseline" --label "Stage C v1" --label "Stage C v2" \\
        --config-name gemma --game twenty_fourty_eight \\
        --out docs/.../stage_a_vs_c_2048.png

CLI scoreboard (``--sep`` tells where each game's tags end)::

    python -m autoresearch.compare scoreboard \\
        --game twenty_fourty_eight --game super_mario --game pokemon_red \\
        --tag harness_check --tag cognitive_check \\
        --tag mario_check \\
        --tag pokemon_check --tag pokemon_check_v3 \\
        --label "Stage A" --label "Stage C" \\
        --label "Stage C" \\
        --label "Stage C v1" --label "Stage D" \\
        --sep 2 --sep 1 --sep 2 \\
        --config-name gemma \\
        --out docs/.../cross_game_scoreboard.png

Python::

    from autoresearch.compare import plot_multi_tag_overlay
    plot_multi_tag_overlay(
        sweeps=[
            ("harness_check", "Stage A baseline (no vmem)"),
            ("cognitive_check", "Stage C v1 (vmem on, uniform)"),
            ("cognitive_check_v2", "Stage C v2 (richer template)"),
        ],
        config_name="gemma",
        game="twenty_fourty_eight",
        out_path="docs/.../stage_a_vs_c_2048.png",
    )
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import typer
import yaml

from autoresearch.results import filter_by_game, get_score, load_results

app = typer.Typer(help="Cross-sweep comparison plots for autoresearch.")


# ── helpers ────────────────────────────────────────────────────────────


def _best_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return max(get_score(r) for r in rows)


# ── multi-tag overlay ──────────────────────────────────────────────────


def plot_multi_tag_overlay(
    sweeps: Sequence[tuple[str, str]],
    *,
    experiments_dir: str | Path = "experiments",
    config_name: str | None = None,
    game: str | None = None,
    out_path: str | Path,
    title: str | None = None,
    annotate_deltas: bool = True,
    figsize: tuple[float, float] = (13.0, 6.5),
    dpi: int = 140,
) -> Path:
    """Overlay multiple sweep results on the same iteration axis.

    Parameters
    ----------
    sweeps:
        List of ``(tag, display_label)`` tuples — one per sweep to overlay.
        The first entry is treated as the baseline for delta annotations.
    experiments_dir:
        Root experiments directory (default ``"experiments"``).
    config_name:
        Per-config sub-directory under ``<experiments_dir>/<tag>/``. Empty
        for flat layout.
    game:
        Optional filter to a single game (rows whose ``game`` field matches).
    out_path:
        Where to write the PNG. Parent dirs are created.
    title:
        Plot title. Auto-generated if ``None``.
    annotate_deltas:
        When ``True``, draw per-iter percentage deltas between the baseline
        sweep and the second sweep.

    Returns
    -------
    Path
        The output PNG path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    palette = ["#777", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    by_label: dict[str, dict[int, float]] = {}

    for i, (tag, label) in enumerate(sweeps):
        rows = filter_by_game(
            load_results(experiments_dir=experiments_dir, tag=tag, config_name=config_name),
            game,
        )
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r.get("experiment", 0))
        xs = [r.get("experiment", j) for j, r in enumerate(rows)]
        ys = [get_score(r) for r in rows]
        statuses = [r.get("status", "?") for r in rows]
        sizes = [220 if s in ("KEEP", "BASELINE") else 100 for s in statuses]
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]

        ax.plot(xs, ys, "-", color=color, alpha=0.4, linewidth=2)
        ax.scatter(
            xs,
            ys,
            c=color,
            s=sizes,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
            label=label,
        )
        for x, y, s in zip(xs, ys, statuses, strict=False):
            ax.annotate(
                f"{y:.2f}\n{s.lower()}",
                xy=(x, y),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold" if s in ("KEEP", "BASELINE") else "normal",
            )
        by_label[label] = dict(zip(xs, ys, strict=False))

    if annotate_deltas and len(sweeps) >= 2:
        baseline_label = sweeps[0][1]
        compare_label = sweeps[1][1]
        if baseline_label in by_label and compare_label in by_label:
            for exp in sorted(set(by_label[baseline_label]) & set(by_label[compare_label])):
                base = by_label[baseline_label][exp]
                comp = by_label[compare_label][exp]
                if base <= 0:
                    continue
                delta_pct = (comp - base) / base * 100
                ax.annotate(
                    f"{'+' if delta_pct >= 0 else ''}{delta_pct:.0f}%",
                    xy=(exp, (base + comp) / 2),
                    xytext=(40, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="#2ca02c" if delta_pct > 0 else "#d62728",
                    arrowprops=dict(
                        arrowstyle="-[",
                        color="#2ca02c" if delta_pct > 0 else "#d62728",
                        lw=1.5,
                    ),
                )

    all_x: list[int] = []
    all_y: list[float] = []
    for d in by_label.values():
        all_x.extend(d.keys())
        all_y.extend(d.values())
    if all_x:
        ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.7)
        ax.set_xticks(range(min(all_x), max(all_x) + 1))
    if all_y:
        ax.set_ylim(0, max(all_y) * 1.2)

    ax.set_xlabel("Iteration #", fontsize=11)
    ax.set_ylabel("Evaluation Score (higher is better)", fontsize=11)
    if title is None:
        scope = f"{game}" if game else "all games"
        title = f"Sweep comparison — {scope}"
    ax.set_title(title, fontsize=13, color="#222", pad=15)
    ax.grid(True, color="#eee", linewidth=0.7, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── cross-game scoreboard ──────────────────────────────────────────────


def plot_cross_game_scoreboard(
    games_to_sweeps: dict[str, list[tuple[str, str]]],
    *,
    experiments_dir: str | Path = "experiments",
    config_name: str | None = None,
    out_path: str | Path,
    title: str | None = None,
    game_titles: dict[str, str] | None = None,
    game_verdicts: dict[str, str] | None = None,
    figsize_per_panel: tuple[float, float] = (5.5, 6.0),
    dpi: int = 140,
) -> Path:
    """Per-game scoreboard — bar chart of best score per sweep.

    Parameters
    ----------
    games_to_sweeps:
        ``{game_name: [(tag, display_label), ...]}``. Each game becomes
        one subplot panel; bars are best ``evaluation_score`` per sweep.
        The best bar per panel gets a green border.
    game_titles, game_verdicts:
        Optional per-game richer panel title and footer line — useful for
        annotating bottleneck/verdict per game.

    Returns
    -------
    Path
        The output PNG path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_games = len(games_to_sweeps)
    fig, axes = plt.subplots(
        1,
        n_games,
        figsize=(figsize_per_panel[0] * n_games, figsize_per_panel[1]),
        dpi=dpi,
    )
    if n_games == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    palette = ["#888", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]

    for ax, (game, sweeps) in zip(axes, games_to_sweeps.items(), strict=False):
        labels = [s[1] for s in sweeps]
        values = [
            _best_score(
                filter_by_game(
                    load_results(
                        experiments_dir=experiments_dir, tag=s[0], config_name=config_name
                    ),
                    game,
                )
            )
            for s in sweeps
        ]
        n = len(labels)
        colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax.bar(range(n), values, color=colors, edgecolor="white", linewidth=2, zorder=3)

        if any(v > 0 for v in values):
            best_idx = max(range(n), key=lambda i: values[i])
            bars[best_idx].set_edgecolor("#2ca02c")
            bars[best_idx].set_linewidth(3)
        else:
            best_idx = -1

        max_v = max(values) if any(v > 0 for v in values) else 1
        for i, v in enumerate(values):
            ax.text(
                i,
                v + max_v * 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold" if i == best_idx else "normal",
                color="#2ca02c" if i == best_idx else "#333",
            )

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        panel_title = (game_titles or {}).get(game, game)
        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
        if ax is axes[0]:
            ax.set_ylabel("Best Evaluation Score")
        ax.grid(True, color="#eee", linewidth=0.7, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max_v * 1.25)

        verdict = (game_verdicts or {}).get(game)
        if verdict:
            ax.text(
                0.5,
                -0.32,
                verdict,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

    if title:
        fig.suptitle(title, fontsize=13, color="#222", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── milestone progression ──────────────────────────────────────────────


@dataclass(frozen=True)
class Milestone:
    """One point in a cross-experiment progression chart.

    Hand-authored from doc verdicts — these are the ``mean_total`` /
    ``no_halluc`` numbers you copy out of n=1000 A/B tables, not data
    pulled from a sweep's ``results.jsonl``.
    """

    label: str
    metrics: dict[str, float]
    description: str = ""


def _walk_dot_path(data: Any, path: str) -> float:
    """Resolve a dot-path (``"metrics.heldout.mean_total"``) on a nested dict.

    Returns the leaf as ``float``. Raises ``KeyError`` if any segment is
    missing and ``TypeError`` if the leaf is not numeric.
    """
    cur: Any = data
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            raise KeyError(f"path {path!r} not found at segment {seg!r}")
        cur = cur[seg]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        raise TypeError(f"path {path!r} resolved to non-numeric {type(cur).__name__}: {cur!r}")
    return float(cur)


def extract_metrics_from_results_jsonl(
    results_jsonl: str | Path,
    extract: dict[str, str],
    *,
    row: str = "last",
) -> dict[str, float]:
    """Pull a metrics dict from one row of a sweep's ``results.jsonl``.

    ``extract`` maps milestone-metric names to dot-paths into the row
    (e.g. ``{"mean_total": "metrics.heldout.mean_total"}``). ``row`` is
    ``"last"`` (default) or ``"best"`` (highest :func:`get_score`).

    Designed for the ``autoresearch-compare append-milestone
    --from-results-jsonl`` CLI mode — keeps the sweep-→-milestone path
    one command long without coupling :func:`append_milestone` itself
    to a specific row schema.
    """
    rows: list[dict[str, Any]] = []
    with Path(results_jsonl).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{results_jsonl}: no rows")
    if row == "last":
        chosen = rows[-1]
    elif row == "best":
        chosen = max(rows, key=get_score)
    else:
        raise ValueError(f"--row must be 'last' or 'best', got {row!r}")
    return {dest: _walk_dot_path(chosen, path) for dest, path in extract.items()}


def append_milestone(
    yaml_path: str | Path,
    *,
    label: str,
    metrics: dict[str, float],
    description: str = "",
) -> Path:
    """Append a milestone entry to a milestones YAML, creating the file if missing.

    Designed to be called once per sweep verdict — the milestones YAML
    becomes the canonical chronological log of cross-experiment progress,
    consumed by :func:`plot_milestone_progression`. Top-level metadata
    (``title``, ``primary_metric``, ``threshold`` etc.) is preserved
    as-is across appends; set those by hand once when seeding the file.

    On a missing file, a stub ``{"milestones": []}`` is created — the
    caller should follow up by editing the YAML to set ``title`` /
    ``primary_metric`` / ``secondary_metric`` / ``threshold`` so the
    chart has axis labels + a ship line.

    Returns the YAML path written.
    """
    yaml_path = Path(yaml_path)
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{yaml_path}: top-level must be a mapping, got {type(raw).__name__}")
    else:
        raw = {"milestones": []}
    raw.setdefault("milestones", [])
    if not isinstance(raw["milestones"], list):
        raise ValueError(f"{yaml_path}: 'milestones' must be a list")

    entry: dict[str, Any] = {"label": label}
    if description:
        entry["description"] = description
    entry["metrics"] = {k: float(v) for k, v in metrics.items()}
    raw["milestones"].append(entry)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=False))
    return yaml_path


def load_milestones_yaml(path: str | Path) -> tuple[list[Milestone], dict[str, Any]]:
    """Load a milestones YAML file into a list of :class:`Milestone`.

    Schema (all top-level keys optional except ``milestones``)::

        title: "..."                  # plot title (passed through)
        primary_metric: mean_total    # default for plot_milestone_progression
        secondary_metric: no_halluc   # ditto
        threshold: -0.5
        threshold_label: "ship"
        milestones:
          - label: vanilla
            description: "Gemma 4 4B base, no fine-tune"
            metrics:
              mean_total: 8.354
              no_halluc: -0.876
          - label: E18
            description: "v2 GRPO champion"
            metrics:
              mean_total: 9.324
              no_halluc: -0.840

    Returns ``(milestones, top_level_kwargs)``. The kwargs dict can be
    splatted directly into :func:`plot_milestone_progression`.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "milestones" not in raw:
        raise ValueError(f"{path}: top-level must be a mapping with a 'milestones' key")
    milestones = [
        Milestone(
            label=str(m["label"]),
            metrics={k: float(v) for k, v in (m.get("metrics") or {}).items()},
            description=str(m.get("description") or ""),
        )
        for m in raw["milestones"]
    ]
    kwargs = {k: v for k, v in raw.items() if k != "milestones"}
    return milestones, kwargs


_PRIMARY_PALETTE = ("#2563eb", "#0ea5e9", "#14b8a6", "#6366f1")
_SECONDARY_PALETTE = ("#dc2626", "#f97316", "#eab308", "#a855f7")
_PRIMARY_MARKERS = ("o", "^", "s", "D")
_SECONDARY_MARKERS = ("s", "D", "v", "P")


def _to_metric_list(arg: str | Sequence[str] | None) -> list[str]:
    if arg is None:
        return []
    if isinstance(arg, str):
        return [arg]
    return list(arg)


def plot_milestone_progression(
    milestones: Sequence[Milestone],
    *,
    primary_metric: str | Sequence[str],
    secondary_metric: str | Sequence[str] | None = None,
    primary_label: str | None = None,
    secondary_label: str | None = None,
    primary_color: str | None = None,
    secondary_color: str | None = None,
    primary_colors: Sequence[str] | None = None,
    secondary_colors: Sequence[str] | None = None,
    primary_ylim: tuple[float, float] | None = None,
    secondary_ylim: tuple[float, float] | None = None,
    threshold: float | None = None,
    threshold_label: str = "",
    threshold_axis: str = "secondary",
    threshold_color: str = "#16a34a",
    title: str | None = None,
    out_path: str | Path,
    figsize: tuple[float, float] = (11.0, 5.5),
    dpi: int = 120,
    highlight_last: bool = True,
    annotate_primary: bool = True,
) -> Path:
    """Twin-axis line chart over a hand-curated sequence of milestones.

    Use this when you want a "where are we now?" trajectory view across
    multiple experiments / PRs — the cross-checkpoint chart that doesn't
    fit the per-tag ``results.jsonl`` shape that
    :func:`plot_multi_tag_overlay` consumes. Each milestone is a fixed
    ``(label, metrics-dict)`` pair authored by the researcher; the
    metrics are typically copied from doc verdict tables (n=1000 A/B
    results sitting in scattered PR descriptions and writeups).

    Plots one or more ``primary_metric`` keys on the left axis as solid
    lines, and optionally one or more ``secondary_metric`` keys on a twin
    right axis as dashed lines. Pass a string for the single-metric case
    (preserves the original visual) or a list to stack multiple lines on
    the same axis. A horizontal reference at ``threshold`` (on whichever
    axis is named in ``threshold_axis``) marks the falsification /
    ship-as-champion line. The last milestone of the first primary
    metric is circled when ``highlight_last`` is true, and the first
    primary metric's values are labelled inline by default.

    Parameters
    ----------
    milestones:
        Sequence of :class:`Milestone`. ``metrics`` for each should contain
        the requested keys; missing keys become ``None`` and are dropped
        from that line.
    primary_metric, secondary_metric:
        ``str`` for one line, or ``Sequence[str]`` to stack multiple
        lines on the same axis (one per metric).
    primary_color, secondary_color:
        Singular alias for ``primary_colors[0]`` / ``secondary_colors[0]``
        — kept for backwards compatibility with the single-metric API.
        Pass ``primary_colors`` / ``secondary_colors`` for multi-metric
        per-line control; if neither is set, the built-in palette is used.
    threshold, threshold_label:
        Optional horizontal reference line + annotation. Common use:
        ``-0.5`` "ship-as-champion" line for a hallucination metric.
    threshold_axis:
        ``"primary"`` or ``"secondary"`` — which axis the threshold lives
        on. Default ``"secondary"``.
    out_path:
        Where to write the PNG. Parent dirs are created.

    Returns
    -------
    Path
        The output PNG path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not milestones:
        raise ValueError("milestones must be non-empty")

    primary_keys = _to_metric_list(primary_metric)
    secondary_keys = _to_metric_list(secondary_metric)
    if not primary_keys:
        raise ValueError("primary_metric must be a string or non-empty sequence")

    if primary_colors is not None and primary_color is not None:
        raise ValueError("pass either primary_color or primary_colors, not both")
    if secondary_colors is not None and secondary_color is not None:
        raise ValueError("pass either secondary_color or secondary_colors, not both")

    primary_palette: list[str] = (
        list(primary_colors)
        if primary_colors is not None
        else ([primary_color, *_PRIMARY_PALETTE[1:]] if primary_color else list(_PRIMARY_PALETTE))
    )
    secondary_palette: list[str] = (
        list(secondary_colors)
        if secondary_colors is not None
        else (
            [secondary_color, *_SECONDARY_PALETTE[1:]]
            if secondary_color
            else list(_SECONDARY_PALETTE)
        )
    )

    xs = list(range(len(milestones)))
    labels = [m.label for m in milestones]

    fig, ax_primary = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax_primary.set_facecolor("white")

    handles: list[Any] = []

    primary_lines = []
    first_valid_primary: list[tuple[int, float]] = []
    for i, key in enumerate(primary_keys):
        vals: list[float | None] = [m.metrics.get(key) for m in milestones]
        valid = [(x, v) for x, v in zip(xs, vals, strict=False) if v is not None]
        if not valid:
            if i == 0:
                raise ValueError(f"no milestone has metric {key!r}")
            continue
        color = primary_palette[i % len(primary_palette)]
        marker = _PRIMARY_MARKERS[i % len(_PRIMARY_MARKERS)]
        line = ax_primary.plot(
            [x for x, _ in valid],
            [v for _, v in valid],
            f"-{marker}",
            color=color,
            linewidth=2.2,
            markersize=7,
            label=f"{key} (left axis)",
        )[0]
        primary_lines.append((key, color, line))
        handles.append(line)
        if i == 0:
            first_valid_primary = valid

    if annotate_primary and first_valid_primary:
        annotate_color = primary_palette[0]
        for x, v in first_valid_primary:
            ax_primary.annotate(
                f"{v:.2f}",
                (x, v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=annotate_color,
            )

    if highlight_last and first_valid_primary:
        last_x, last_v = first_valid_primary[-1]
        ax_primary.scatter(
            [last_x],
            [last_v],
            s=160,
            facecolors="none",
            edgecolors=primary_palette[0],
            linewidth=2,
            zorder=5,
        )

    ax_secondary = None
    if secondary_keys:
        for i, key in enumerate(secondary_keys):
            vals = [m.metrics.get(key) for m in milestones]
            valid = [(x, v) for x, v in zip(xs, vals, strict=False) if v is not None]
            if not valid:
                continue
            if ax_secondary is None:
                ax_secondary = ax_primary.twinx()
            color = secondary_palette[i % len(secondary_palette)]
            marker = _SECONDARY_MARKERS[i % len(_SECONDARY_MARKERS)]
            line = ax_secondary.plot(
                [x for x, _ in valid],
                [v for _, v in valid],
                f"--{marker}",
                color=color,
                linewidth=2.0,
                markersize=6,
                label=f"{key} (right axis)",
            )[0]
            handles.append(line)
        if ax_secondary is not None:
            sec_label = secondary_label or (
                secondary_keys[0] if len(secondary_keys) == 1 else "metrics (right)"
            )
            ax_secondary.set_ylabel(sec_label, color=secondary_palette[0])
            ax_secondary.tick_params(axis="y", labelcolor=secondary_palette[0])
            if secondary_ylim is not None:
                ax_secondary.set_ylim(*secondary_ylim)

    if threshold is not None:
        target_ax = ax_secondary if (threshold_axis == "secondary" and ax_secondary) else ax_primary
        target_ax.axhline(threshold, color=threshold_color, linestyle=":", alpha=0.5, linewidth=1)
        if threshold_label:
            target_ax.text(
                len(milestones) - 0.5,
                threshold,
                f"  {threshold} {threshold_label}",
                color=threshold_color,
                fontsize=8,
                va="bottom",
                ha="right",
            )

    ax_primary.set_xticks(xs)
    ax_primary.set_xticklabels(labels, rotation=20, ha="right")
    pri_label = primary_label or (primary_keys[0] if len(primary_keys) == 1 else "metrics (left)")
    ax_primary.set_ylabel(pri_label, color=primary_palette[0])
    ax_primary.tick_params(axis="y", labelcolor=primary_palette[0])
    if primary_ylim is not None:
        ax_primary.set_ylim(*primary_ylim)
    ax_primary.grid(True, alpha=0.25)

    if title:
        ax_primary.set_title(title, fontsize=11, pad=12)

    if handles:
        ax_primary.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────


@app.command()
def overlay(
    tag: list[str] = typer.Option(..., "--tag", "-t", help="Sweep tag (repeat for multi-sweep)"),
    label: list[str] = typer.Option(
        ...,
        "--label",
        "-l",
        help="Display label per tag (same order)",
    ),
    out: str = typer.Option(..., help="Output PNG path"),
    experiments_dir: str = typer.Option("experiments", help="Root experiments directory"),
    config_name: str | None = typer.Option(
        None,
        help="Per-config sub-dir under experiments/<tag>/",
    ),
    game: str | None = typer.Option(None, help="Filter to a specific game"),
    title: str | None = typer.Option(None, help="Plot title (default auto)"),
    no_deltas: bool = typer.Option(False, help="Disable per-iter delta annotations"),
) -> None:
    """Overlay multiple sweeps on the same iter axis with delta annotations."""
    if len(tag) != len(label):
        raise typer.BadParameter(
            f"--tag count ({len(tag)}) must match --label count ({len(label)})"
        )
    p = plot_multi_tag_overlay(
        sweeps=list(zip(tag, label, strict=False)),
        experiments_dir=experiments_dir,
        config_name=config_name,
        game=game,
        out_path=out,
        title=title,
        annotate_deltas=not no_deltas,
    )
    typer.echo(f"wrote {p}")


@app.command()
def scoreboard(
    out: str = typer.Option(..., help="Output PNG path"),
    experiments_dir: str = typer.Option("experiments", help="Root experiments directory"),
    config_name: str | None = typer.Option(None, help="Per-config sub-dir"),
    game: list[str] = typer.Option(
        ...,
        "--game",
        "-g",
        help="Game name (repeat for multiple games — each game gets its own panel)",
    ),
    tag: list[str] = typer.Option(
        ...,
        "--tag",
        "-t",
        help="Sweep tag — these are MATCHED to games positionally via --sep",
    ),
    label: list[str] = typer.Option(
        ...,
        "--label",
        "-l",
        help="Display label per tag, same order as --tag",
    ),
    sep: list[int] = typer.Option(
        ...,
        help="Number of (tag, label) pairs per game — must sum to len(tag)",
    ),
    title: str | None = typer.Option(None, help="Top-level title"),
) -> None:
    """Cross-game scoreboard. Pass --game once per panel; --tag/--label
    multiple times overall; --sep tells where each game's tags end."""
    if sum(sep) != len(tag) or len(tag) != len(label):
        raise typer.BadParameter(
            f"--sep ({sep}, sum={sum(sep)}) must add up to len(tag)={len(tag)} "
            f"== len(label)={len(label)}"
        )
    if len(sep) != len(game):
        raise typer.BadParameter(
            f"--sep entries ({len(sep)}) must match --game entries ({len(game)})"
        )

    games_to_sweeps: dict[str, list[tuple[str, str]]] = {}
    cursor = 0
    for g, n in zip(game, sep, strict=False):
        games_to_sweeps[g] = list(
            zip(tag[cursor : cursor + n], label[cursor : cursor + n], strict=False)
        )
        cursor += n

    p = plot_cross_game_scoreboard(
        games_to_sweeps=games_to_sweeps,
        experiments_dir=experiments_dir,
        config_name=config_name,
        out_path=out,
        title=title,
    )
    typer.echo(f"wrote {p}")


def _resolve_axis_metrics(
    cli_values: list[str],
    yaml_default: Any,
) -> str | list[str] | None:
    """CLI flag wins; otherwise pass YAML value through (str or list)."""
    if cli_values:
        return cli_values if len(cli_values) > 1 else cli_values[0]
    return yaml_default


@app.command()
def progression(
    milestones_yaml: str = typer.Option(
        ...,
        "--milestones-yaml",
        "-m",
        help="Path to milestones YAML (see Milestone schema in load_milestones_yaml).",
    ),
    out: str = typer.Option(..., help="Output PNG path"),
    primary: list[str] = typer.Option(
        [],
        "--primary",
        help=(
            "Primary metric name (left axis). Repeat to stack multiple lines on the "
            "same axis. Falls back to YAML 'primary_metric' (scalar or list)."
        ),
    ),
    secondary: list[str] = typer.Option(
        [],
        "--secondary",
        help=(
            "Secondary metric name (right axis). Repeat to stack multiple lines on the "
            "same axis. Falls back to YAML 'secondary_metric' (scalar or list)."
        ),
    ),
    threshold: float | None = typer.Option(
        None,
        help="Horizontal reference line value. Falls back to YAML 'threshold'.",
    ),
    threshold_label: str | None = typer.Option(None, help="Label next to the threshold line."),
    title: str | None = typer.Option(None, help="Plot title. Falls back to YAML 'title'."),
) -> None:
    """Cross-experiment trajectory chart from a hand-curated YAML of milestones."""
    milestones, defaults = load_milestones_yaml(milestones_yaml)
    pri = _resolve_axis_metrics(primary, defaults.get("primary_metric"))
    if not pri:
        raise typer.BadParameter(
            "primary metric required: pass --primary or set 'primary_metric' in YAML"
        )
    sec = _resolve_axis_metrics(secondary, defaults.get("secondary_metric"))
    p = plot_milestone_progression(
        milestones,
        primary_metric=pri,
        secondary_metric=sec,
        threshold=threshold if threshold is not None else defaults.get("threshold"),
        threshold_label=(
            threshold_label
            if threshold_label is not None
            else (defaults.get("threshold_label") or "")
        ),
        title=title or defaults.get("title"),
        out_path=out,
    )
    typer.echo(f"wrote {p}")


@app.command("append-milestone")
def append_milestone_cmd(
    milestones_yaml: str = typer.Option(
        ...,
        "--milestones-yaml",
        "-m",
        help="Target YAML — appended to in place, created if missing.",
    ),
    label: str = typer.Option(..., "--label", help="Milestone label (e.g. v3_slot_grounded)."),
    description: str = typer.Option(
        "", "--description", help="Free-text description of what changed."
    ),
    metric: list[str] = typer.Option(
        [],
        "--metric",
        help="Metric in KEY=VALUE form (numeric). Repeat for multiple metrics.",
    ),
    from_results_jsonl: Path | None = typer.Option(
        None,
        "--from-results-jsonl",
        help="Pull metrics from this sweep's results.jsonl (combine with --extract).",
    ),
    extract: list[str] = typer.Option(
        [],
        "--extract",
        help=(
            "DEST=DOT.PATH (e.g. mean_total=metrics.heldout.mean_total). "
            "Only used with --from-results-jsonl. Repeat for multiple metrics."
        ),
    ),
    row: str = typer.Option(
        "last",
        "--row",
        help="Which results.jsonl row to read: 'last' (default) or 'best' (highest score).",
    ),
) -> None:
    """Append one milestone entry to a milestones YAML.

    Two ways to provide metrics — combine freely, ``--metric`` overrides
    on key conflict:

    \b
    --metric KEY=VALUE         hand-typed numbers (n=1000 eval verdicts)
    --from-results-jsonl PATH  pull from a sweep's results.jsonl
      --extract DEST=DOT.PATH  required: which fields to lift out
      --row last|best          which row to read (default last)

    Example — hand-typed n=1000 verdict::

        autoresearch-compare append-milestone \\
            -m docs/experiments/dd_explainer/milestones.yaml \\
            --label v3_slot_grounded \\
            --description "Slot-grounded JSON output" \\
            --metric mean_total=10.96 \\
            --metric no_halluc=-0.48

    Example — pull from a sweep's results.jsonl::

        autoresearch-compare append-milestone \\
            -m docs/experiments/dd_explainer/milestones.yaml \\
            --label e25_run \\
            --from-results-jsonl experiments/dd_explainer/train_v2_80gb/results.jsonl \\
            --row best \\
            --extract mean_total=metrics.heldout.mean_total \\
            --extract no_halluc=metrics.heldout.no_hallucinated_facts_mean
    """
    metrics_dict: dict[str, float] = {}

    if from_results_jsonl is not None:
        if not extract:
            raise typer.BadParameter(
                "--from-results-jsonl requires at least one --extract DEST=DOT.PATH"
            )
        extract_map: dict[str, str] = {}
        for e in extract:
            if "=" not in e:
                raise typer.BadParameter(f"--extract must be DEST=DOT.PATH, got {e!r}")
            k, _, v = e.partition("=")
            extract_map[k.strip()] = v.strip()
        try:
            metrics_dict.update(
                extract_metrics_from_results_jsonl(from_results_jsonl, extract_map, row=row)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    for m in metric:
        if "=" not in m:
            raise typer.BadParameter(f"--metric must be KEY=VALUE, got {m!r}")
        k, _, v = m.partition("=")
        try:
            metrics_dict[k.strip()] = float(v)
        except ValueError as e:
            raise typer.BadParameter(f"--metric {m!r}: value must be numeric") from e

    if not metrics_dict:
        raise typer.BadParameter(
            "no metrics provided: pass --metric KEY=VALUE and/or --from-results-jsonl --extract"
        )

    p = append_milestone(
        milestones_yaml,
        label=label,
        metrics=metrics_dict,
        description=description,
    )
    typer.echo(f"appended {label} to {p}")


def cli() -> None:
    """Entry point used by ``python -m autoresearch.compare`` and pyproject scripts."""
    app()


if __name__ == "__main__":
    cli()
