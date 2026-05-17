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

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import typer
import yaml

from autoresearch.results import filter_by_game, get_score, load_results, read_jsonl

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
    ax.set_ylabel("Evaluation Score — % normalised, 0–100 (higher is better)", fontsize=11)
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


_SCOREBOARD_PALETTE = ["#888", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]


def _render_scoreboard_panels(
    games_to_bars: dict[str, list[tuple[str, float]]],
    *,
    out_path: Path,
    title: str | None,
    game_titles: dict[str, str] | None,
    game_verdicts: dict[str, str] | None,
    figsize_per_panel: tuple[float, float],
    dpi: int,
) -> Path:
    """Render the cross-game bar chart.

    Shared rendering for :func:`plot_cross_game_scoreboard` and
    :func:`plot_cross_game_scoreboard_from_index`; takes pre-resolved
    ``{game: [(label, value), ...]}`` so callers differ only in how they
    compute the bar values.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_games = len(games_to_bars)
    fig, axes = plt.subplots(
        1,
        n_games,
        figsize=(figsize_per_panel[0] * n_games, figsize_per_panel[1]),
        dpi=dpi,
    )
    if n_games == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    try:
        for ax, (game, bars_in) in zip(axes, games_to_bars.items(), strict=True):
            labels = [b[0] for b in bars_in]
            values = [b[1] for b in bars_in]
            n = len(labels)
            colors = [_SCOREBOARD_PALETTE[i % len(_SCOREBOARD_PALETTE)] for i in range(n)]
            bars = ax.bar(range(n), values, color=colors, edgecolor="white", linewidth=2, zorder=3)

            if any(v > 0 for v in values):
                best_idx = max(range(n), key=lambda i: values[i])
                bars[best_idx].set_edgecolor("#2ca02c")
                bars[best_idx].set_linewidth(3)
                max_v = max(values)
            else:
                best_idx = -1
                max_v = 1.0

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
                ax.set_ylabel("Best Evaluation Score (% normalised, 0–100)")
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
    finally:
        plt.close(fig)
    return out_path


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
    games_to_bars: dict[str, list[tuple[str, float]]] = {
        game: [
            (
                label,
                _best_score(
                    filter_by_game(
                        load_results(
                            experiments_dir=experiments_dir, tag=tag, config_name=config_name
                        ),
                        game,
                    )
                ),
            )
            for tag, label in sweeps
        ]
        for game, sweeps in games_to_sweeps.items()
    }
    return _render_scoreboard_panels(
        games_to_bars,
        out_path=Path(out_path),
        title=title,
        game_titles=game_titles,
        game_verdicts=game_verdicts,
        figsize_per_panel=figsize_per_panel,
        dpi=dpi,
    )


# ── scoreboard-from-index (single canonical results.jsonl) ────────────


def _best_score_for_variant(rows: list[dict[str, Any]], *, game: str, variant: str) -> float:
    """Best score among rows matching both ``game`` and ``variant``."""
    return _best_score([r for r in filter_by_game(rows, game) if r.get("variant") == variant])


def plot_cross_game_scoreboard_from_index(
    *,
    index_path: str | Path,
    games_to_variants: dict[str, list[tuple[str, str]]],
    out_path: str | Path,
    title: str | None = None,
    game_titles: dict[str, str] | None = None,
    game_verdicts: dict[str, str] | None = None,
    figsize_per_panel: tuple[float, float] = (5.5, 6.0),
    dpi: int = 140,
) -> Path:
    """Per-game scoreboard reading from a single consolidated index file.

    Same chart as :func:`plot_cross_game_scoreboard`, but the data source
    is one consolidated ``results.jsonl`` (built by
    :func:`autoresearch.results.consolidate`) and bars are filtered by
    ``(game, variant)`` instead of per-tag directory lookups. Use when
    one chain's ``results.jsonl`` holds many variants you want to render
    as separate bars — avoids per-variant shadow tag dirs that
    :func:`plot_cross_game_scoreboard` would require.

    Parameters
    ----------
    index_path:
        Path to the consolidated ``results.jsonl`` (one row per measurement;
        rows must have ``game`` and ``variant`` fields).
    games_to_variants:
        ``{game_name: [(variant, display_label), ...]}``. Each game is a
        panel; each ``(variant, label)`` pair is a bar. The bar value is
        the max ``evaluation_score`` for rows matching ``(game, variant)``.
        Missing variants render as 0.0 bars rather than raising.

    Returns
    -------
    Path
        The output PNG path.
    """
    rows = read_jsonl(index_path)
    games_to_bars: dict[str, list[tuple[str, float]]] = {
        game: [
            (label, _best_score_for_variant(rows, game=game, variant=variant))
            for variant, label in variants
        ]
        for game, variants in games_to_variants.items()
    }
    return _render_scoreboard_panels(
        games_to_bars,
        out_path=Path(out_path),
        title=title,
        game_titles=game_titles,
        game_verdicts=game_verdicts,
        figsize_per_panel=figsize_per_panel,
        dpi=dpi,
    )


# ── milestone progression ──────────────────────────────────────────────


@dataclass(frozen=True)
class Milestone:
    """One point in a cross-experiment progression chart.

    Hand-authored from doc verdicts — these are the ``mean_total`` /
    ``no_halluc`` numbers you copy out of n=1000 A/B tables, not data
    pulled from a sweep's ``results.jsonl``.

    ``metric_stds`` is optional per-metric standard deviation. When a key
    appears in both ``metrics`` and ``metric_stds``,
    :func:`plot_milestone_progression` renders an error bar of
    ``mean ± std`` at that point. Milestones without a std for a given
    metric draw as bare markers (no whisker), so you can mix n=1 and
    n>1 milestones on the same chart.

    ``metric_scores`` is optional per-metric raw iter scores. When a key
    appears, :func:`plot_milestone_progression` overlays the individual
    scores as scatter dots on top of the mean — surfaces single-iter
    outliers (e.g. an iter breaching a ceiling once) that a wide σ band
    would otherwise hide.

    ``verdict`` and ``n`` are opt-in fields consumed by the bar-form chart
    (:func:`plot_milestone_bars`) for cross-stage A/B comparison.
    ``verdict`` is a free-form tag (typically ``BASELINE`` / ``FLAT`` /
    ``NEUTRAL+`` / ``REGRESS`` / ``LIFT`` / ``PENDING``) that drives the
    bar colour; ``n`` is the sample count rendered inside the bar.
    """

    label: str
    metrics: dict[str, float]
    description: str = ""
    metric_stds: dict[str, float] = field(default_factory=dict)
    metric_scores: dict[str, list[float]] = field(default_factory=dict)
    verdict: str | None = None
    n: int | None = None


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
    extract_stds: dict[str, str] | None = None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    """Pull a metrics dict from one row of a sweep's ``results.jsonl``.

    ``extract`` maps milestone-metric names to dot-paths into the row
    (e.g. ``{"mean_total": "metrics.heldout.mean_total"}``). ``row`` is
    ``"last"`` (default) or ``"best"`` (highest :func:`get_score`).

    When ``extract_stds`` is provided, returns a ``(metrics, stds)`` tuple
    instead of a single dict. ``extract_stds`` mirrors ``extract`` (keys
    are milestone metric names; values are dot-paths into the row). Use
    this to pull both ``evaluation_score`` and ``evaluation_score_std``
    in one call so :func:`append_milestone` can stamp the error bar at
    the same time as the mean.

    Designed for the ``autoresearch-compare append-milestone
    --from-results-jsonl`` CLI mode — keeps the sweep-→-milestone path
    one command long without coupling :func:`append_milestone` itself
    to a specific row schema.
    """
    rows = read_jsonl(results_jsonl)
    if not rows:
        raise ValueError(f"{results_jsonl}: no rows")
    if row == "last":
        chosen = rows[-1]
    elif row == "best":
        chosen = max(rows, key=get_score)
    else:
        raise ValueError(f"--row must be 'last' or 'best', got {row!r}")
    metrics = {dest: _walk_dot_path(chosen, path) for dest, path in extract.items()}
    if extract_stds is None:
        return metrics
    stds = {dest: _walk_dot_path(chosen, path) for dest, path in extract_stds.items()}
    return metrics, stds


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
    milestones = [_milestone_from_raw(m) for m in raw["milestones"]]
    kwargs = {k: v for k, v in raw.items() if k != "milestones"}
    return milestones, kwargs


def _milestone_from_raw(m: dict[str, Any]) -> Milestone:
    """Parse one milestone dict. A metric value may be either a scalar
    (existing schema) or a dict ``{mean: float, std?: float, scores?: list[float]}``
    — ``std`` populates ``metric_stds`` (error bar) and ``scores`` populates
    ``metric_scores`` (per-iter scatter overlay)."""
    metrics: dict[str, float] = {}
    metric_stds: dict[str, float] = {}
    metric_scores: dict[str, list[float]] = {}
    for k, v in (m.get("metrics") or {}).items():
        if isinstance(v, dict):
            if "mean" not in v:
                raise ValueError(
                    f"milestone {m.get('label')!r} metric {k!r}: "
                    f"dict form requires 'mean' key, got {sorted(v.keys())}"
                )
            metrics[k] = float(v["mean"])
            if "std" in v:
                metric_stds[k] = float(v["std"])
            if "scores" in v:
                metric_scores[k] = [float(x) for x in v["scores"]]
        else:
            metrics[k] = float(v)
    return Milestone(
        label=str(m["label"]),
        metrics=metrics,
        description=str(m.get("description") or ""),
        metric_stds=metric_stds,
        metric_scores=metric_scores,
        verdict=str(m["verdict"]) if m.get("verdict") is not None else None,
        n=int(m["n"]) if m.get("n") is not None else None,
    )


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


def _resolve_palette(
    colors: Sequence[str] | None,
    color: str | None,
    default: Sequence[str],
    name: str,
) -> list[str]:
    """Pick the per-line color list for primary/secondary axes.

    `colors` (full override) and `color` (single first-line override) are
    mutually exclusive. Falls back to `default` when neither is set.
    """
    if colors is not None and color is not None:
        raise ValueError(f"pass either {name} or {name}s, not both")
    if colors is not None:
        return list(colors)
    if color is not None:
        return [color, *default[1:]]
    return list(default)


def _draw_metric_series(
    ax: Any,
    keys: Sequence[str],
    milestones: Sequence[Milestone],
    xs: Sequence[int],
    *,
    palette: Sequence[str],
    markers: Sequence[str],
    linestyle: str,
    linewidth: float,
    markersize: float,
    label_suffix: str,
    handles: list[Any],
    require_first: bool = False,
) -> tuple[list[tuple[str, str, Any]], list[tuple[int, float]]]:
    """Plot one line per metric key onto ``ax``. Returns (drawn_lines, first_valid_xy).

    `drawn_lines` is `[(key, color, line), ...]` for the metrics that had at
    least one non-None value. `first_valid_xy` is the (x, v) pairs for the
    first key — used by the caller for inline annotations + last-point
    highlighting. Raises if `require_first` and the first key has no data.
    """
    drawn: list[tuple[str, str, Any]] = []
    first_valid: list[tuple[int, float]] = []
    # Asymmetric yerr — uses min/max range when metric_scores is present
    # for a milestone, falls back to ±std otherwise. Keeps the whisker caps
    # honest with respect to the observed data range.
    all_lower, all_upper = _metric_yerr(milestones, keys[0]) if keys else ([], [])
    for i, key in enumerate(keys):
        vals: list[float | None] = [m.metrics.get(key) for m in milestones]
        # Per-milestone presence of any spread to draw — either metric_stds
        # or metric_scores. Skips points with neither so the chart never
        # draws a phantom zero-length bar.
        has_spread = [(key in m.metric_stds) or bool(m.metric_scores.get(key)) for m in milestones]
        if i != 0:
            all_lower, all_upper = _metric_yerr(milestones, key)
        valid = [
            (x, v, lo, up, sp)
            for x, v, lo, up, sp in zip(xs, vals, all_lower, all_upper, has_spread, strict=False)
            if v is not None
        ]
        if not valid:
            if i == 0 and require_first:
                raise ValueError(f"no milestone has metric {key!r}")
            continue
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]
        line = ax.plot(
            [x for x, _, _, _, _ in valid],
            [v for _, v, _, _, _ in valid],
            f"{linestyle}{marker}",
            color=color,
            linewidth=linewidth,
            markersize=markersize,
            label=f"{key} {label_suffix}",
        )[0]
        err_xs = [x for x, _, _, _, sp in valid if sp]
        err_ys = [v for _, v, _, _, sp in valid if sp]
        err_lower = [lo for _, _, lo, _, sp in valid if sp]
        err_upper = [up for _, _, _, up, sp in valid if sp]
        if err_xs:
            ax.errorbar(
                err_xs,
                err_ys,
                yerr=[err_lower, err_upper],
                fmt="none",
                ecolor=color,
                elinewidth=linewidth * 0.6,
                capsize=4,
                capthick=linewidth * 0.6,
                alpha=0.75,
                zorder=2,
            )
        # Per-iter scatter overlay where metric_scores is populated. Each
        # raw score plots as a dot at the milestone's x — surfaces single-iter
        # outliers (e.g. one iter breaching a ceiling) that the σ band hides.
        # Min/max are colour-coded (red/green) so a glance distinguishes
        # "one iter breached" from "one iter collapsed".
        for x, ms in zip(xs, milestones, strict=False):
            raw = ms.metric_scores.get(key)
            if not raw:
                continue
            ax.scatter(
                [x] * len(raw),
                raw,
                c=_score_dot_colors(raw, neutral=color),
                s=70,
                edgecolor="white",
                linewidth=1.4,
                alpha=0.95,
                zorder=4,
            )
        drawn.append((key, color, line))
        handles.append(line)
        if i == 0:
            first_valid = [(x, v) for x, v, _, _, _ in valid]
    return drawn, first_valid


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

    primary_palette = _resolve_palette(
        primary_colors, primary_color, _PRIMARY_PALETTE, "primary_color"
    )
    secondary_palette = _resolve_palette(
        secondary_colors, secondary_color, _SECONDARY_PALETTE, "secondary_color"
    )

    xs = list(range(len(milestones)))
    labels = [m.label for m in milestones]

    fig, ax_primary = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax_primary.set_facecolor("white")

    handles: list[Any] = []

    _, first_valid_primary = _draw_metric_series(
        ax_primary,
        primary_keys,
        milestones,
        xs,
        palette=primary_palette,
        markers=_PRIMARY_MARKERS,
        linestyle="-",
        linewidth=2.2,
        markersize=7,
        label_suffix="(left axis)",
        handles=handles,
        require_first=True,
    )

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
        ax_secondary = ax_primary.twinx()
        drawn, _ = _draw_metric_series(
            ax_secondary,
            secondary_keys,
            milestones,
            xs,
            palette=secondary_palette,
            markers=_SECONDARY_MARKERS,
            linestyle="--",
            linewidth=2.0,
            markersize=6,
            label_suffix="(right axis)",
            handles=handles,
        )
        if not drawn:
            ax_secondary.remove()
            ax_secondary = None
        else:
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


# ── milestone bars (cross-stage A/B comparison) ────────────────────────

_DOT_MAX_COLOR = "#16a34a"  # green — best iter (often the breach)
_DOT_MIN_COLOR = "#dc2626"  # red — worst iter (often the collapse)


def _metric_yerr(milestones: Sequence[Milestone], key: str) -> tuple[list[float], list[float]]:
    """Return (lower_err, upper_err) lists for asymmetric matplotlib yerr.

    When ``metric_scores[key]`` is populated, the error bar uses the
    actual observed range (mean−min, max−mean) so the whisker caps sit
    exactly on the data — never above the highest iter. With small n
    and a skewed distribution, ±std overshoots the range (a 5-iter
    [57.14]×4 + [28.57]×1 has mean=51.43, std=12.78, so mean+std=64.21
    sits *above* the actual max 57.14, implying a phantom upper tail).

    Falls back to symmetric ±std when only ``metric_stds[key]`` is given.
    Returns (0.0, 0.0) for the milestone when neither is available.
    """
    lower: list[float] = []
    upper: list[float] = []
    for m in milestones:
        mean = m.metrics.get(key)
        if mean is None:
            lower.append(0.0)
            upper.append(0.0)
            continue
        raw = m.metric_scores.get(key)
        if raw:
            lower.append(mean - min(raw))
            upper.append(max(raw) - mean)
        else:
            std = m.metric_stds.get(key, 0.0)
            lower.append(std)
            upper.append(std)
    return lower, upper


def _count_iters_per_score(scores: Sequence[float]) -> list[tuple[float, int]]:
    """Count how many iters landed on each unique score value.

    Returns ``[(value, iter_count), ...]`` — one entry per distinct
    value in ``scores``, with the number of iters that hit it. The
    chart uses this to render a single dot per value and annotate
    ``×N`` when ``iter_count > 1`` so multiple iters landing on the
    same score (e.g. Stage L's 4× ceiling iters at 57.14) don't
    collapse to a lone dot and silently hide the other 3 data points.
    """
    counts: dict[float, int] = {}
    for v in scores:
        counts[v] = counts.get(v, 0) + 1
    return list(counts.items())


def _score_dot_colors(scores: Sequence[float], *, neutral: str) -> list[str]:
    """Color-code per-iter dots: max → green, min → red, middle → neutral.

    Returns a per-score colour list aligned with ``scores``. When all
    values are equal (no spread to highlight), every dot gets ``neutral``.
    Lets a glance at the chart distinguish *"one iter breached"* from
    *"one iter collapsed"* without reading the legend.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [neutral] * len(scores)
    return [_DOT_MAX_COLOR if v == hi else _DOT_MIN_COLOR if v == lo else neutral for v in scores]


VERDICT_PALETTE: dict[str, str] = {
    "BASELINE": "#7f8c8d",
    "FLAT": "#3498db",
    "NEUTRAL+": "#2ecc71",
    "LIFT": "#9b59b6",
    "REGRESS": "#e74c3c",
    "PENDING": "#bdc3c7",
}


def plot_milestone_bars(
    milestones: Sequence[Milestone],
    *,
    primary_metric: str,
    out_path: str | Path,
    palette: dict[str, str] | None = None,
    thresholds: Sequence[dict[str, Any]] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    pending_value: float | None = None,
    show_descriptions: bool = True,
    value_format: str = "{:.2f}",
    figsize: tuple[float, float] = (13.0, 7.0),
    dpi: int = 140,
    return_fig: bool = False,
) -> Path | Any:
    """Cross-stage A/B bar chart over a hand-curated sequence of milestones.

    Use this when the question is *"which intervention won?"* — each
    milestone is an independent experiment, not a point on a trajectory.
    Complements :func:`plot_milestone_progression` (twin-axis trajectory
    view); pick one based on whether the milestones form a sequence
    (lines) or a comparison (bars).

    Each bar is coloured by ``milestone.verdict`` via ``palette`` (defaults
    to :data:`VERDICT_PALETTE`). Milestones with ``verdict == "PENDING"``
    render as a hatched translucent placeholder at ``pending_value`` (or
    the top threshold) so they read as *"sweep in flight, score TBD"*
    rather than *"0% measured"*. Inside-bar text shows
    ``"<verdict>  n=<n>"`` rotated 90°.

    ``metric_stds`` renders an error bar (mean ± std). ``metric_scores``
    overlays per-iter scatter dots — surfaces single-iter outliers that
    a wide σ band would hide.

    ``thresholds`` is a list of ``{value, label, color}`` dicts; each
    renders as a dashed horizontal reference line with a left-edge label.

    ``show_descriptions`` appends a monospace footer enumerating each
    milestone's ``description`` — useful when the bar labels are short
    stage codes.
    """
    palette = palette if palette is not None else VERDICT_PALETTE
    thresholds = list(thresholds or [])

    labels = [m.label for m in milestones]
    means = [m.metrics[primary_metric] for m in milestones]
    verdicts = [m.verdict or "FLAT" for m in milestones]
    descriptions = [m.description for m in milestones]
    ns = [m.n for m in milestones]
    colors = [palette.get(v, "#34495e") for v in verdicts]

    fig, ax = plt.subplots(figsize=figsize)
    xs = list(range(len(labels)))

    # PENDING rendering: hatched translucent placeholder at the highest
    # threshold (or pending_value) so the bar reads "TBD" rather than 0%.
    placeholder_h = (
        pending_value
        if pending_value is not None
        else max((t["value"] for t in thresholds), default=max(means + [0.0]) * 1.1)
    )
    plot_means = [
        placeholder_h if v == "PENDING" else m for v, m in zip(verdicts, means, strict=False)
    ]
    # Asymmetric yerr — uses min/max range when metric_scores is present,
    # falls back to ±std otherwise. PENDING rows render with no whisker.
    err_lower, err_upper = _metric_yerr(milestones, primary_metric)
    err_lower = [0.0 if v == "PENDING" else e for v, e in zip(verdicts, err_lower, strict=False)]
    err_upper = [0.0 if v == "PENDING" else e for v, e in zip(verdicts, err_upper, strict=False)]
    hatches = ["//" if v == "PENDING" else None for v in verdicts]
    alphas = [0.45 if v == "PENDING" else 1.0 for v in verdicts]

    bars = ax.bar(
        xs,
        plot_means,
        yerr=[err_lower, err_upper],
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    for b, h, a in zip(bars, hatches, alphas, strict=False):
        if h:
            b.set_hatch(h)
        b.set_alpha(a)

    # Per-iter scatter overlay (same machinery as plot_milestone_progression).
    # Min/max are colour-coded (red/green) so a glance distinguishes a
    # breach iter from a collapse iter at the chart level. Repeated iters
    # at the same value collapse to a single dot with an ``×N`` badge so
    # the chart doesn't silently hide stacked data points.
    for x, m in zip(xs, milestones, strict=False):
        raw = m.metric_scores.get(primary_metric)
        if not raw:
            continue
        iters_per_score = _count_iters_per_score(raw)
        unique_vals = [v for v, _ in iters_per_score]
        colors = _score_dot_colors(unique_vals, neutral="black")
        ax.scatter(
            [x] * len(unique_vals),
            unique_vals,
            c=colors,
            s=85,
            edgecolor="white",
            linewidth=1.6,
            zorder=5,
        )
        for (value, iter_count), color in zip(iters_per_score, colors, strict=False):
            if iter_count > 1:
                ax.annotate(
                    f"×{iter_count}",
                    xy=(x, value),
                    xytext=(9, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                    zorder=6,
                )

    for x, m, v, n in zip(xs, means, verdicts, ns, strict=False):
        if v == "PENDING":
            ax.text(
                x,
                placeholder_h + 1.8,
                "TBD",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="#666",
            )
        else:
            # Anchor the value label just above the mean bar top — the
            # number reads "near the mean", not at the top of the
            # whisker (which is the max, not the mean). White bbox masks
            # the whisker line passing behind the text for asymmetric
            # error bars where mean + 1.8 lands inside the whisker range.
            ax.text(
                x,
                m + 1.8,
                value_format.format(m),
                ha="center",
                fontsize=10,
                fontweight="bold",
                zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
            )
        n_label = f"  n={n}" if n is not None else ""
        ax.text(
            x,
            (placeholder_h if v == "PENDING" else m) / 2,
            f"{v}{n_label}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if v != "PENDING" else "#444",
            fontweight="bold",
            rotation=90,
        )

    for t in thresholds:
        ax.axhline(t["value"], linestyle="--", color=t["color"], linewidth=1.0, alpha=0.7)
        ax.text(
            -0.45,
            t["value"] + 0.6,
            f"{t['label']} ({t['value']:.2f})",
            ha="left",
            fontsize=8,
            color=t["color"],
        )

    ax.set_ylim(0, max(placeholder_h + 8, max(plot_means + err_upper) + 10))
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel or primary_metric, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ordered_verdicts = ["BASELINE", "FLAT", "NEUTRAL+", "LIFT", "REGRESS", "PENDING"]
    legend_labels = sorted(
        {v for v in verdicts},
        key=lambda v: ordered_verdicts.index(v) if v in ordered_verdicts else 99,
    )
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=palette.get(v, "#34495e")) for v in legend_labels],
        legend_labels,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        ncol=len(legend_labels),
        bbox_to_anchor=(1.0, 1.0),
    )

    if show_descriptions and any(descriptions):
        footer = "\n".join(
            f"  {lab:<22s} {desc}" for lab, desc in zip(labels, descriptions, strict=False)
        )
        fig.text(
            0.06,
            -0.02,
            "Levers tested:\n" + footer,
            ha="left",
            va="top",
            fontsize=8,
            color="#444",
            family="monospace",
        )
        fig.tight_layout(rect=[0, 0.18, 1, 1])
    else:
        fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if return_fig:
        return fig
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


def _parse_grouped_pairs(
    items: list[str],
    labels: list[str],
    seps: list[int],
    games: list[str],
    *,
    items_name: str,
) -> dict[str, list[tuple[str, str]]]:
    """Group `(item, label)` pairs by game using `seps` as section sizes.

    ``--sep 2 --sep 3 --game A --game B`` means the first 2 ``items``/``labels``
    belong to game A, the next 3 to game B. Raises :class:`typer.BadParameter`
    on mismatched lengths.
    """
    if sum(seps) != len(items) or len(items) != len(labels):
        raise typer.BadParameter(
            f"--sep ({seps}, sum={sum(seps)}) must add up to len({items_name})={len(items)} "
            f"== len(label)={len(labels)}"
        )
    if len(seps) != len(games):
        raise typer.BadParameter(
            f"--sep entries ({len(seps)}) must match --game entries ({len(games)})"
        )
    out: dict[str, list[tuple[str, str]]] = {}
    cursor = 0
    for g, n in zip(games, seps, strict=True):
        out[g] = list(zip(items[cursor : cursor + n], labels[cursor : cursor + n], strict=True))
        cursor += n
    return out


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
    games_to_sweeps = _parse_grouped_pairs(tag, label, sep, game, items_name="tag")
    p = plot_cross_game_scoreboard(
        games_to_sweeps=games_to_sweeps,
        experiments_dir=experiments_dir,
        config_name=config_name,
        out_path=out,
        title=title,
    )
    typer.echo(f"wrote {p}")


@app.command("scoreboard-from-index")
def scoreboard_from_index_cli(
    from_file: str = typer.Option(..., help="Path to the consolidated results.jsonl"),
    out: str = typer.Option(..., help="Output PNG path"),
    game: list[str] = typer.Option(..., "--game", "-g", help="Game name (repeat per panel)"),
    variant: list[str] = typer.Option(
        ...,
        "--variant",
        "-v",
        help="Variant to render as one bar (matched against the 'variant' field). "
        "Repeat per bar; interleave with --label/--sep per game.",
    ),
    label: list[str] = typer.Option(..., "--label", "-l", help="Display label per variant"),
    sep: list[int] = typer.Option(
        ...,
        help="Number of (variant, label) pairs per game — must sum to len(variant)",
    ),
    title: str | None = typer.Option(None, help="Top-level title"),
) -> None:
    """Cross-game scoreboard from a single consolidated index file.

    Unlike ``scoreboard``, this command doesn't need one tag-dir per bar —
    it reads from a single ``--from-file`` and filters by ``(game, variant)``
    to produce each bar. Build the index with
    :func:`autoresearch.results.consolidate`.
    """
    games_to_variants = _parse_grouped_pairs(variant, label, sep, game, items_name="variant")
    p = plot_cross_game_scoreboard_from_index(
        index_path=from_file,
        games_to_variants=games_to_variants,
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
