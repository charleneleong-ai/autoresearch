"""Tests for autoresearch.compare — comparison plot helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoresearch.compare import (
    Milestone,
    _best_score_for_variant,
    app,
    append_milestone,
    extract_metrics_from_results_jsonl,
    load_milestones_yaml,
    plot_cross_game_scoreboard,
    plot_cross_game_scoreboard_from_index,
    plot_milestone_bars,
    plot_milestone_progression,
    plot_multi_tag_overlay,
)
from autoresearch.results import log_experiment


@pytest.fixture
def two_sweep_dir(tmp_path: Path) -> Path:
    """Two sweeps (`baseline`, `feature_on`) × one game with 2 iters each."""
    common = dict(
        experiments_dir=str(tmp_path),
        game="game_x",
    )
    # Baseline sweep
    log_experiment(
        **common, tag="baseline", score=4.0, status="KEEP", description="iter 0", runtime_min=10
    )
    log_experiment(
        **common, tag="baseline", score=3.5, status="DISCARD", description="iter 1", runtime_min=10
    )
    # Feature-on sweep
    log_experiment(
        **common, tag="feature_on", score=6.0, status="KEEP", description="iter 0", runtime_min=10
    )
    log_experiment(
        **common,
        tag="feature_on",
        score=5.0,
        status="DISCARD",
        description="iter 1",
        runtime_min=10,
    )
    return tmp_path


def test_overlay_writes_png(two_sweep_dir: Path) -> None:
    out = two_sweep_dir / "out.png"
    p = plot_multi_tag_overlay(
        sweeps=[("baseline", "Baseline"), ("feature_on", "Feature on")],
        experiments_dir=str(two_sweep_dir),
        game="game_x",
        out_path=out,
    )
    assert p == out
    assert out.exists()
    # PNG header check
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_overlay_handles_missing_tag(tmp_path: Path) -> None:
    """Sweep with no data shouldn't crash — just gets no points plotted."""
    out = tmp_path / "out.png"
    plot_multi_tag_overlay(
        sweeps=[("does_not_exist", "Empty")],
        experiments_dir=str(tmp_path),
        out_path=out,
    )
    assert out.exists()


def test_overlay_filters_by_game(two_sweep_dir: Path) -> None:
    log_experiment(
        experiments_dir=str(two_sweep_dir),
        tag="baseline",
        game="other_game",
        score=99.0,
        status="KEEP",
        description="iter X",
        runtime_min=10,
    )
    out = two_sweep_dir / "out.png"
    plot_multi_tag_overlay(
        sweeps=[("baseline", "Baseline")],
        experiments_dir=str(two_sweep_dir),
        game="game_x",  # other_game's score=99 should be ignored
        out_path=out,
    )
    assert out.exists()


def test_scoreboard_writes_png(two_sweep_dir: Path) -> None:
    out = two_sweep_dir / "scoreboard.png"
    plot_cross_game_scoreboard(
        games_to_sweeps={
            "game_x": [("baseline", "Baseline"), ("feature_on", "Feature on")],
        },
        experiments_dir=str(two_sweep_dir),
        out_path=out,
        title="test scoreboard",
    )
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_scoreboard_with_verdicts(two_sweep_dir: Path) -> None:
    out = two_sweep_dir / "scoreboard_v.png"
    plot_cross_game_scoreboard(
        games_to_sweeps={"game_x": [("baseline", "B"), ("feature_on", "F")]},
        experiments_dir=str(two_sweep_dir),
        out_path=out,
        game_titles={"game_x": "Game X (test)"},
        game_verdicts={"game_x": "feature helps (+50%)"},
    )
    assert out.exists()


def test_scoreboard_handles_empty_panel(tmp_path: Path) -> None:
    """A game with no rows still renders (zero-bar panel)."""
    out = tmp_path / "empty.png"
    plot_cross_game_scoreboard(
        games_to_sweeps={"empty_game": [("nothing", "Nothing")]},
        experiments_dir=str(tmp_path),
        out_path=out,
    )
    assert out.exists()


# ── milestone progression ───────────────────────────────────────────────


def test_progression_writes_png(tmp_path: Path) -> None:
    milestones = [
        Milestone(label="vanilla", metrics={"score": 5.0, "halluc": -0.9}),
        Milestone(label="v1", metrics={"score": 7.5, "halluc": -0.7}),
        Milestone(label="v2", metrics={"score": 9.0, "halluc": -0.4}),
    ]
    out = tmp_path / "progression.png"
    p = plot_milestone_progression(
        milestones,
        primary_metric="score",
        secondary_metric="halluc",
        threshold=-0.5,
        threshold_label="ship",
        title="trajectory",
        out_path=out,
    )
    assert p == out
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_progression_secondary_optional(tmp_path: Path) -> None:
    """Without secondary_metric we get a single-axis line chart."""
    milestones = [
        Milestone(label="a", metrics={"score": 1.0}),
        Milestone(label="b", metrics={"score": 2.0}),
    ]
    out = tmp_path / "single.png"
    plot_milestone_progression(
        milestones,
        primary_metric="score",
        out_path=out,
    )
    assert out.exists()


def test_progression_missing_metric_raises(tmp_path: Path) -> None:
    milestones = [Milestone(label="a", metrics={"other": 1.0})]
    with pytest.raises(ValueError, match="no milestone has metric"):
        plot_milestone_progression(
            milestones,
            primary_metric="score",
            out_path=tmp_path / "out.png",
        )


def test_progression_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_milestone_progression(
            [],
            primary_metric="score",
            out_path=tmp_path / "out.png",
        )


# ── error-bar support ───────────────────────────────────────────────────


def test_milestone_accepts_metric_stds_as_optional_field() -> None:
    """``Milestone.metric_stds`` is an opt-in dict keyed by metric name.

    When omitted (existing call sites) it defaults to an empty dict so
    every existing Milestone construction keeps working unchanged.
    """
    m_no_std = Milestone(label="a", metrics={"score": 5.0})
    assert m_no_std.metric_stds == {}

    m_with_std = Milestone(label="b", metrics={"score": 5.0}, metric_stds={"score": 1.2})
    assert m_with_std.metric_stds == {"score": 1.2}


def test_load_milestones_yaml_parses_dict_form_for_std(tmp_path: Path) -> None:
    """New YAML schema: a metric value may be either a scalar (existing) or
    a dict ``{mean: ..., std: ...}``. Scalars stay in ``metrics``; std goes
    into ``metric_stds``."""
    yaml_path = tmp_path / "milestones.yaml"
    yaml_path.write_text(
        """
title: "with stds"
primary_metric: score
milestones:
  - label: a
    metrics:
      score: 5.0
  - label: b
    metrics:
      score:
        mean: 7.0
        std: 1.5
"""
    )
    milestones, meta = load_milestones_yaml(yaml_path)
    assert meta["title"] == "with stds"
    assert milestones[0].metrics == {"score": 5.0}
    assert milestones[0].metric_stds == {}
    assert milestones[1].metrics == {"score": 7.0}
    assert milestones[1].metric_stds == {"score": 1.5}


def test_load_milestones_yaml_rejects_dict_without_mean(tmp_path: Path) -> None:
    """A dict form must include ``mean``. Missing mean = parse error."""
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        """
milestones:
  - label: a
    metrics:
      score:
        std: 1.0
"""
    )
    with pytest.raises(ValueError, match="mean"):
        load_milestones_yaml(yaml_path)


def test_progression_renders_with_error_bars(tmp_path: Path) -> None:
    """The headline behaviour: a Milestone with metric_stds set causes the
    chart to render error bars. We verify the PNG is produced; visual
    inspection in CI is via the committed sample fixture if needed."""
    milestones = [
        Milestone(label="a", metrics={"score": 5.0}, metric_stds={"score": 0.5}),
        Milestone(label="b", metrics={"score": 7.5}, metric_stds={"score": 1.2}),
        Milestone(label="c", metrics={"score": 9.0}),  # no std → no error bar on this point
    ]
    out = tmp_path / "errbars.png"
    plot_milestone_progression(
        milestones,
        primary_metric="score",
        out_path=out,
    )
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_milestone_accepts_metric_scores_as_optional_field() -> None:
    """``Milestone.metric_scores`` is an opt-in per-metric list of raw iter
    scores. Renders as a scatter overlay on the chart so single-iter
    breaches don't get hidden inside a wide mean ± std band."""
    m_no_scores = Milestone(label="a", metrics={"score": 5.0})
    assert m_no_scores.metric_scores == {}

    m_with_scores = Milestone(
        label="b",
        metrics={"score": 42.86},
        metric_stds={"score": 20.2},
        metric_scores={"score": [71.43, 57.14, 28.57, 28.57, 28.57]},
    )
    assert m_with_scores.metric_scores == {"score": [71.43, 57.14, 28.57, 28.57, 28.57]}


def test_load_milestones_yaml_parses_scores_list(tmp_path: Path) -> None:
    """YAML schema: a metric value may carry a ``scores`` list alongside
    ``mean`` / ``std``. Populates ``metric_scores`` for scatter rendering."""
    yaml_path = tmp_path / "milestones.yaml"
    yaml_path.write_text(
        """
primary_metric: score
milestones:
  - label: a
    metrics:
      score:
        mean: 42.86
        std: 20.2
        scores: [71.43, 57.14, 28.57, 28.57, 28.57]
"""
    )
    milestones, _ = load_milestones_yaml(yaml_path)
    assert milestones[0].metrics == {"score": 42.86}
    assert milestones[0].metric_stds == {"score": 20.2}
    assert milestones[0].metric_scores == {"score": [71.43, 57.14, 28.57, 28.57, 28.57]}


def test_progression_overlays_scatter_when_metric_scores_present(tmp_path: Path) -> None:
    """A Milestone with ``metric_scores`` renders per-iter scatter dots on top
    of the mean ± std line — surfaces iter-level outliers (e.g. a single iter
    breaching a ceiling) that the σ band hides."""
    milestones = [
        Milestone(
            label="a",
            metrics={"score": 42.86},
            metric_stds={"score": 20.2},
            metric_scores={"score": [71.43, 57.14, 28.57, 28.57, 28.57]},
        ),
        Milestone(label="b", metrics={"score": 50.0}),  # no scatter — bare marker
    ]
    out = tmp_path / "scatter.png"
    plot_milestone_progression(milestones, primary_metric="score", out_path=out)
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_progression_no_error_bars_when_all_stds_missing(tmp_path: Path) -> None:
    """Backwards compat: existing milestones without metric_stds render
    exactly as before (no error bars). This is essentially the existing
    ``test_progression_writes_png`` path — pinning that the absence of
    metric_stds keeps the chart visually equivalent."""
    milestones = [
        Milestone(label="a", metrics={"score": 1.0}),
        Milestone(label="b", metrics={"score": 2.0}),
        Milestone(label="c", metrics={"score": 3.0}),
    ]
    out = tmp_path / "plain.png"
    plot_milestone_progression(
        milestones,
        primary_metric="score",
        out_path=out,
    )
    assert out.exists()


def test_extract_metrics_from_results_jsonl_supports_std(tmp_path: Path) -> None:
    """``extract_metrics_from_results_jsonl`` should be able to pull both
    a mean and a std for a metric in one call.

    New ``extract_stds`` kwarg mirrors ``extract`` — keys are milestone
    metric names; values are dot-paths into the row. Returns a second dict
    of std values that the caller can pass to ``append_milestone`` /
    Milestone constructor. Keeps the sweep→milestone path one-shot."""
    from autoresearch.compare import extract_metrics_from_results_jsonl

    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text(json.dumps({"evaluation_score": 51.43, "evaluation_score_std": 12.78}) + "\n")
    metrics = extract_metrics_from_results_jsonl(
        jsonl,
        extract={"score": "evaluation_score"},
    )
    assert metrics == {"score": 51.43}

    metrics, stds = extract_metrics_from_results_jsonl(
        jsonl,
        extract={"score": "evaluation_score"},
        extract_stds={"score": "evaluation_score_std"},
    )
    assert metrics == {"score": 51.43}
    assert stds == {"score": 12.78}


def test_load_milestones_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "milestones.yaml"
    yaml_path.write_text(
        """
title: "trajectory"
primary_metric: score
secondary_metric: halluc
threshold: -0.5
threshold_label: ship
milestones:
  - label: vanilla
    description: "no fine-tune"
    metrics:
      score: 5.0
      halluc: -0.9
  - label: v1
    metrics:
      score: 7.5
      halluc: -0.7
        """
    )
    milestones, kwargs = load_milestones_yaml(yaml_path)
    assert len(milestones) == 2
    assert milestones[0].label == "vanilla"
    assert milestones[0].description == "no fine-tune"
    assert milestones[0].metrics["score"] == 5.0
    assert kwargs["primary_metric"] == "score"
    assert kwargs["threshold"] == -0.5

    # Round-trip into the plot fn
    out = tmp_path / "from_yaml.png"
    plot_milestone_progression(
        milestones,
        out_path=out,
        **{
            k: v
            for k, v in kwargs.items()
            if k != "title"  # title is its own param
        },
        title=kwargs.get("title"),
    )
    assert out.exists()


def test_load_milestones_yaml_missing_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_milestones: []\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_milestones_yaml(bad)


def test_append_milestone_creates_file(tmp_path: Path) -> None:
    """First append on a missing path creates the file with one entry."""
    yaml_path = tmp_path / "milestones.yaml"
    p = append_milestone(
        yaml_path,
        label="vanilla",
        metrics={"mean_total": 8.35, "no_halluc": -0.88},
        description="base, no fine-tune",
    )
    assert p == yaml_path
    assert yaml_path.exists()

    milestones, kwargs = load_milestones_yaml(yaml_path)
    assert len(milestones) == 1
    assert milestones[0].label == "vanilla"
    assert milestones[0].description == "base, no fine-tune"
    assert milestones[0].metrics == {"mean_total": 8.35, "no_halluc": -0.88}
    assert kwargs == {}  # no top-level metadata seeded


def test_append_milestone_appends_to_existing(tmp_path: Path) -> None:
    """Second append adds to existing list; first entry preserved."""
    yaml_path = tmp_path / "milestones.yaml"
    append_milestone(yaml_path, label="v1", metrics={"score": 5.0})
    append_milestone(yaml_path, label="v2", metrics={"score": 7.5})

    milestones, _ = load_milestones_yaml(yaml_path)
    assert [m.label for m in milestones] == ["v1", "v2"]
    assert [m.metrics["score"] for m in milestones] == [5.0, 7.5]


def test_append_milestone_preserves_top_level_metadata(tmp_path: Path) -> None:
    """Top-level title/primary_metric/threshold survive appends — they're
    set once when seeding the file, not re-asserted per milestone."""
    yaml_path = tmp_path / "milestones.yaml"
    yaml_path.write_text(
        "title: trajectory\n"
        "primary_metric: score\n"
        "secondary_metric: halluc\n"
        "threshold: -0.5\n"
        "threshold_label: ship\n"
        "milestones:\n"
        "  - label: vanilla\n"
        "    metrics:\n"
        "      score: 5.0\n"
        "      halluc: -0.9\n"
    )
    append_milestone(yaml_path, label="v1", metrics={"score": 7.5, "halluc": -0.7})

    milestones, kwargs = load_milestones_yaml(yaml_path)
    assert [m.label for m in milestones] == ["vanilla", "v1"]
    assert kwargs["title"] == "trajectory"
    assert kwargs["primary_metric"] == "score"
    assert kwargs["threshold"] == -0.5
    assert kwargs["threshold_label"] == "ship"


def test_append_milestone_rejects_non_mapping_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        append_milestone(bad, label="x", metrics={"score": 1.0})


# ── multi-metric per axis ─────────────────────────────────────────────


def test_progression_multi_primary(tmp_path: Path) -> None:
    """A list of primary metrics stacks N lines on the left axis."""
    milestones = [
        Milestone(label="v0", metrics={"a": 1.0, "b": 2.0, "c": 3.0}),
        Milestone(label="v1", metrics={"a": 1.5, "b": 2.5, "c": 3.5}),
    ]
    out = tmp_path / "stacked.png"
    plot_milestone_progression(
        milestones,
        primary_metric=["a", "b", "c"],
        out_path=out,
    )
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_progression_multi_primary_and_secondary(tmp_path: Path) -> None:
    milestones = [
        Milestone(label="v0", metrics={"a": 1.0, "b": 2.0, "x": -0.9, "y": -0.7}),
        Milestone(label="v1", metrics={"a": 1.5, "b": 2.5, "x": -0.6, "y": -0.5}),
    ]
    out = tmp_path / "both_stacked.png"
    plot_milestone_progression(
        milestones,
        primary_metric=["a", "b"],
        secondary_metric=["x", "y"],
        threshold=-0.5,
        threshold_label="ship",
        out_path=out,
    )
    assert out.exists()


def test_progression_string_metric_still_works(tmp_path: Path) -> None:
    """Single-metric (str) call path is unchanged for backwards compat."""
    milestones = [Milestone(label=f"v{i}", metrics={"score": float(i)}) for i in range(3)]
    out = tmp_path / "single.png"
    plot_milestone_progression(
        milestones,
        primary_metric="score",
        out_path=out,
    )
    assert out.exists()


def test_progression_mutual_exclusion_color_vs_colors(tmp_path: Path) -> None:
    milestones = [Milestone(label="a", metrics={"score": 1.0})]
    with pytest.raises(ValueError, match="primary_color or primary_colors"):
        plot_milestone_progression(
            milestones,
            primary_metric="score",
            primary_color="#000",
            primary_colors=["#fff", "#000"],
            out_path=tmp_path / "x.png",
        )


# ── milestone bars ─────────────────────────────────────────────────────


def test_milestone_accepts_verdict_and_n_as_optional_fields() -> None:
    """``Milestone.verdict`` + ``Milestone.n`` are opt-in; they default to None
    so existing call sites are unaffected. Bars-form plot consumes them."""
    m_bare = Milestone(label="a", metrics={"score": 5.0})
    assert m_bare.verdict is None
    assert m_bare.n is None

    m_full = Milestone(label="b", metrics={"score": 5.0}, verdict="LIFT", n=5)
    assert m_full.verdict == "LIFT"
    assert m_full.n == 5


def test_load_milestones_yaml_parses_verdict_and_n(tmp_path: Path) -> None:
    """YAML schema: top-level ``verdict`` + ``n`` per milestone populate the
    new fields."""
    yaml_path = tmp_path / "milestones.yaml"
    yaml_path.write_text(
        """
primary_metric: score
milestones:
  - label: a
    verdict: BASELINE
    n: 3
    metrics:
      score: 5.0
  - label: b
    metrics:
      score: 6.0
"""
    )
    milestones, _ = load_milestones_yaml(yaml_path)
    assert milestones[0].verdict == "BASELINE"
    assert milestones[0].n == 3
    assert milestones[1].verdict is None
    assert milestones[1].n is None


def test_plot_milestone_bars_writes_png(tmp_path: Path) -> None:
    """Smoke: writes a valid PNG file for a minimal milestone list."""
    out = tmp_path / "bars.png"
    plot_milestone_bars(
        [Milestone(label="a", metrics={"score": 5.0}, verdict="FLAT")],
        primary_metric="score",
        out_path=out,
    )
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_milestone_bars_colors_by_verdict(tmp_path: Path) -> None:
    """Each bar's facecolor must match the verdict palette so the chart
    conveys outcome category at a glance (BASELINE grey, FLAT blue,
    NEUTRAL+ green, REGRESS red, LIFT purple, PENDING light grey)."""
    palette = {
        "BASELINE": "#7f8c8d",
        "REGRESS": "#e74c3c",
        "LIFT": "#9b59b6",
    }
    milestones = [
        Milestone(label="a", metrics={"score": 5.0}, verdict="BASELINE"),
        Milestone(label="b", metrics={"score": 3.0}, verdict="REGRESS"),
        Milestone(label="c", metrics={"score": 8.0}, verdict="LIFT"),
    ]
    fig = plot_milestone_bars(
        milestones,
        primary_metric="score",
        palette=palette,
        out_path=tmp_path / "v.png",
        return_fig=True,
    )
    bars = [c for c in fig.axes[0].containers if hasattr(c, "patches")][0]
    import matplotlib.colors as mcolors

    actual = [mcolors.to_hex(p.get_facecolor()) for p in bars.patches]
    expected = [palette[v] for v in ["BASELINE", "REGRESS", "LIFT"]]
    assert actual == expected


def test_plot_milestone_bars_renders_pending_as_hatched_placeholder(tmp_path: Path) -> None:
    """A PENDING milestone (sweep in flight, score unknown) renders as a
    hatched translucent bar at the chart-top ceiling — reads as 'TBD' not
    as a measured 0%."""
    milestones = [
        Milestone(label="done", metrics={"score": 5.0}, verdict="FLAT"),
        Milestone(label="next", metrics={"score": 0.0}, verdict="PENDING"),
    ]
    fig = plot_milestone_bars(
        milestones,
        primary_metric="score",
        out_path=tmp_path / "p.png",
        return_fig=True,
    )
    bars = [c for c in fig.axes[0].containers if hasattr(c, "patches")][0]
    # The PENDING bar should be hatched and the FLAT bar shouldn't.
    assert bars.patches[0].get_hatch() is None
    assert bars.patches[1].get_hatch() is not None
    assert bars.patches[1].get_alpha() and bars.patches[1].get_alpha() < 1.0


def test_plot_milestone_bars_renders_threshold_lines(tmp_path: Path) -> None:
    """``thresholds=[{value, label, color}]`` renders horizontal reference
    lines — the ceilings the milestones are trying to beat."""
    fig = plot_milestone_bars(
        [Milestone(label="a", metrics={"score": 50.0}, verdict="FLAT")],
        primary_metric="score",
        thresholds=[{"value": 57.14, "label": "M4 ceiling", "color": "tab:green"}],
        out_path=tmp_path / "t.png",
        return_fig=True,
    )
    hlines = [ln for ln in fig.axes[0].get_lines() if ln.get_linestyle() == "--"]
    assert any(abs(ln.get_ydata()[0] - 57.14) < 1e-6 for ln in hlines)


def test_dedupe_scores_groups_duplicates_with_count() -> None:
    """Stacked-dot count helper: collapses repeated scores into
    ``(value, count)`` pairs so the chart can annotate ``×N`` next to
    a dot that's hiding multiple iters."""
    from autoresearch.compare import _dedupe_scores

    # Stage L pattern: 4 ceiling iters + 1 dropout
    out = _dedupe_scores([57.14, 57.14, 57.14, 28.57, 57.14])
    assert sorted(out) == [(28.57, 1), (57.14, 4)]

    # All distinct: every dot is its own (value, 1)
    assert sorted(_dedupe_scores([1.0, 2.0, 3.0])) == [(1.0, 1), (2.0, 1), (3.0, 1)]

    # Empty list → empty
    assert _dedupe_scores([]) == []


def test_plot_milestone_bars_value_format_controls_label(tmp_path: Path) -> None:
    """``value_format`` is applied to the value label printed above each bar.
    Default is ``'{:.2f}'``; passing ``'{:.2f}%'`` adds the percent sign."""
    out = tmp_path / "fmt.png"
    fig = plot_milestone_bars(
        [Milestone(label="a", metrics={"score": 42.86}, verdict="FLAT")],
        primary_metric="score",
        value_format="{:.2f}%",
        out_path=out,
        return_fig=True,
    )
    label_texts = {t.get_text() for t in fig.axes[0].texts}
    assert "42.86%" in label_texts
    assert "42.86" not in label_texts  # bare value not rendered


def test_metric_yerr_uses_min_max_when_scores_present() -> None:
    """When metric_scores is populated, the error bar uses the actual
    min/max range — symmetric ±std overshoots the observed data when
    n is small and the distribution is skewed (e.g. Stage L's [57.14]×4
    + [28.57]×1 gives mean=51.43, std=12.78, but mean+std=64.21 sits
    above the actual max of 57.14)."""
    from autoresearch.compare import _metric_yerr

    ms = [
        Milestone(
            label="L",
            metrics={"score": 51.43},
            metric_stds={"score": 12.78},
            metric_scores={"score": [57.14, 57.14, 57.14, 28.57, 57.14]},
        ),
        Milestone(label="P", metrics={"score": 57.14}, metric_stds={"score": 0.0}),
        Milestone(label="bare", metrics={"score": 50.0}),
    ]
    lower, upper = _metric_yerr(ms, "score")
    # L: mean - min = 51.43 - 28.57 = 22.86; max - mean = 57.14 - 51.43 = 5.71
    assert lower[0] == pytest.approx(22.86, abs=0.01)
    assert upper[0] == pytest.approx(5.71, abs=0.01)
    # P: only std → symmetric (both = std)
    assert lower[1] == 0.0 and upper[1] == 0.0
    # bare: no std, no scores → no whisker (0.0 on both sides)
    assert lower[2] == 0.0 and upper[2] == 0.0


def test_score_dot_colors_marks_min_and_max() -> None:
    """Helper that color-codes per-iter scores: max breach → green,
    min collapse → red, middle scores → neutral. When all scores are
    equal (no spread) every dot stays neutral."""
    from autoresearch.compare import _score_dot_colors

    # Spread → extremes coloured.
    colors = _score_dot_colors([71.43, 57.14, 28.57, 28.57, 28.57], neutral="#000")
    assert colors == ["#16a34a", "#000", "#dc2626", "#dc2626", "#dc2626"]

    # All equal → no extremes coloured.
    flat = _score_dot_colors([57.14, 57.14, 57.14], neutral="#000")
    assert flat == ["#000", "#000", "#000"]

    # Two values → both are extremes.
    two = _score_dot_colors([10.0, 20.0], neutral="#000")
    assert two == ["#dc2626", "#16a34a"]


def test_plot_milestone_bars_overlays_error_bars_and_scatter(tmp_path: Path) -> None:
    """metric_stds renders an error bar; metric_scores renders scatter dots —
    same machinery as plot_milestone_progression, just on bars."""
    out = tmp_path / "scatter.png"
    plot_milestone_bars(
        [
            Milestone(
                label="q",
                metrics={"score": 42.86},
                metric_stds={"score": 20.2},
                metric_scores={"score": [71.43, 57.14, 28.57, 28.57, 28.57]},
                verdict="REGRESS",
                n=5,
            )
        ],
        primary_metric="score",
        out_path=out,
    )
    assert out.exists()


# ── extract from results.jsonl ────────────────────────────────────────


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_extract_last_row(tmp_path: Path) -> None:
    rows = [
        {"score": 5.0, "metrics": {"heldout": {"mean_total": 8.5, "no_halluc": -0.9}}},
        {"score": 7.5, "metrics": {"heldout": {"mean_total": 10.9, "no_halluc": -0.5}}},
    ]
    path = _write_jsonl(tmp_path / "results.jsonl", rows)
    out = extract_metrics_from_results_jsonl(
        path,
        {
            "mean_total": "metrics.heldout.mean_total",
            "no_halluc": "metrics.heldout.no_halluc",
        },
        row="last",
    )
    assert out == {"mean_total": 10.9, "no_halluc": -0.5}


def test_extract_best_row_uses_get_score(tmp_path: Path) -> None:
    """`row='best'` selects via get_score, not chronologically."""
    rows = [
        {"score": 9.0, "metrics": {"heldout": {"mean_total": 11.0}}},
        {"score": 7.5, "metrics": {"heldout": {"mean_total": 10.0}}},  # newer but lower score
    ]
    path = _write_jsonl(tmp_path / "results.jsonl", rows)
    out = extract_metrics_from_results_jsonl(
        path, {"mean_total": "metrics.heldout.mean_total"}, row="best"
    )
    assert out == {"mean_total": 11.0}


def test_extract_missing_path_raises(tmp_path: Path) -> None:
    rows = [{"score": 1.0, "metrics": {"heldout": {"mean_total": 1.0}}}]
    path = _write_jsonl(tmp_path / "results.jsonl", rows)
    with pytest.raises(KeyError, match="missing_segment"):
        extract_metrics_from_results_jsonl(
            path, {"x": "metrics.heldout.missing_segment"}, row="last"
        )


def test_extract_non_numeric_raises(tmp_path: Path) -> None:
    rows = [{"score": 1.0, "status": "BASELINE"}]
    path = _write_jsonl(tmp_path / "results.jsonl", rows)
    with pytest.raises(TypeError, match="non-numeric"):
        extract_metrics_from_results_jsonl(path, {"x": "status"}, row="last")


def test_extract_empty_jsonl_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        extract_metrics_from_results_jsonl(path, {"x": "score"}, row="last")


def test_extract_invalid_row_arg(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "results.jsonl", [{"score": 1.0}])
    with pytest.raises(ValueError, match="'last' or 'best'"):
        extract_metrics_from_results_jsonl(path, {"x": "score"}, row="median")


# ── scoreboard-from-index ──────────────────────────────────────────────


def _write_index(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a consolidated jsonl index."""
    out = tmp_path / "consolidated.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out


def test_scoreboard_from_index_writes_png(tmp_path: Path) -> None:
    """Renders one bar per (game, variant) from a single index file."""
    index = _write_index(
        tmp_path,
        [
            {"variant": "stage_a", "game": "game_x", "evaluation_score": 28.5},
            {"variant": "stage_d", "game": "game_x", "evaluation_score": 57.1},
            {"variant": "stage_a", "game": "game_y", "evaluation_score": 54.5},
            {"variant": "stage_b", "game": "game_y", "evaluation_score": 63.6},
        ],
    )
    out = tmp_path / "scoreboard.png"
    plot_cross_game_scoreboard_from_index(
        index_path=index,
        games_to_variants={
            "game_x": [("stage_a", "Stage A"), ("stage_d", "Stage D")],
            "game_y": [("stage_a", "Stage A"), ("stage_b", "Stage B")],
        },
        out_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_best_score_for_variant_returns_max() -> None:
    """`_best_score_for_variant` picks the largest score across matching rows."""
    rows = [
        {"variant": "stage_a", "game": "g", "evaluation_score": 14.29},
        {"variant": "stage_a", "game": "g", "evaluation_score": 28.57},  # best
        {"variant": "stage_a", "game": "g", "evaluation_score": 0.0},
        {"variant": "stage_b", "game": "g", "evaluation_score": 99.0},  # wrong variant
        {"variant": "stage_a", "game": "other", "evaluation_score": 99.0},  # wrong game
    ]
    assert _best_score_for_variant(rows, game="g", variant="stage_a") == 28.57


def test_best_score_for_variant_missing_returns_zero() -> None:
    """No matching rows → 0.0 (not a crash, not NaN)."""
    rows = [{"variant": "x", "game": "g", "evaluation_score": 1.0}]
    assert _best_score_for_variant(rows, game="g", variant="missing") == 0.0


def test_scoreboard_from_index_zero_when_variant_missing(tmp_path: Path) -> None:
    """Requested (game, variant) not in the index → 0.0 bar, no crash."""
    index = _write_index(tmp_path, [{"variant": "x", "game": "g", "evaluation_score": 1.0}])
    out = tmp_path / "scoreboard.png"
    plot_cross_game_scoreboard_from_index(
        index_path=index,
        games_to_variants={"g": [("missing", "Missing")]},
        out_path=out,
    )
    assert out.exists()


def test_scoreboard_from_index_cli(tmp_path: Path) -> None:
    """The Typer subcommand is wired with --game/--variant/--label/--sep."""
    index = _write_index(
        tmp_path,
        [
            {"variant": "stage_a", "game": "g", "evaluation_score": 1.0},
            {"variant": "stage_b", "game": "g", "evaluation_score": 2.0},
        ],
    )
    out = tmp_path / "scoreboard.png"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scoreboard-from-index",
            "--from-file",
            str(index),
            "--out",
            str(out),
            "--game",
            "g",
            "--variant",
            "stage_a",
            "--label",
            "Stage A",
            "--variant",
            "stage_b",
            "--label",
            "Stage B",
            "--sep",
            "2",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
    assert out.exists()
