"""Tests for autoresearch.compare — comparison plot helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.compare import (
    Milestone,
    append_milestone,
    load_milestones_yaml,
    plot_cross_game_scoreboard,
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
