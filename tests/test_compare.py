"""Tests for autoresearch.compare — comparison plot helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.compare import plot_cross_game_scoreboard, plot_multi_tag_overlay
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
