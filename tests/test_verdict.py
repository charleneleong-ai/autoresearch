"""Tests for autoresearch.verdict — spec parsing, computation, markdown rendering, polling."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from autoresearch.results import log_experiment
from autoresearch.verdict import (
    GameSpec,
    VerdictSpec,
    _treatments_ready,
    compute_verdict,
    format_markdown,
    load_spec,
)


@pytest.fixture
def two_game_spec() -> VerdictSpec:
    return VerdictSpec(
        threshold_pct=10.0,
        labels={"baseline": "A", "comparison": "C", "treatment": "D"},
        games=[
            GameSpec(
                name="game_x",
                display="Game X",
                baseline="b_tag",
                comparison="c_tag",
                treatment="t_tag",
            ),
            GameSpec(
                name="game_y",
                display="Game Y",
                baseline="b_tag",
                comparison=None,
                treatment="t_tag",
            ),
        ],
    )


@pytest.fixture
def seeded_dir(tmp_path: Path) -> Path:
    """Two games × three tags so all delta paths exercise."""
    log_experiment(
        experiments_dir=str(tmp_path),
        tag="b_tag",
        game="game_x",
        score=10.0,
        status="KEEP",
        description="x baseline",
    )
    log_experiment(
        experiments_dir=str(tmp_path),
        tag="c_tag",
        game="game_x",
        score=15.0,
        status="KEEP",
        description="x comparison",
    )
    # 18 vs 15 = +20% — HELPS
    log_experiment(
        experiments_dir=str(tmp_path),
        tag="t_tag",
        game="game_x",
        score=18.0,
        status="KEEP",
        description="x treatment",
    )

    log_experiment(
        experiments_dir=str(tmp_path),
        tag="b_tag",
        game="game_y",
        score=4.0,
        status="KEEP",
        description="y baseline",
    )
    # 4.2 vs 4.0 = +5% — NEUTRAL (no comparison for game_y, fall back to baseline)
    log_experiment(
        experiments_dir=str(tmp_path),
        tag="t_tag",
        game="game_y",
        score=4.2,
        status="KEEP",
        description="y treatment",
    )
    return tmp_path


def test_load_spec_minimal(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "threshold_pct": 5,
                "games": [
                    {"name": "g", "baseline": "a", "treatment": "b"},
                ],
            }
        )
    )
    s = load_spec(p)
    assert s.threshold_pct == 5
    assert s.games[0].comparison is None
    assert s.games[0].display == "g"  # default to name


def test_load_spec_with_optional_fields(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "threshold_pct": 12.5,
                "config_name": "gemma",
                "labels": {"baseline": "A", "treatment": "B"},
                "games": [
                    {
                        "name": "g",
                        "display": "Game",
                        "baseline": "a",
                        "comparison": "c",
                        "treatment": "b",
                    },
                ],
            }
        )
    )
    s = load_spec(p)
    assert s.config_name == "gemma"
    assert s.labels == {"baseline": "A", "treatment": "B"}
    assert s.games[0].display == "Game"
    assert s.games[0].comparison == "c"


def test_compute_verdict_three_way_helps(two_game_spec: VerdictSpec, seeded_dir: Path) -> None:
    verdicts = compute_verdict(two_game_spec, experiments_dir=str(seeded_dir))
    gx, gy = verdicts
    assert gx.baseline_best == 10.0
    assert gx.comparison_best == 15.0
    assert gx.treatment_best == 18.0
    assert abs(gx.delta_vs_comparison_pct - 20.0) < 0.01
    assert abs(gx.delta_vs_baseline_pct - 80.0) < 0.01
    assert gx.classify(10.0) == "HELPS"


def test_compute_verdict_two_way_neutral(two_game_spec: VerdictSpec, seeded_dir: Path) -> None:
    """Game Y has no comparison — falls back to baseline delta (+5%, NEUTRAL at thresh 10)."""
    verdicts = compute_verdict(two_game_spec, experiments_dir=str(seeded_dir))
    _, gy = verdicts
    assert gy.comparison_best is None
    assert gy.delta_vs_comparison_pct is None
    assert abs(gy.delta_vs_baseline_pct - 5.0) < 0.01
    assert gy.classify(10.0) == "NEUTRAL"


def test_compute_verdict_handles_missing_treatment(
    two_game_spec: VerdictSpec, tmp_path: Path
) -> None:
    """No data at all — treatment_best is None, classify returns ?."""
    verdicts = compute_verdict(two_game_spec, experiments_dir=str(tmp_path))
    assert all(v.treatment_best is None for v in verdicts)
    assert all(v.classify(10.0) == "?" for v in verdicts)


def test_compute_verdict_zero_baseline_skips_delta(
    two_game_spec: VerdictSpec, tmp_path: Path
) -> None:
    """Baseline of 0 must not cause divide-by-zero — delta returns None."""
    log_experiment(
        experiments_dir=str(tmp_path),
        tag="b_tag",
        game="game_x",
        score=0.0,
        status="KEEP",
        description="zero",
    )
    log_experiment(
        experiments_dir=str(tmp_path),
        tag="t_tag",
        game="game_x",
        score=5.0,
        status="KEEP",
        description="five",
    )
    spec = VerdictSpec(
        threshold_pct=10.0,
        labels={},
        games=[
            GameSpec(
                name="game_x", display="X", baseline="b_tag", comparison=None, treatment="t_tag"
            )
        ],
    )
    v = compute_verdict(spec, experiments_dir=str(tmp_path))[0]
    assert v.delta_vs_baseline_pct is None
    assert v.classify(10.0) == "?"


def test_format_markdown_three_way(two_game_spec: VerdictSpec, seeded_dir: Path) -> None:
    verdicts = compute_verdict(two_game_spec, experiments_dir=str(seeded_dir))
    out = format_markdown(verdicts, two_game_spec, title="Test verdict")
    assert "## Test verdict" in out
    assert "**HELPS**" in out
    assert "**NEUTRAL**" in out
    assert "Game X" in out and "Game Y" in out
    assert "+20%" in out  # game X delta vs comparison
    # PR-comment-friendly markdown: three-way header has both Δ columns
    assert "Δ vs C" in out
    assert "Δ vs A" in out


def test_format_markdown_timed_out(two_game_spec: VerdictSpec, seeded_dir: Path) -> None:
    verdicts = compute_verdict(two_game_spec, experiments_dir=str(seeded_dir))
    out = format_markdown(verdicts, two_game_spec, timed_out=True)
    assert "(timed out)" in out


def test_format_markdown_missing_data(two_game_spec: VerdictSpec, tmp_path: Path) -> None:
    """No data: should render the table with _(missing)_ cells, no crash."""
    verdicts = compute_verdict(two_game_spec, experiments_dir=str(tmp_path))
    out = format_markdown(verdicts, two_game_spec)
    assert "_(missing)_" in out
    assert "Insufficient data" in out


def test_treatments_ready_returns_pending(two_game_spec: VerdictSpec, seeded_dir: Path) -> None:
    # seeded_dir has 1 row per (treatment, game); target=2 means both pending
    ready, pending = _treatments_ready(two_game_spec, str(seeded_dir), target_iters=2)
    assert not ready
    assert len(pending) == 2
    assert "t_tag/game_x" in pending[0] or "t_tag/game_y" in pending[0]


def test_treatments_ready_when_complete(two_game_spec: VerdictSpec, seeded_dir: Path) -> None:
    ready, pending = _treatments_ready(two_game_spec, str(seeded_dir), target_iters=1)
    assert ready
    assert pending == []
