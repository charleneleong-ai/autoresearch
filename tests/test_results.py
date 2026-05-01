"""Smoke tests for autoresearch.results — flat + per-config layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.results import (
    filter_by_game,
    get_score,
    load_results,
    log_experiment,
    tag_dir,
)


def test_tag_dir_flat(tmp_path: Path) -> None:
    d = tag_dir(tmp_path, tag="my_sweep")
    assert d == tmp_path / "my_sweep"
    assert d.is_dir()


def test_tag_dir_per_config(tmp_path: Path) -> None:
    d = tag_dir(tmp_path, tag="my_sweep", config_name="gemma")
    assert d == tmp_path / "my_sweep" / "gemma"
    assert d.is_dir()


def test_tag_dir_normalises_spaces_and_case(tmp_path: Path) -> None:
    d = tag_dir(tmp_path, tag="My Sweep", config_name="Gemma 4")
    assert d == tmp_path / "my_sweep" / "gemma_4"


def test_log_then_load_round_trip(tmp_path: Path) -> None:
    log_experiment(
        experiments_dir=tmp_path,
        tag="my_sweep",
        game="dd_explainer",
        score=12.5,
        steps=200,
        status="KEEP",
        description="baseline",
    )
    rows = load_results(tmp_path, "my_sweep")
    assert len(rows) == 1
    r = rows[0]
    assert r["game"] == "dd_explainer"
    assert r["status"] == "KEEP"
    assert r["score"] == 12.5
    assert r["evaluation_score"] == 12.5  # both fields written for cross-project compat
    assert r["experiment"] == 0


def test_log_per_game_experiment_numbering(tmp_path: Path) -> None:
    for i in range(3):
        log_experiment(
            experiments_dir=tmp_path,
            tag="t",
            game="A",
            score=i,
            status="KEEP",
        )
    log_experiment(experiments_dir=tmp_path, tag="t", game="B", score=99, status="KEEP")
    rows = load_results(tmp_path, "t")
    a_rows = [r for r in rows if r["game"] == "A"]
    b_rows = [r for r in rows if r["game"] == "B"]
    assert [r["experiment"] for r in a_rows] == [0, 1, 2]
    assert b_rows[0]["experiment"] == 0  # B gets its own counter


def test_per_config_isolation(tmp_path: Path) -> None:
    log_experiment(
        experiments_dir=tmp_path,
        tag="t",
        config_name="gemma",
        game="A",
        score=1,
        status="KEEP",
    )
    log_experiment(
        experiments_dir=tmp_path,
        tag="t",
        config_name="qwen",
        game="A",
        score=99,
        status="KEEP",
    )
    gemma = load_results(tmp_path, "t", "gemma")
    qwen = load_results(tmp_path, "t", "qwen")
    flat = load_results(tmp_path, "t")
    assert len(gemma) == 1 and gemma[0]["score"] == 1
    assert len(qwen) == 1 and qwen[0]["score"] == 99
    assert flat == []  # flat layout sees nothing — sub-dirs are isolated


def test_load_returns_empty_when_no_file(tmp_path: Path) -> None:
    assert load_results(tmp_path, "missing") == []


# ── score / game-filter helpers ────────────────────────────────────────


def test_get_score_prefers_evaluation_score() -> None:
    assert get_score({"evaluation_score": 5.0, "score": 1.0}) == 5.0


def test_get_score_falls_back_to_score() -> None:
    assert get_score({"score": 1.0}) == 1.0


def test_get_score_returns_zero_when_missing() -> None:
    assert get_score({}) == 0.0


def test_get_score_explicit_field_wins() -> None:
    row = {"score": 1.0, "evaluation_score": 5.0, "custom": 9.0}
    assert get_score(row, score_field="custom") == 9.0


def test_get_score_explicit_field_falls_back_when_absent() -> None:
    row = {"evaluation_score": 5.0}  # no "custom"
    assert get_score(row, score_field="custom") == 5.0


def test_filter_by_game_filters() -> None:
    rows = [{"game": "a"}, {"game": "b"}, {"game": "a"}]
    assert filter_by_game(rows, "a") == [{"game": "a"}, {"game": "a"}]


def test_filter_by_game_none_returns_all() -> None:
    rows = [{"game": "a"}, {"game": "b"}]
    assert filter_by_game(rows, None) == rows


def test_filter_by_game_empty_string_returns_all() -> None:
    rows = [{"game": "a"}]
    assert filter_by_game(rows, "") == rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
