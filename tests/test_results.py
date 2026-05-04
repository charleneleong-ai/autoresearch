"""Smoke tests for autoresearch.results — flat + per-config layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.results import (
    KILL_CATEGORIES,
    KILL_GPU_HANG,
    KILL_GPU_SLOW,
    KILL_GPU_SPIKE,
    KILL_GPU_UNDERSIZED,
    KILL_GPU_WASTED,
    KILL_LOSS_BLOWUP,
    KILL_NO_LEARNING,
    KILL_POLICY_DIVERGENCE,
    KILL_UNKNOWN,
    categorize_kill_reason,
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


# ── categorize_kill_reason ─────────────────────────────────────────────


def test_categorize_kill_reason_policy_with_kl_value() -> None:
    cat, extras = categorize_kill_reason("|kl|=0.7 suggests policy divergence")
    assert cat == KILL_POLICY_DIVERGENCE
    assert extras == {"kl": "0.7"}


def test_categorize_kill_reason_policy_without_value() -> None:
    cat, extras = categorize_kill_reason("policy collapse — kl divergence in head")
    assert cat == KILL_POLICY_DIVERGENCE
    assert extras == {}


def test_categorize_kill_reason_loss_blowup() -> None:
    cat, extras = categorize_kill_reason("|loss|=12.3 suggests divergence")
    assert cat == KILL_LOSS_BLOWUP
    assert extras == {"loss": "12.3"}


def test_categorize_kill_reason_loss_blow_word() -> None:
    cat, _ = categorize_kill_reason("loss blow-up at step 4")
    assert cat == KILL_LOSS_BLOWUP


def test_categorize_kill_reason_gpu_spike() -> None:
    cat, extras = categorize_kill_reason("step_time spike 210.5s on step 4")
    assert cat == KILL_GPU_SPIKE
    assert extras == {"step_time": "210.5"}


def test_categorize_kill_reason_gpu_slow_with_value() -> None:
    cat, extras = categorize_kill_reason("mean step_time over last 5 = 145.2s > 130.0s")
    assert cat == KILL_GPU_SLOW
    assert extras == {"step_time": "145.2"}


def test_categorize_kill_reason_gpu_hang() -> None:
    cat, _ = categorize_kill_reason("GPU util 3% < 8% for 5min+ — likely hang")
    assert cat == KILL_GPU_HANG


def test_categorize_kill_reason_gpu_wasted() -> None:
    cat, _ = categorize_kill_reason("GPU util sustained <35% for 15min+ — wasted compute")
    assert cat == KILL_GPU_WASTED


def test_categorize_kill_reason_gpu_undersized() -> None:
    cat, _ = categorize_kill_reason("peak GPU mem 28% < 35% for 30min+ — undersized config")
    assert cat == KILL_GPU_UNDERSIZED


def test_categorize_kill_reason_no_learning() -> None:
    cat, _ = categorize_kill_reason("no reward > baseline-1 (4.50) in last 25 steps; max=3.20")
    assert cat == KILL_NO_LEARNING


def test_categorize_kill_reason_unknown_falls_through() -> None:
    cat, extras = categorize_kill_reason("something we have not seen before")
    assert cat == KILL_UNKNOWN
    assert extras == {}


def test_categorize_kill_reason_empty_input() -> None:
    assert categorize_kill_reason("") == (KILL_UNKNOWN, {})
    assert categorize_kill_reason(None) == (KILL_UNKNOWN, {})  # type: ignore[arg-type]


def test_categorize_kill_reason_categories_constant_is_complete() -> None:
    # Every category code returned by the function must be in the public
    # KILL_CATEGORIES tuple — otherwise switch-on-category callers will
    # silently break when a new code is added.
    samples = [
        "|kl|=0.5",
        "|loss|=12 divergence",
        "step_time spike 200s",
        "mean step_time slow",
        "hang",
        "wasted compute",
        "peak mem undersized",
        "no reward",
        "novel reason",
    ]
    for s in samples:
        cat, _ = categorize_kill_reason(s)
        assert cat in KILL_CATEGORIES, f"category {cat!r} from {s!r} missing from KILL_CATEGORIES"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
