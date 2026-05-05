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
    STATUS_BASELINE,
    STATUS_DISCARD,
    STATUS_KEEP,
    categorize_kill_reason,
    decide_status,
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


# ── extra_classifier hook ──────────────────────────────────────────────


def test_extra_classifier_wins_when_returns_tuple() -> None:
    def extra(kr: str) -> tuple[str, dict[str, str]] | None:
        if "tokenizer race" in kr:
            return "tokenizer_race", {}
        return None

    cat, extras = categorize_kill_reason(
        "tokenizer race detected at step 12", extra_classifier=extra
    )
    assert cat == "tokenizer_race"
    assert extras == {}


def test_extra_classifier_extras_passthrough() -> None:
    import re as _re

    def extra(kr: str) -> tuple[str, dict[str, str]] | None:
        m = _re.search(r"retry-after=([\d.]+)", kr)
        if "wandb 429" in kr:
            return "wandb_throttle", ({"retry_after": m.group(1)} if m else {})
        return None

    cat, extras = categorize_kill_reason("wandb 429 retry-after=30.0", extra_classifier=extra)
    assert cat == "wandb_throttle"
    assert extras == {"retry_after": "30.0"}


def test_extra_classifier_returning_none_falls_through_to_builtins() -> None:
    def extra(kr: str) -> tuple[str, dict[str, str]] | None:
        return None  # never matches anything

    cat, extras = categorize_kill_reason(
        "|kl|=0.7 suggests policy divergence", extra_classifier=extra
    )
    assert cat == KILL_POLICY_DIVERGENCE
    assert extras == {"kl": "0.7"}


def test_extra_classifier_can_override_builtin_categories() -> None:
    # Project deliberately reclassifies a string the builtin would catch.
    # This is intentional — extra_classifier runs FIRST.
    def extra(kr: str) -> tuple[str, dict[str, str]] | None:
        if "kl" in kr:
            return "custom_kl_handler", {"raw": kr}
        return None

    cat, extras = categorize_kill_reason(
        "|kl|=0.7 suggests policy divergence", extra_classifier=extra
    )
    assert cat == "custom_kl_handler"
    assert extras == {"raw": "|kl|=0.7 suggests policy divergence"}


def test_extra_classifier_receives_lowercased_input() -> None:
    seen: list[str] = []

    def extra(kr: str) -> tuple[str, dict[str, str]] | None:
        seen.append(kr)
        return None

    categorize_kill_reason("MIXED Case Reason", extra_classifier=extra)
    assert seen == ["mixed case reason"]


def test_extra_classifier_not_called_for_empty_reason() -> None:
    seen: list[str] = []

    def extra(kr: str) -> tuple[str, dict[str, str]] | None:
        seen.append(kr)
        return ("never", {})

    cat, extras = categorize_kill_reason("", extra_classifier=extra)
    assert cat == KILL_UNKNOWN
    assert extras == {}
    assert seen == []  # short-circuit before classifier runs

    cat2, extras2 = categorize_kill_reason(None, extra_classifier=extra)
    assert cat2 == KILL_UNKNOWN
    assert extras2 == {}
    assert seen == []


def test_extra_classifier_default_none_keeps_builtin_behaviour() -> None:
    # No regression for callers that don't pass extra_classifier.
    cat, extras = categorize_kill_reason("|kl|=0.5 suggests policy divergence")
    assert cat == KILL_POLICY_DIVERGENCE
    assert extras == {"kl": "0.5"}


# ── decide_status ──────────────────────────────────────────────────────


def test_decide_status_empty_history_returns_baseline() -> None:
    assert decide_status([], 5.0) == STATUS_BASELINE


def test_decide_status_only_discard_priors_treated_as_empty() -> None:
    """A history of only DISCARD/EARLY_KILL rows means no comparable baseline
    has been admitted yet → BASELINE."""
    prior = [
        {"status": "DISCARD", "score": 99.0},
        {"status": "EARLY_KILL", "score": 99.0},
        {"status": "CRASH", "score": 99.0},
    ]
    assert decide_status(prior, 1.0) == STATUS_BASELINE


def test_decide_status_keep_when_strictly_better() -> None:
    prior = [{"status": STATUS_BASELINE, "score": 5.0}]
    assert decide_status(prior, 5.5) == STATUS_KEEP


def test_decide_status_discard_when_equal_or_worse() -> None:
    """Strict-better only — ties go to DISCARD so the original baseline holds."""
    prior = [{"status": STATUS_BASELINE, "score": 5.0}]
    assert decide_status(prior, 5.0) == STATUS_DISCARD
    assert decide_status(prior, 4.9) == STATUS_DISCARD


def test_decide_status_compares_against_max_keep_baseline() -> None:
    """Mixed KEEP+BASELINE+DISCARD history — only KEEP/BASELINE rows count."""
    prior = [
        {"status": STATUS_BASELINE, "score": 5.0},
        {"status": STATUS_KEEP, "score": 9.0},
        {"status": STATUS_DISCARD, "score": 100.0},  # ignored — DISCARD
    ]
    assert decide_status(prior, 9.5) == STATUS_KEEP
    assert decide_status(prior, 9.0) == STATUS_DISCARD


def test_decide_status_uses_evaluation_score_alias() -> None:
    """Default ``score_fn=get_score`` falls back to ``evaluation_score`` when
    ``score`` is absent — matches the orak row schema."""
    prior = [{"status": STATUS_KEEP, "evaluation_score": 7.0}]
    assert decide_status(prior, 8.0) == STATUS_KEEP


def test_decide_status_custom_score_fn_for_heldout_metrics() -> None:
    """gemma4-rlvr's apples-to-apples heldout comparison via custom extractor."""

    def heldout(row: dict) -> float | None:
        return ((row.get("metrics") or {}).get("heldout") or {}).get("mean_total")

    prior = [
        {"status": STATUS_BASELINE, "score": 1.0, "metrics": {"heldout": {"mean_total": 12.0}}},
        {"status": STATUS_KEEP, "score": 2.0, "metrics": {"heldout": {"mean_total": 14.0}}},
    ]
    # New row scored on the same heldout scale.
    assert decide_status(prior, 14.5, score_fn=heldout) == STATUS_KEEP
    assert decide_status(prior, 14.0, score_fn=heldout) == STATUS_DISCARD


def test_decide_status_score_fn_returning_none_skips_row() -> None:
    """When prior KEEP/BASELINE rows have no comparable score, treat as empty
    history → BASELINE. Stops stale rows from blocking new metric rollouts."""

    def heldout(row: dict) -> float | None:
        return ((row.get("metrics") or {}).get("heldout") or {}).get("mean_total")

    # Prior row exists and is admitted, but lacks the heldout sub-key.
    prior = [{"status": STATUS_KEEP, "score": 5.0}]
    assert decide_status(prior, 1.0, score_fn=heldout) == STATUS_BASELINE


def test_decide_status_custom_keep_statuses() -> None:
    """Project that tracks an extra admit status (eg. PROMOTED)."""
    prior = [{"status": "PROMOTED", "score": 10.0}]
    # Default keep_statuses ignores PROMOTED → no comparable history.
    assert decide_status(prior, 1.0) == STATUS_BASELINE
    # Override to include PROMOTED.
    assert (
        decide_status(prior, 11.0, keep_statuses=(STATUS_KEEP, STATUS_BASELINE, "PROMOTED"))
        == STATUS_KEEP
    )


def test_decide_status_missing_status_field_treated_as_unkept() -> None:
    """Defensive — old rows without a ``status`` field are excluded from comparison."""
    prior = [{"score": 99.0}]  # no status
    assert decide_status(prior, 1.0) == STATUS_BASELINE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
