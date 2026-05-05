"""Tests for autoresearch.normalization — per-task score → 0-100 registry."""

from __future__ import annotations

import pytest

from autoresearch.normalization import (
    _REGISTRY,
    get_normalizer,
    normalize_score,
    register_normalizer,
)


@pytest.fixture(autouse=True)
def restore_registry():
    """Each test snapshots the registry and restores it on exit so per-test
    registrations don't leak across tests (or shadow the orak defaults)."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


# ── default registrations (orak compatibility) ─────────────────────────


def test_super_mario_passes_through() -> None:
    assert normalize_score("super_mario", 67.5) == 67.5
    assert normalize_score("super_mario", 0.0) == 0.0
    assert normalize_score("super_mario", 100.0) == 100.0


def test_twenty_fourty_eight_fraction_scaled_to_percent() -> None:
    assert normalize_score("twenty_fourty_eight", 0.0) == 0.0
    assert normalize_score("twenty_fourty_eight", 0.42) == pytest.approx(42.0)
    assert normalize_score("twenty_fourty_eight", 0.999) == pytest.approx(99.9)


def test_twenty_fourty_eight_already_percent_clamped() -> None:
    """Edge case: server hands us a number ≥1 — treat as already-percent and clamp."""
    assert normalize_score("twenty_fourty_eight", 1.0) == 1.0
    assert normalize_score("twenty_fourty_eight", 50.0) == 50.0
    assert normalize_score("twenty_fourty_eight", 150.0) == 100.0


def test_pokemon_red_flag_count_to_percent() -> None:
    assert normalize_score("pokemon_red", 0) == 0
    assert normalize_score("pokemon_red", 7) == pytest.approx(100.0)
    assert normalize_score("pokemon_red", 3.5) == pytest.approx(50.0)


def test_pokemon_red_already_percent_clamped() -> None:
    assert normalize_score("pokemon_red", 50.0) == 50.0
    assert normalize_score("pokemon_red", 200.0) == 100.0


# ── registry mechanics ─────────────────────────────────────────────────


def test_unknown_task_passes_through_unchanged() -> None:
    """No registered normaliser → the raw value is the right answer (it was
    already the project's chosen scale)."""
    assert normalize_score("totally_new_task", 12.34) == 12.34
    assert normalize_score("totally_new_task", -5.0) == -5.0


def test_register_normalizer_adds_new_task() -> None:
    register_normalizer("my_task", lambda s: s * 10)
    assert normalize_score("my_task", 7.0) == 70.0
    assert get_normalizer("my_task") is not None


def test_register_normalizer_replaces_existing() -> None:
    """Re-registering replaces — useful when projects want to override the
    default for a task name they happen to share."""
    register_normalizer("super_mario", lambda s: 0.0)
    assert normalize_score("super_mario", 50.0) == 0.0


def test_get_normalizer_returns_none_for_unknown() -> None:
    assert get_normalizer("unregistered_task") is None
