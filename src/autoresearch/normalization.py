"""Per-task score normalisation — map heterogeneous raw scores to 0-100.

Lifted (and generalised) from `orak-2025-starter-kit/experiments/experiment_progress.py`'s
hardcoded `normalize_eval_score`. Backends emit task scores on different
scales (Mario gives 0-100 % progress, 2048 gives a 0-1 fraction, Pokemon
gives a flag count 0-7), but cross-task milestone charts and verdicts
need a single comparable axis.

Rather than baking specific task names into the package, this module
exposes a small registry:

* :func:`register_normalizer` — projects register their own task → fn mapping
* :func:`normalize_score` — looks up the registered fn (or returns the raw
  value as-is when no normaliser is registered, which is the right
  no-op behaviour for already-percent metrics)

The default registry ships with the three orak normalisers as a
convenience (``super_mario``, ``twenty_fourty_eight``, ``pokemon_red``)
so the orak migration is a one-line import swap. Override or extend by
calling :func:`register_normalizer` at import time.
"""

from __future__ import annotations

from collections.abc import Callable

ScoreNormalizer = Callable[[float], float]


_REGISTRY: dict[str, ScoreNormalizer] = {}


def register_normalizer(task: str, fn: ScoreNormalizer) -> None:
    """Register (or replace) a per-task score normaliser.

    The function takes the raw score and returns a value in ``[0, 100]``.
    """
    _REGISTRY[task] = fn


def get_normalizer(task: str) -> ScoreNormalizer | None:
    """Return the registered normaliser for ``task`` or ``None``."""
    return _REGISTRY.get(task)


def normalize_score(task: str, raw_score: float) -> float:
    """Normalise ``raw_score`` to 0-100 using the task's registered normaliser.

    If no normaliser is registered the raw value is returned unchanged
    (the right no-op for already-percent metrics).
    """
    fn = _REGISTRY.get(task)
    return fn(raw_score) if fn is not None else raw_score


# ── default registrations (orak compatibility) ─────────────────────────


def _mario(raw: float) -> float:
    """Mario: server already returns 0-100 % progress."""
    return raw


def _twenty_fourty_eight(raw: float) -> float:
    """2048: server returns a 0-1 fraction (game_score / 20000); fall back to
    clamp on values that already look like percentages."""
    if raw < 1.0:
        return raw * 100
    return min(raw, 100.0)


def _pokemon_red(raw: float) -> float:
    """Pokemon Red: server returns a raw flag count (0-7); rescale to 0-100,
    fall back to clamp on values that already look like percentages."""
    if raw <= 7:
        return (raw / 7) * 100
    return min(raw, 100.0)


register_normalizer("super_mario", _mario)
register_normalizer("twenty_fourty_eight", _twenty_fourty_eight)
register_normalizer("pokemon_red", _pokemon_red)


__all__ = [
    "ScoreNormalizer",
    "get_normalizer",
    "normalize_score",
    "register_normalizer",
]
