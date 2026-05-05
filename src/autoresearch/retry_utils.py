"""Retry helpers — jittered backoff + lightweight error classification.

Lifted from `orak-2025-starter-kit/agents/_harness/retry_utils.py`. Useful
for any LLM-call agent that needs to survive transient backend flakiness
(rate limits, 5xx, connection resets) without silently turning failures
into fallback actions that pollute the trajectory.

Why decorrelated jitter: under sweep-runner parallelism multiple workers
hit the same vLLM server and would otherwise retry in lock-step on a 5xx,
producing thundering-herd reload spikes.

Why a classifier on top of bare retry: ``try / except → return "fallback"``
silently turns transient errors into agent decisions; we want
**transient → retry, terminal → raise** so the caller's logs reflect
reality and the trajectory writer can mark the step as a fallback only
when it really is one.
"""

from __future__ import annotations

import enum
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Decorrelated exponential backoff. ``attempt`` is 1-based.

    Returns ``min(base_delay * 2^(attempt-1), max_delay)`` plus a random
    jitter in ``[0, jitter_ratio * delay)``. The jitter seed is mixed
    from a process-wide counter and the wall clock so concurrent callers
    don't collide on identical sleeps.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2**exponent), max_delay)

    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)
    return delay + jitter


class ErrorClass(enum.Enum):
    """How a failed call should be treated."""

    TRANSIENT = "transient"  # retry
    TERMINAL = "terminal"  # raise immediately
    UNKNOWN = "unknown"  # retry but log loudly


@dataclass
class ClassifiedError:
    """The classifier's decision plus enough context to act on it."""

    cls: ErrorClass
    status: int | None
    message: str
    original: BaseException


_TRANSIENT_MARKERS = (
    "rate limit",
    "rate_limit",
    "timed out",
    "timeout",
    "connection",
    "temporarily unavailable",
    "overloaded",
    "503",
    "502",
    "504",
    "broken pipe",
    "reset by peer",
)

_TERMINAL_MARKERS = (
    "invalid api key",
    "unauthorized",
    "permission denied",
    "model not found",
    "context length",
)


def classify(error: BaseException) -> ClassifiedError:
    """Classify by HTTP status (if exposed) then exception message.

    Liberal on the transient side — we'd rather retry once and waste a
    second than silently swallow a recoverable error. ``UNKNOWN`` errors
    are retried but logged loudly so they don't sneak past review.
    """
    status = _extract_status(error)
    msg = str(error).lower()

    if status is not None:
        if status == 429 or 500 <= status < 600:
            return ClassifiedError(ErrorClass.TRANSIENT, status, str(error), error)
        if status in (401, 403):
            return ClassifiedError(ErrorClass.TERMINAL, status, str(error), error)
        if 400 <= status < 500:
            # 400/404/etc — usually our fault (bad schema / URL) → terminal
            return ClassifiedError(ErrorClass.TERMINAL, status, str(error), error)

    if any(m in msg for m in _TRANSIENT_MARKERS):
        return ClassifiedError(ErrorClass.TRANSIENT, status, str(error), error)
    if any(m in msg for m in _TERMINAL_MARKERS):
        return ClassifiedError(ErrorClass.TERMINAL, status, str(error), error)

    return ClassifiedError(ErrorClass.UNKNOWN, status, str(error), error)


def _extract_status(error: BaseException) -> int | None:
    """Pull an HTTP status off whichever attribute the SDK happens to use."""
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(error, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(error, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
    return None


def with_retries(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    label: str = "llm_call",
) -> Any:
    """Run ``fn`` with classified retries + jittered backoff.

    Raises the original exception when all attempts exhaust or the error
    is terminal. The raised exception carries a ``__classified__``
    attribute holding the :class:`ClassifiedError` so callers can inspect
    the final classification without changing exception types.
    """
    last: ClassifiedError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except BaseException as e:
            ce = classify(e)
            last = ce
            if ce.cls is ErrorClass.TERMINAL:
                logger.error("[%s] terminal error: %s %s", label, ce.status, ce.message[:200])
                e.__classified__ = ce  # type: ignore[attr-defined]
                raise
            if attempt >= max_attempts:
                logger.error(
                    "[%s] giving up after %d attempts: %s",
                    label,
                    attempt,
                    ce.message[:200],
                )
                e.__classified__ = ce  # type: ignore[attr-defined]
                raise
            delay = jittered_backoff(attempt, base_delay=base_delay, max_delay=max_delay)
            logger.warning(
                "[%s] %s (status=%s) — retry %d/%d in %.1fs: %s",
                label,
                ce.cls.value,
                ce.status,
                attempt,
                max_attempts - 1,
                delay,
                ce.message[:120],
            )
            time.sleep(delay)
    # Unreachable — every path above either returns or raises
    assert last is not None
    raise last.original


__all__ = [
    "ClassifiedError",
    "ErrorClass",
    "classify",
    "jittered_backoff",
    "with_retries",
]
