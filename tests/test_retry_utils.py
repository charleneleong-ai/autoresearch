"""Tests for autoresearch.retry_utils — jittered backoff + classify + with_retries."""

from __future__ import annotations

import pytest

from autoresearch.retry_utils import (
    ClassifiedError,
    ErrorClass,
    classify,
    jittered_backoff,
    with_retries,
)

# ── jittered_backoff ───────────────────────────────────────────────────


def test_backoff_grows_exponentially_until_max_delay() -> None:
    base, max_d = 1.0, 30.0
    # attempt=1 → delay=1, jitter ratio 0.5 → range [1, 1.5]
    # attempt=2 → delay=2, jitter ratio 0.5 → range [2, 3]
    # attempt=3 → delay=4 → range [4, 6]
    # attempt=10 → 2^9=512 capped to 30 → range [30, 45]
    assert 1.0 <= jittered_backoff(1, base_delay=base, max_delay=max_d) <= 1.5 + 1e-9
    assert 2.0 <= jittered_backoff(2, base_delay=base, max_delay=max_d) <= 3.0 + 1e-9
    assert 4.0 <= jittered_backoff(3, base_delay=base, max_delay=max_d) <= 6.0 + 1e-9
    assert 30.0 <= jittered_backoff(10, base_delay=base, max_delay=max_d) <= 45.0 + 1e-9


def test_backoff_attempt_zero_treated_as_one() -> None:
    """``attempt=0`` shouldn't go negative — clamps to attempt 1."""
    assert 1.0 <= jittered_backoff(0, base_delay=1.0, max_delay=10.0) <= 1.5 + 1e-9


def test_backoff_huge_attempt_doesnt_overflow() -> None:
    """attempt=63+ would overflow 2**N — should saturate at max_delay."""
    d = jittered_backoff(100, base_delay=1.0, max_delay=10.0)
    assert 10.0 <= d <= 15.0 + 1e-9


def test_backoff_zero_base_returns_max_delay_only() -> None:
    """base_delay=0 short-circuits to max_delay (avoids 0 * 2^N=0)."""
    d = jittered_backoff(1, base_delay=0.0, max_delay=10.0)
    assert 10.0 <= d <= 15.0 + 1e-9


def test_backoff_jitter_decorrelates_concurrent_calls() -> None:
    """Two calls in the same nanosecond should still produce different delays
    because the process-wide counter is mixed into the seed."""
    delays = {jittered_backoff(2, base_delay=10.0, max_delay=100.0) for _ in range(50)}
    # Even with 50 same-attempt calls we expect lots of distinct values.
    assert len(delays) > 30


# ── classify ───────────────────────────────────────────────────────────


class _StatusErr(Exception):
    """Fake SDK error with a status_code attribute."""

    def __init__(self, msg: str, status: int) -> None:
        super().__init__(msg)
        self.status_code = status


class _ResponseErr(Exception):
    """Fake SDK error that stashes status on a `response` attribute."""

    def __init__(self, msg: str, status: int) -> None:
        super().__init__(msg)

        class _R:
            status_code = status

        self.response = _R()


def test_classify_429_is_transient() -> None:
    out = classify(_StatusErr("rate limit exceeded", 429))
    assert out.cls is ErrorClass.TRANSIENT
    assert out.status == 429


def test_classify_5xx_is_transient() -> None:
    for s in (500, 502, 503, 504, 599):
        out = classify(_StatusErr("upstream", s))
        assert out.cls is ErrorClass.TRANSIENT, f"status {s} should be transient"


def test_classify_4xx_auth_is_terminal() -> None:
    for s in (401, 403):
        out = classify(_StatusErr("nope", s))
        assert out.cls is ErrorClass.TERMINAL, f"status {s} should be terminal"


def test_classify_other_4xx_is_terminal() -> None:
    """400 / 404 / 422 etc are usually our fault — terminal so we don't loop on bad input."""
    for s in (400, 404, 409, 422):
        out = classify(_StatusErr("bad", s))
        assert out.cls is ErrorClass.TERMINAL, f"status {s} should be terminal"


def test_classify_status_via_response_attr() -> None:
    """Some SDKs put status on `error.response.status_code`, not on the error itself."""
    out = classify(_ResponseErr("server down", 503))
    assert out.cls is ErrorClass.TRANSIENT
    assert out.status == 503


def test_classify_message_transient_marker_no_status() -> None:
    """Falls back to message scan when no status is exposed."""
    out = classify(Exception("Read timed out after 30s"))
    assert out.cls is ErrorClass.TRANSIENT


def test_classify_message_terminal_marker_no_status() -> None:
    out = classify(Exception("Invalid API key — please regenerate"))
    assert out.cls is ErrorClass.TERMINAL


def test_classify_unknown_when_no_signal() -> None:
    out = classify(RuntimeError("something weird happened"))
    assert out.cls is ErrorClass.UNKNOWN


def test_classify_carries_original_exception() -> None:
    err = ValueError("boom")
    out = classify(err)
    assert out.original is err


# ── with_retries ───────────────────────────────────────────────────────


def test_with_retries_returns_value_on_first_success() -> None:
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    assert with_retries(fn, max_attempts=3, base_delay=0, max_delay=0) == "ok"
    assert len(calls) == 1


def test_with_retries_recovers_after_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """First two calls 503 then succeed — caller sees success, 3 attempts made."""
    monkeypatch.setattr("autoresearch.retry_utils.time.sleep", lambda _s: None)
    monkeypatch.setattr("autoresearch.retry_utils.jittered_backoff", lambda *a, **kw: 0.0)

    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _StatusErr("upstream", 503)
        return "recovered"

    assert with_retries(fn, max_attempts=5, base_delay=0, max_delay=0) == "recovered"
    assert attempts["n"] == 3


def test_with_retries_terminal_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """No retries on terminal — raises after the first attempt."""
    monkeypatch.setattr("autoresearch.retry_utils.time.sleep", lambda _s: None)

    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        raise _StatusErr("bad creds", 401)

    with pytest.raises(_StatusErr) as exc:
        with_retries(fn, max_attempts=5, base_delay=0, max_delay=0)
    assert attempts["n"] == 1
    classified: ClassifiedError = exc.value.__classified__  # type: ignore[attr-defined]
    assert classified.cls is ErrorClass.TERMINAL
    assert classified.status == 401


def test_with_retries_exhausts_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """All attempts fail transiently — raises original with classification attached."""
    monkeypatch.setattr("autoresearch.retry_utils.time.sleep", lambda _s: None)
    monkeypatch.setattr("autoresearch.retry_utils.jittered_backoff", lambda *a, **kw: 0.0)

    attempts = {"n": 0}

    def fn() -> str:
        attempts["n"] += 1
        raise _StatusErr("upstream", 503)

    with pytest.raises(_StatusErr) as exc:
        with_retries(fn, max_attempts=3, base_delay=0, max_delay=0)
    assert attempts["n"] == 3
    classified: ClassifiedError = exc.value.__classified__  # type: ignore[attr-defined]
    assert classified.cls is ErrorClass.TRANSIENT
