"""Tests for autoresearch.prompt_caching — extract_cache_stats across backends."""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.prompt_caching import extract_cache_stats

# ── fixtures matching the three real shapes ────────────────────────────


@dataclass
class _ChatPromptDetails:
    cached_tokens: int


@dataclass
class _ChatUsage:
    """vLLM / OpenAI ChatCompletions CompletionUsage."""

    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: _ChatPromptDetails | None = None


@dataclass
class _RespInputDetails:
    cached_tokens: int


@dataclass
class _RespUsage:
    """OpenAI Responses API ResponseUsage."""

    input_tokens: int
    output_tokens: int
    input_tokens_details: _RespInputDetails | None = None


# ── tests ──────────────────────────────────────────────────────────────


def test_none_usage_returns_zeros() -> None:
    assert extract_cache_stats(None) == {
        "cached_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_chatcompletions_with_cached_tokens() -> None:
    usage = _ChatUsage(
        prompt_tokens=120,
        completion_tokens=15,
        prompt_tokens_details=_ChatPromptDetails(cached_tokens=80),
    )
    assert extract_cache_stats(usage) == {
        "cached_tokens": 80,
        "input_tokens": 120,
        "output_tokens": 15,
    }


def test_chatcompletions_no_cache_details() -> None:
    """No prompt_tokens_details → cached_tokens=0 but input/output still populate."""
    usage = _ChatUsage(prompt_tokens=42, completion_tokens=7)
    assert extract_cache_stats(usage) == {
        "cached_tokens": 0,
        "input_tokens": 42,
        "output_tokens": 7,
    }


def test_responses_api_with_cached_tokens() -> None:
    usage = _RespUsage(
        input_tokens=200,
        output_tokens=20,
        input_tokens_details=_RespInputDetails(cached_tokens=150),
    )
    assert extract_cache_stats(usage) == {
        "cached_tokens": 150,
        "input_tokens": 200,
        "output_tokens": 20,
    }


def test_dict_usage_chat_shape() -> None:
    """Plain-dict usage (custom adapters) goes through the same extractor."""
    usage = {
        "prompt_tokens": 50,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 30},
    }
    assert extract_cache_stats(usage) == {
        "cached_tokens": 30,
        "input_tokens": 50,
        "output_tokens": 5,
    }


def test_dict_usage_responses_shape() -> None:
    usage = {
        "input_tokens": 90,
        "output_tokens": 9,
        "input_tokens_details": {"cached_tokens": 70},
    }
    assert extract_cache_stats(usage) == {
        "cached_tokens": 70,
        "input_tokens": 90,
        "output_tokens": 9,
    }


def test_mixed_shape_sums_both_cached_fields() -> None:
    """Defensive: if a backend exposes both fields we sum them rather than
    silently dropping one — the caller can decide if that ever happens."""
    usage = {
        "prompt_tokens_details": {"cached_tokens": 10},
        "input_tokens_details": {"cached_tokens": 5},
        "prompt_tokens": 100,
    }
    assert extract_cache_stats(usage)["cached_tokens"] == 15


def test_none_cached_tokens_treated_as_zero() -> None:
    """Backends that emit ``cached_tokens=None`` (instead of omitting) shouldn't
    blow up — we coerce to 0 before adding."""
    usage = {"prompt_tokens_details": {"cached_tokens": None}, "prompt_tokens": 1}
    out = extract_cache_stats(usage)
    assert out["cached_tokens"] == 0
    assert out["input_tokens"] == 1


def test_missing_input_output_default_to_zero() -> None:
    """Half-shaped usage objects (eg. mocked tests) shouldn't raise."""
    assert extract_cache_stats({}) == {
        "cached_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
