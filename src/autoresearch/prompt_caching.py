"""Prompt-caching helpers — measure prefix cache hits across LLM backends.

Lifted from `orak-2025-starter-kit/agents/_harness/prompt_caching.py`.
Three usage shapes appear in practice and they all need the same view:

* **vLLM / OpenAI ChatCompletions** ``CompletionUsage`` —
  ``prompt_tokens_details.cached_tokens`` + ``prompt_tokens`` +
  ``completion_tokens``.
* **OpenAI Responses API** ``ResponseUsage`` —
  ``input_tokens_details.cached_tokens`` + ``input_tokens`` +
  ``output_tokens``.
* **Plain dict** — some custom adapters wrap usage objects as ``dict``.

:func:`extract_cache_stats` returns a uniform shape regardless of which
backend produced the input.
"""

from __future__ import annotations

from typing import Any


def extract_cache_stats(usage: Any) -> dict[str, int]:
    """Return ``{"cached_tokens", "input_tokens", "output_tokens"}``.

    Missing fields default to 0. Works on the three shapes documented in
    the module docstring (vLLM/ChatCompletions ``CompletionUsage``, OpenAI
    Responses ``ResponseUsage``, and plain dicts).
    """
    if usage is None:
        return {"cached_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    cached = 0

    prompt_details = _get(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached += _get(prompt_details, "cached_tokens", 0) or 0

    inp_details = _get(usage, "input_tokens_details", None)
    if inp_details is not None:
        cached += _get(inp_details, "cached_tokens", 0) or 0

    return {
        "cached_tokens": cached,
        "input_tokens": _get(usage, "prompt_tokens", 0) or _get(usage, "input_tokens", 0) or 0,
        "output_tokens": _get(usage, "completion_tokens", 0)
        or _get(usage, "output_tokens", 0)
        or 0,
    }


def _get(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


__all__ = ["extract_cache_stats"]
