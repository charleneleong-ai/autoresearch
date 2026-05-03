"""URL-parsing tests for autoresearch.wandb_history.

The fetch_history() network path is exercised indirectly via the gradient_collapse
detector tests (which monkeypatch fetch_history). These tests cover the pure-
parsing surface so URL handling stays correct without needing wandb installed.
"""

from __future__ import annotations

import pytest

from autoresearch.wandb_history import WandbRunRef, parse_run_url


def test_parse_full_https_url() -> None:
    ref = parse_run_url("https://wandb.ai/charlene/orak/runs/abc123")
    assert ref == WandbRunRef("charlene", "orak", "abc123")
    assert ref.path == "charlene/orak/abc123"


def test_parse_full_http_url() -> None:
    ref = parse_run_url("http://wandb.ai/charlene/orak/runs/abc123")
    assert ref.entity == "charlene"


def test_parse_url_with_www_prefix() -> None:
    ref = parse_run_url("https://www.wandb.ai/charlene/orak/runs/abc123")
    assert ref.entity == "charlene"


def test_parse_url_strips_trailing_slash_and_querystring() -> None:
    ref = parse_run_url("https://wandb.ai/charlene/orak/runs/abc123/")
    assert ref.run_id == "abc123"
    ref2 = parse_run_url("https://wandb.ai/charlene/orak/runs/abc123?workspace=user-charlene")
    assert ref2.run_id == "abc123"


def test_parse_short_form() -> None:
    ref = parse_run_url("charlene/orak/abc123")
    assert ref.path == "charlene/orak/abc123"


def test_parse_run_id_with_underscores() -> None:
    # Real wandb run IDs are often timestamps + hashes
    ref = parse_run_url("charlene/orak/20260503_071254_orak-super-mario")
    assert ref.run_id == "20260503_071254_orak-super-mario"


def test_parse_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognised wandb run reference"):
        parse_run_url("not a url")
    with pytest.raises(ValueError):
        parse_run_url("https://wandb.ai/just-entity")  # missing project + run
    with pytest.raises(ValueError):
        parse_run_url("a/b")  # only 2 segments — not enough


def test_fetch_history_raises_import_error_when_wandb_missing() -> None:
    """If `import wandb` fails, fetch_history should re-raise ImportError with
    a helpful install hint. We rely on wandb genuinely being uninstalled in
    the test environment; if a future dev installs `[wandb]` extra, this test
    will be skipped automatically."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        pass  # expected — proceed to call fetch_history
    else:
        pytest.skip("wandb is installed; ImportError path can't be exercised here")

    from autoresearch.wandb_history import fetch_history

    with pytest.raises(ImportError, match=r"\[wandb\] extra"):
        fetch_history(run_url="charlene/orak/abc123", keys=["train/loss"])
