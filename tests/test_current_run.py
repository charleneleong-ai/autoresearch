"""Tests for autoresearch.current_run — log-format parsing + sidecar lifecycle.

Exercises both built-in `LOG_FORMATS` presets ("default" and "untimed")
against synthetic log fixtures so the position-based "iter is done" check
and the per-format description extraction are covered end-to-end without
spinning up a real sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.current_run import (
    LOG_FORMATS,
    LogFormat,
    _resolve_description,
    _tick,
)

# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """Self-contained logs/sidecar/results layout for a single test."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    sidecar = tmp_path / "current_run.json"
    results = tmp_path / "results.jsonl"
    return {"logs_dir": logs_dir, "sidecar": sidecar, "results": results}


def _write_log(logs_dir: Path, name: str, body: str) -> Path:
    """Drop a synthetic log file matching the daemon's `autoresearch_*T*Z.log` glob."""
    path = logs_dir / f"autoresearch_{name}T120000Z.log"
    path.write_text(body)
    return path


# ── default format (gemma4-style) ──────────────────────────────────────


def test_default_format_iter_in_flight_writes_sidecar(workspace: dict[str, Path]) -> None:
    body = (
        "[2026-05-08T12:00:00Z] Iter 1/3: warmup\n"
        "$ python train.py -d [autoresearch 1/3] baseline run --max-steps 100\n"
    )
    _write_log(workspace["logs_dir"], "20260508", body)

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name="cfg",
        fmt=LOG_FORMATS["default"],
    )

    assert workspace["sidecar"].exists()
    payload = json.loads(workspace["sidecar"].read_text())
    assert payload["iter_marker"] == "Iter 1/3"
    assert payload["description"] == "[autoresearch 1/3] baseline run"
    assert payload["started_at"] == "2026-05-08T12:00:00Z"
    assert payload["config_name"] == "cfg"


def test_default_format_per_iter_finished_drops_sidecar(workspace: dict[str, Path]) -> None:
    workspace["sidecar"].write_text(json.dumps({"iter_marker": "Iter 1/3"}))
    body = "[2026-05-08T12:00:00Z] Iter 1/3: warmup\n[2026-05-08T12:05:00Z] Iter 1/3 finished\n"
    _write_log(workspace["logs_dir"], "20260508", body)

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name=None,
        fmt=LOG_FORMATS["default"],
    )

    assert not workspace["sidecar"].exists()


def test_default_format_only_drops_sidecar_for_latest_iter(
    workspace: dict[str, Path],
) -> None:
    """If iter 1 finished but iter 2 is in flight, sidecar must reflect iter 2."""
    body = (
        "[2026-05-08T12:00:00Z] Iter 1/3: warmup\n"
        "[2026-05-08T12:05:00Z] Iter 1/3 finished\n"
        "[2026-05-08T12:05:30Z] Iter 2/3: main training\n"
    )
    _write_log(workspace["logs_dir"], "20260508", body)

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name=None,
        fmt=LOG_FORMATS["default"],
    )

    assert workspace["sidecar"].exists()
    payload = json.loads(workspace["sidecar"].read_text())
    assert payload["iter_marker"] == "Iter 2/3"


# ── untimed format (orak-style) ────────────────────────────────────────


def test_untimed_format_iter_in_flight_writes_sidecar(workspace: dict[str, Path]) -> None:
    body = (
        "# Iteration 3/30\n"
        "Run ID: 20260508_120000\n"
        "Description: 2048 — eager strategy\n"
        "https://wandb.ai/orak/runs/abc123\n"
    )
    _write_log(workspace["logs_dir"], "20260508", body)

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name=None,
        fmt=LOG_FORMATS["untimed"],
    )

    assert workspace["sidecar"].exists()
    payload = json.loads(workspace["sidecar"].read_text())
    assert payload["iter_marker"] == "Iter 3/30"
    assert payload["description"] == "2048 — eager strategy"
    assert payload["wandb_url"] == "https://wandb.ai/orak/runs/abc123"
    # Untimed format has no `ts` capture — `started_at` must be omitted.
    assert "started_at" not in payload


def test_untimed_format_sweep_complete_drops_sidecar(workspace: dict[str, Path]) -> None:
    workspace["sidecar"].write_text(json.dumps({"iter_marker": "Iter 30/30"}))
    body = "# Iteration 30/30\nDescription: final iter\nAutoresearch complete after 30 iterations\n"
    _write_log(workspace["logs_dir"], "20260508", body)

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name=None,
        fmt=LOG_FORMATS["untimed"],
    )

    assert not workspace["sidecar"].exists()


def test_untimed_format_completion_before_latest_iter_doesnt_drop(
    workspace: dict[str, Path],
) -> None:
    """A completion marker BEFORE the latest iter shouldn't drop the new iter's sidecar.

    (Edge case: stale completion lines from a previous sweep run leaving residue
    in the same log file are still positionally before any new in-flight iter.)
    """
    body = (
        "Autoresearch complete after 5 iterations\n"  # stale marker
        "# Iteration 1/3\n"  # NEW sweep starting
        "Description: new sweep iter 1\n"
    )
    _write_log(workspace["logs_dir"], "20260508", body)

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name=None,
        fmt=LOG_FORMATS["untimed"],
    )

    assert workspace["sidecar"].exists()
    payload = json.loads(workspace["sidecar"].read_text())
    assert payload["iter_marker"] == "Iter 1/3"


# ── description fallback chain ─────────────────────────────────────────


def test_resolve_description_falls_back_to_rest_when_no_desc_re() -> None:
    """When fmt has no `desc_re` and the iter line has a `rest` group, use rest."""
    fmt = LogFormat(
        iter_start_re=LOG_FORMATS["default"].iter_start_re,
        iter_done_re=LOG_FORMATS["default"].iter_done_re,
        desc_re=None,
    )
    text = "[2026-05-08T12:00:00Z] Iter 1/3: warmup phase\n"
    last = fmt.iter_start_re.search(text)
    assert last is not None
    assert _resolve_description(text, last, fmt, 1, 3) == "warmup phase"


def test_resolve_description_falls_back_to_generic_when_no_rest_no_desc() -> None:
    """When fmt has no `desc_re` AND no `rest` capture, fall back to `iter N/M`."""
    fmt = LogFormat(
        iter_start_re=LOG_FORMATS["untimed"].iter_start_re,  # no `rest` group
        iter_done_re=LOG_FORMATS["untimed"].iter_done_re,
        desc_re=None,
    )
    text = "# Iteration 7/10\n"
    last = fmt.iter_start_re.search(text)
    assert last is not None
    assert _resolve_description(text, last, fmt, 7, 10) == "iter 7/10"


# ── no-change short-circuit ────────────────────────────────────────────


def test_tick_no_change_doesnt_rewrite_sidecar(workspace: dict[str, Path]) -> None:
    body = (
        "[2026-05-08T12:00:00Z] Iter 1/3: warmup\n"
        "$ python train.py -d [autoresearch 1/3] baseline run --max-steps 100\n"
    )
    _write_log(workspace["logs_dir"], "20260508", body)

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name="cfg",
        fmt=LOG_FORMATS["default"],
    )
    first_mtime = workspace["sidecar"].stat().st_mtime_ns

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name="cfg",
        fmt=LOG_FORMATS["default"],
    )

    assert workspace["sidecar"].stat().st_mtime_ns == first_mtime


# ── empty-log no-op ────────────────────────────────────────────────────


def test_tick_no_log_is_noop(workspace: dict[str, Path]) -> None:
    """No log files yet — sidecar absent should stay absent, no exception."""
    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name=None,
        fmt=LOG_FORMATS["default"],
    )
    assert not workspace["sidecar"].exists()


def test_tick_log_without_iter_lines_is_noop(workspace: dict[str, Path]) -> None:
    """Log exists but has no iter lines — sidecar should not be created."""
    _write_log(workspace["logs_dir"], "20260508", "boot, but no iters yet\n")

    _tick(
        workspace["logs_dir"],
        workspace["sidecar"],
        workspace["results"],
        config_name=None,
        fmt=LOG_FORMATS["default"],
    )
    assert not workspace["sidecar"].exists()
