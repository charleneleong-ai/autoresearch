"""Tests for triage being optional in SweepRunner + the built-in
NullTriageMonitor / GPUTriageMonitor / CompositeTriageMonitor classes.

The original SweepRunner required a ``TriageMonitor`` parameter, which
forced every caller — including those who just wanted a single rollout
launched and tagged — to implement the three-method protocol with no-op
bodies. These tests cover making it optional + the three built-ins that
cover common needs (no triage / only GPU triage / composing detectors).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autoresearch.gpu_monitor import GPUSample, GPUTriageThresholds
from autoresearch.results import load_results
from autoresearch.sweep_runner import (
    CompositeTriageMonitor,
    GPUTriageMonitor,
    IterPlan,
    NullTriageMonitor,
    SweepRunner,
)

# ── helpers ────────────────────────────────────────────────────────────


class _StaticPlanner:
    def __init__(self, plans: list[IterPlan]) -> None:
        self._plans = plans

    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        yield from self._plans


class _StaticExtractor:
    def extract(
        self,
        plan: IterPlan,
        run_id: str | None,
        exit_code: int,
    ) -> list[dict[str, Any]]:
        return [{"score": 1.0, "steps": 100}]


def _mock_popen() -> MagicMock:
    proc = MagicMock()
    proc.returncode = 0
    return proc


# ── NullTriageMonitor ──────────────────────────────────────────────────


def test_null_triage_setup_returns_none() -> None:
    triage = NullTriageMonitor()
    proc = MagicMock()
    assert triage.setup(IterPlan(cmd=["true"], description="x"), proc, 0.0) is None


def test_null_triage_check_returns_none() -> None:
    triage = NullTriageMonitor()
    assert triage.check(0.0) is None
    assert triage.check(1000.0) is None


def test_null_triage_teardown_is_noop() -> None:
    NullTriageMonitor().teardown()  # no-op


# ── SweepRunner — triage defaults to NullTriageMonitor ─────────────────


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_sweep_runner_runs_without_triage_arg(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """SweepRunner can be constructed without passing a triage argument —
    falls back to a no-op NullTriageMonitor."""
    mock_popen.return_value = _mock_popen()

    runner = SweepRunner(
        tag="no_triage",
        planner=_StaticPlanner([IterPlan(cmd=["true"], description="one")]),
        extractor=_StaticExtractor(),
        experiments_dir=tmp_path,
        pause_between_iters_s=0,
    )
    result = runner.run()

    assert result.iterations == 1
    assert result.kills == 0
    rows = load_results(tmp_path, "no_triage")
    assert len(rows) == 1
    assert rows[0]["score"] == 1.0


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_sweep_runner_explicit_null_triage_equivalent(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """Passing NullTriageMonitor() explicitly behaves identically to omitting it."""
    mock_popen.return_value = _mock_popen()

    runner = SweepRunner(
        tag="explicit_null",
        planner=_StaticPlanner([IterPlan(cmd=["true"], description="one")]),
        triage=NullTriageMonitor(),
        extractor=_StaticExtractor(),
        experiments_dir=tmp_path,
        pause_between_iters_s=0,
    )
    result = runner.run()
    assert result.iterations == 1


# ── GPUTriageMonitor ───────────────────────────────────────────────────


def test_gpu_triage_monitor_setup_returns_none() -> None:
    """GPUTriageMonitor doesn't discover run_ids (that's project-specific)."""
    triage = GPUTriageMonitor()
    assert triage.setup(IterPlan(cmd=["true"], description="x"), MagicMock(), 0.0) is None


def test_gpu_triage_monitor_check_returns_none_with_no_samples() -> None:
    """When nvidia-smi is unavailable / returns no samples, no kill fires."""
    triage = GPUTriageMonitor()
    with patch("autoresearch.sweep_runner._nvidia_smi_sample", return_value=None):
        assert triage.check(0.0) is None


def test_gpu_triage_monitor_returns_kill_reason_when_thresholds_hit() -> None:
    """GPUTriageMonitor fires a kill reason when GPUTriage's thresholds latch.

    Two samples needed: first sample sets ``_hang_since``, second confirms
    the window has elapsed and latches the kill.
    """
    # Aggressive thresholds for fast deterministic test — no grace, zero window
    # so the second sample immediately latches.
    thresholds = GPUTriageThresholds(
        grace_s=0,
        hang_util_pct=10,
        hang_window_s=0,
    )
    # poll_interval_s=0 so check() doesn't throttle in tests
    triage = GPUTriageMonitor(thresholds=thresholds, poll_interval_s=0)

    sample = GPUSample(util_pct=2, mem_used_gb=50.0, mem_total_gb=80.0)
    with patch("autoresearch.sweep_runner._nvidia_smi_sample", return_value=sample):
        # First sample — starts tracking, no kill yet
        assert triage.check(0.0) is None
        # Second sample — window elapsed, kill latches
        reason = triage.check(0.1)
    assert reason is not None
    assert "hang" in reason.lower() or "util" in reason.lower()


def test_gpu_triage_monitor_resets_between_iters() -> None:
    """setup() resets the underlying GPUTriage so previous iter's latched
    state doesn't carry over into a fresh iter."""
    thresholds = GPUTriageThresholds(grace_s=0, hang_util_pct=10, hang_window_s=0)
    triage = GPUTriageMonitor(thresholds=thresholds, poll_interval_s=0)

    bad = GPUSample(util_pct=2, mem_used_gb=50.0, mem_total_gb=80.0)
    good = GPUSample(util_pct=80, mem_used_gb=60.0, mem_total_gb=80.0)

    # Iter 1: two samples → kill latches
    with patch("autoresearch.sweep_runner._nvidia_smi_sample", return_value=bad):
        triage.check(0.0)
        assert triage.check(0.1) is not None

    # Reset for iter 2
    triage.setup(IterPlan(cmd=["true"], description="x"), MagicMock(), 0.0)
    # Iter 2: healthy sample shouldn't return the stale latched reason
    with patch("autoresearch.sweep_runner._nvidia_smi_sample", return_value=good):
        assert triage.check(0.0) is None
        assert triage.check(0.1) is None


# ── CompositeTriageMonitor ─────────────────────────────────────────────


def test_composite_triage_with_no_children_is_noop() -> None:
    triage = CompositeTriageMonitor([])
    assert triage.setup(IterPlan(cmd=["true"], description="x"), MagicMock(), 0.0) is None
    assert triage.check(0.0) is None
    triage.teardown()  # no children, no error


def test_composite_triage_polls_all_children() -> None:
    """check() polls every child until one returns non-None."""
    child_a = MagicMock(spec=NullTriageMonitor())
    child_a.check.return_value = None
    child_b = MagicMock(spec=NullTriageMonitor())
    child_b.check.return_value = "child B fired"
    child_c = MagicMock(spec=NullTriageMonitor())
    child_c.check.return_value = "child C fired (should not be reached)"

    triage = CompositeTriageMonitor([child_a, child_b, child_c])
    reason = triage.check(5.0)
    assert reason == "child B fired"
    child_a.check.assert_called_once_with(5.0)
    child_b.check.assert_called_once_with(5.0)
    # Short-circuit — child_c never called
    child_c.check.assert_not_called()


def test_composite_triage_setup_returns_first_non_none_run_id() -> None:
    """When multiple children might discover run_ids, the first non-None wins."""
    child_a = MagicMock()
    child_a.setup.return_value = None
    child_b = MagicMock()
    child_b.setup.return_value = "my-run-123"
    child_c = MagicMock()
    child_c.setup.return_value = "should-not-be-used"

    triage = CompositeTriageMonitor([child_a, child_b, child_c])
    run_id = triage.setup(IterPlan(cmd=["true"], description="x"), MagicMock(), 0.0)
    assert run_id == "my-run-123"
    # All three setups are invoked (so each child can latch its signal channels)
    child_a.setup.assert_called_once()
    child_b.setup.assert_called_once()
    child_c.setup.assert_called_once()


def test_composite_triage_teardown_runs_all_children() -> None:
    """teardown() calls every child's teardown, even if one raises."""
    child_a = MagicMock()
    child_b = MagicMock()
    child_b.teardown.side_effect = RuntimeError("teardown failure in child B")
    child_c = MagicMock()

    triage = CompositeTriageMonitor([child_a, child_b, child_c])
    with pytest.raises(RuntimeError, match="child B"):
        triage.teardown()

    # All children still teardown-attempted
    child_a.teardown.assert_called_once()
    child_b.teardown.assert_called_once()
    child_c.teardown.assert_called_once()


# ── run_one convenience ────────────────────────────────────────────────


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_run_one_returns_single_iter_outcome(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """SweepRunner.run_one launches one plan and returns the single IterOutcome."""
    mock_popen.return_value = _mock_popen()

    plan = IterPlan(cmd=["true"], description="single rollout")
    outcome = SweepRunner.run_one(
        plan=plan,
        tag="single",
        extractor=_StaticExtractor(),
        experiments_dir=tmp_path,
    )

    assert outcome.plan is plan
    assert outcome.exit_code == 0
    assert outcome.kill_reason is None
    assert len(outcome.rows) == 1
    assert outcome.rows[0]["score"] == 1.0

    rows = load_results(tmp_path, "single")
    assert len(rows) == 1


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_run_one_accepts_triage_arg(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """run_one accepts the same triage / retrospective_spec / iter_timeout
    knobs as the full SweepRunner constructor."""
    mock_popen.return_value = _mock_popen()

    outcome = SweepRunner.run_one(
        plan=IterPlan(cmd=["true"], description="single-gpu"),
        tag="single",
        extractor=_StaticExtractor(),
        triage=GPUTriageMonitor(),
        experiments_dir=tmp_path,
    )
    assert outcome.kill_reason is None


# ── integration — runner + GPU triage built-in ─────────────────────────


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_runner_with_gpu_triage_only(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """End-to-end: runner with ONLY GPU triage (no project plateau monitor)."""
    mock_popen.return_value = _mock_popen()

    runner = SweepRunner(
        tag="gpu_only",
        planner=_StaticPlanner([IterPlan(cmd=["true"], description="one")]),
        triage=GPUTriageMonitor(),
        extractor=_StaticExtractor(),
        experiments_dir=tmp_path,
        pause_between_iters_s=0,
    )
    result = runner.run()
    assert result.iterations == 1
    assert result.kills == 0  # no triage trigger fired
