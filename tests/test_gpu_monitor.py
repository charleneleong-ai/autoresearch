"""Tests for autoresearch.gpu_monitor."""

from __future__ import annotations

import time
from unittest.mock import patch

from autoresearch.gpu_monitor import (
    GPUMonitor,
    GPUSample,
    GPUTriage,
    GPUTriageThresholds,
    _nvidia_smi_sample,
)


def test_summary_with_no_samples():
    """No samples → returns zeroed summary with a 'no samples' hint.

    Mock nvidia-smi to None so this test is deterministic on hosts that
    do/don't have a real GPU.
    """
    with patch("autoresearch.gpu_monitor._nvidia_smi_sample", return_value=None):
        mon = GPUMonitor(poll_interval_s=0.05)
        mon.start()
        mon.stop()
    s = mon.summary()
    assert s.n_samples == 0
    assert s.mean_util_pct == 0.0
    assert any("no GPU samples" in h for h in s.hints)


def test_summary_with_synthetic_samples():
    """Inject synthetic samples and verify the summary math + hints."""
    mon = GPUMonitor(low_util_pct=35, low_mem_pct=35)
    mon._samples = [
        GPUSample(util_pct=20, mem_used_gb=20.0, mem_total_gb=80.0),
        GPUSample(util_pct=25, mem_used_gb=22.0, mem_total_gb=80.0),
        GPUSample(util_pct=15, mem_used_gb=18.0, mem_total_gb=80.0),
    ]
    mon._t_start = 0.0
    mon._t_stop = 30.0
    s = mon.summary()
    assert s.n_samples == 3
    assert s.mean_util_pct == (20 + 25 + 15) / 3
    assert s.peak_util_pct == 25
    assert s.peak_mem_gb == 22.0
    assert s.mem_total_gb == 80.0
    assert s.peak_mem_pct == 22.0 / 80.0 * 100  # ~27.5%
    # peak_mem_pct < 35 → memory underused hint
    assert any("Memory underused" in h for h in s.hints)
    # mean util 20 < 35 → compute underused hint
    assert any("Compute underused" in h for h in s.hints)


def test_summary_with_well_utilised_gpu():
    """High util + memory triggers the ✓ hints, no underuse warnings."""
    mon = GPUMonitor()
    mon._samples = [
        GPUSample(util_pct=90, mem_used_gb=70.0, mem_total_gb=80.0),
        GPUSample(util_pct=92, mem_used_gb=72.0, mem_total_gb=80.0),
    ]
    mon._t_start = 0.0
    mon._t_stop = 10.0
    s = mon.summary()
    assert s.mean_util_pct == 91.0
    assert any("memory well-utilised" in h for h in s.hints)
    assert any("compute well-utilised" in h for h in s.hints)
    assert not any("underused" in h for h in s.hints)


def test_format_summary_includes_metrics():
    """Smoke: format_summary returns a multi-line string with the expected fields."""
    mon = GPUMonitor()
    mon._samples = [GPUSample(util_pct=50, mem_used_gb=40.0, mem_total_gb=80.0)]
    mon._t_start = 0.0
    mon._t_stop = 5.0
    out = mon.format_summary()
    assert "mean_util=50%" in out
    assert "peak_mem=40.0/80GB" in out
    assert "1 samples" in out


def test_context_manager_lifecycle():
    """`with GPUMonitor() as mon:` start/stops cleanly even with no nvidia-smi."""
    with patch("autoresearch.gpu_monitor._nvidia_smi_sample", return_value=None):
        with GPUMonitor(poll_interval_s=0.05) as mon:
            time.sleep(0.15)
        s = mon.summary()
        # No samples since nvidia-smi mocked to None
        assert s.n_samples == 0


def test_nvidia_smi_sample_returns_none_when_missing():
    """If nvidia-smi binary is missing, sample returns None gracefully."""
    with patch("autoresearch.gpu_monitor.shutil.which", return_value=None):
        assert _nvidia_smi_sample() is None


# ── GPUTriage ──────────────────────────────────────────────────────────


def _busy(util: int = 80, mem_gb: float = 60.0, total: float = 80.0) -> GPUSample:
    return GPUSample(util_pct=util, mem_used_gb=mem_gb, mem_total_gb=total)


def _fast_thresholds() -> GPUTriageThresholds:
    """Tight windows so tests run in milliseconds rather than minutes."""
    return GPUTriageThresholds(
        grace_s=0,
        hang_util_pct=8,
        hang_window_s=10,
        wasted_util_pct=35,
        wasted_window_s=30,
        undersized_mem_pct=50,
        undersized_window_s=60,
    )


def test_triage_warmup_grace_blocks_kills() -> None:
    """While inside the grace window, even a hang sample yields no kill."""
    triage = GPUTriage(GPUTriageThresholds(grace_s=120, hang_window_s=1))
    # Sample 0 anchors started_at; 60s later still inside the 120s grace.
    assert triage.update(GPUSample(util_pct=0, mem_used_gb=0.5, mem_total_gb=80.0), now=0.0) is None
    assert (
        triage.update(GPUSample(util_pct=0, mem_used_gb=0.5, mem_total_gb=80.0), now=60.0) is None
    )


def test_triage_hang_kill_after_window() -> None:
    """Sustained near-zero util latches the hang reason."""
    triage = GPUTriage(_fast_thresholds())
    # Start latch at t=0
    assert triage.update(_busy(util=0), now=0.0) is None
    # Inside hang_window (10s) — still pending
    assert triage.update(_busy(util=0), now=5.0) is None
    # At/past the window — fires
    reason = triage.update(_busy(util=0), now=11.0)
    assert reason is not None
    assert "likely hang" in reason
    assert "GPU util 0%" in reason
    assert "< 8%" in reason


def test_triage_wasted_compute_kill() -> None:
    """Util in the (hang, wasted) band for wasted_window_s latches wasted-compute."""
    triage = GPUTriage(_fast_thresholds())
    # 25% sits between hang (8%) and wasted (35%) — only the wasted latch should run.
    assert triage.update(_busy(util=25), now=0.0) is None
    assert triage.update(_busy(util=25), now=15.0) is None
    reason = triage.update(_busy(util=25), now=31.0)
    assert reason is not None
    assert "wasted compute" in reason
    assert "<35%" in reason or "< 35%" in reason


def test_triage_hang_zone_also_arms_wasted_latch() -> None:
    """A run that sits at util=0% throughout should fire hang first (shorter
    window) — the wasted latch is also armed but loses the race.
    """
    triage = GPUTriage(_fast_thresholds())
    triage.update(_busy(util=0), now=0.0)
    triage.update(_busy(util=0), now=5.0)
    reason = triage.update(_busy(util=0), now=11.0)
    assert reason is not None
    assert "likely hang" in reason


def test_triage_recovery_resets_latches() -> None:
    """A util sample above the wasted threshold clears both latches so prior
    underuse doesn't accumulate across recovery."""
    triage = GPUTriage(_fast_thresholds())
    triage.update(_busy(util=20), now=0.0)
    triage.update(_busy(util=20), now=15.0)
    # Recovery — both latches clear.
    assert triage.update(_busy(util=80), now=20.0) is None
    # Now another stretch of underuse — must run a *fresh* window from t=25.
    triage.update(_busy(util=20), now=25.0)
    # 30s later (=t=55), still under the wasted_window of 30s? 55-25 = 30s exactly.
    # We're at the edge — assert no kill before the window, kill at/after.
    assert triage.update(_busy(util=20), now=54.0) is None
    reason = triage.update(_busy(util=20), now=56.0)
    assert reason is not None
    assert "wasted compute" in reason


def test_triage_undersized_uses_monotonic_peak() -> None:
    """Peak memory is monotonic — a transient eval spike rescues the run from
    undersized-config kill, even if subsequent samples drop back down."""
    triage = GPUTriage(_fast_thresholds())
    # Memory at 30% (well below the 50% undersized threshold) for ages — would normally fire.
    triage.update(_busy(util=80, mem_gb=24.0, total=80.0), now=0.0)
    # One transient spike to 60% bumps peak above the threshold.
    triage.update(_busy(util=80, mem_gb=48.0, total=80.0), now=20.0)
    assert triage.peak_mem_pct == 60.0
    # Drop back to 30% — peak stays at 60% so undersized never latches.
    assert triage.update(_busy(util=80, mem_gb=24.0, total=80.0), now=120.0) is None
    assert triage.kill_reason is None


def test_triage_undersized_kill_when_peak_never_recovers() -> None:
    """Peak memory stays below threshold for the full window → undersized fires."""
    triage = GPUTriage(_fast_thresholds())
    # Peak stays at 30% throughout (24/80GB).
    triage.update(_busy(util=80, mem_gb=24.0, total=80.0), now=0.0)
    triage.update(_busy(util=80, mem_gb=24.0, total=80.0), now=30.0)
    reason = triage.update(_busy(util=80, mem_gb=24.0, total=80.0), now=61.0)
    assert reason is not None
    assert "undersized" in reason
    assert "< 50%" in reason
    assert "24/80GB" in reason


def test_triage_kill_reason_latches_idempotently() -> None:
    """Once a kill latches, further `update` calls return the same reason."""
    triage = GPUTriage(_fast_thresholds())
    triage.update(_busy(util=0), now=0.0)
    first = triage.update(_busy(util=0), now=11.0)
    assert first is not None
    # Subsequent calls — even with a healthy sample — return the same reason.
    second = triage.update(_busy(util=90, mem_gb=70.0), now=12.0)
    assert second == first
    assert triage.kill_reason == first


def test_triage_zero_total_mem_does_not_div_by_zero() -> None:
    """A sample with mem_total_gb=0 (eg. CPU-only mock) should be safe."""
    triage = GPUTriage(_fast_thresholds())
    assert triage.update(GPUSample(util_pct=80, mem_used_gb=0.0, mem_total_gb=0.0), now=0.0) is None
    assert triage.peak_mem_pct == 0.0
