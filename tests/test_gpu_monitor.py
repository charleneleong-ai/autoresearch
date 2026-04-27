"""Tests for autoresearch.gpu_monitor."""

from __future__ import annotations

import time
from unittest.mock import patch

from autoresearch.gpu_monitor import (
    GPUMonitor,
    GPUSample,
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
