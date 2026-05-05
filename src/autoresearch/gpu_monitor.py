"""GPU utilisation + memory monitor — reusable across training, sweeps, eval.

A lightweight background poller that samples `nvidia-smi` while a workload
runs, then emits a summary with mean util, peak memory, throughput, and
rightsizing hints (matching the autoresearch orchestrator's thresholds).

Usage as a context manager:

    from autoresearch.gpu_monitor import GPUMonitor

    with GPUMonitor() as mon:
        run_my_eval()  # or training, or sweep, etc.

    print(mon.format_summary())
    # → mean_util=42%  peak_mem=18.9/80GB  runtime=12m
    #   • Memory underused (18.9/80GB, 61GB free) — try larger batch
    #   • Compute underused (mean 42%) — possible: dataloader / small batch / pipe-bound

Or programmatic:

    mon = GPUMonitor(poll_interval_s=10)
    mon.start()
    ...your work...
    mon.stop()
    summary = mon.summary()  # dict: mean_util, peak_mem_gb, ...

Designed to drop in with no extra deps (uses subprocess + threading).
Returns a no-op if nvidia-smi is unavailable so it's safe to leave in
CPU-only environments.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Defaults match the autoresearch orchestrator's triage thresholds (calibrated
# against an A100-class run). Override per workload if your hardware envelope
# is different.
DEFAULT_POLL_INTERVAL_S = 10
DEFAULT_LOW_UTIL_PCT = 35  # mean util below this → compute-underused
DEFAULT_LOW_MEM_PCT = 35  # peak mem % below this → undersized config


@dataclass
class GPUSample:
    """One nvidia-smi snapshot."""

    util_pct: int
    mem_used_gb: float
    mem_total_gb: float


@dataclass
class GPUSummary:
    """Aggregated stats over a monitored interval."""

    n_samples: int
    runtime_s: float
    mean_util_pct: float
    peak_util_pct: int
    peak_mem_gb: float
    mem_total_gb: float
    hints: list[str] = field(default_factory=list)

    @property
    def peak_mem_pct(self) -> float:
        if self.mem_total_gb <= 0:
            return 0.0
        return self.peak_mem_gb / self.mem_total_gb * 100


def _nvidia_smi_sample() -> GPUSample | None:
    """Return current util/mem or None if nvidia-smi missing or fails."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            text=True,
        )
        line = out.strip().splitlines()[0]
        util, used, total = (x.strip() for x in line.split(","))
        return GPUSample(
            util_pct=int(util),
            mem_used_gb=round(int(used) / 1024, 2),
            mem_total_gb=round(int(total) / 1024, 2),
        )
    except Exception:
        return None


class GPUMonitor:
    """Background poller for nvidia-smi util + memory.

    Use as a context manager (recommended) or call `start()` / `stop()`
    directly. After stop, `summary()` returns the aggregated stats.
    """

    def __init__(
        self,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        low_util_pct: int = DEFAULT_LOW_UTIL_PCT,
        low_mem_pct: int = DEFAULT_LOW_MEM_PCT,
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self.low_util_pct = low_util_pct
        self.low_mem_pct = low_mem_pct

        self._samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start: float | None = None
        self._t_stop: float | None = None

    def __enter__(self) -> GPUMonitor:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return  # already running
        self._t_start = time.monotonic()

        def _loop() -> None:
            while not self._stop.is_set():
                s = _nvidia_smi_sample()
                if s is not None:
                    self._samples.append(s)
                self._stop.wait(self.poll_interval_s)

        self._thread = threading.Thread(target=_loop, name="GPUMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.poll_interval_s + 5)
        self._thread = None
        self._t_stop = time.monotonic()

    def summary(self) -> GPUSummary:
        if not self._samples:
            return GPUSummary(
                n_samples=0,
                runtime_s=(self._t_stop or 0) - (self._t_start or 0),
                mean_util_pct=0.0,
                peak_util_pct=0,
                peak_mem_gb=0.0,
                mem_total_gb=0.0,
                hints=["no GPU samples (nvidia-smi unavailable or failed)"],
            )
        utils = [s.util_pct for s in self._samples]
        mems = [s.mem_used_gb for s in self._samples]
        total = self._samples[0].mem_total_gb
        runtime_s = (self._t_stop or time.monotonic()) - (self._t_start or time.monotonic())
        peak_mem = max(mems)

        hints: list[str] = []
        if peak_mem / total * 100 < self.low_mem_pct:
            free = total - peak_mem
            hints.append(
                f"Memory underused (peak {peak_mem:.1f}/{total:.0f}GB, "
                f"{free:.0f}GB free) — consider larger batch / num_generations / "
                "max_seq_length / lora_rank"
            )
        elif peak_mem / total * 100 >= 85:
            hints.append(f"GPU memory well-utilised (peak {peak_mem:.1f}/{total:.0f}GB) ✓")

        mean_util = sum(utils) / len(utils)
        if mean_util < self.low_util_pct:
            hints.append(
                f"Compute underused (mean {mean_util:.0f}%) — possible: "
                "dataloader bottleneck / small batch / sequential gen / pipe-bound"
            )
        elif mean_util >= 85:
            hints.append(f"GPU compute well-utilised (mean {mean_util:.0f}%) ✓")

        return GPUSummary(
            n_samples=len(self._samples),
            runtime_s=runtime_s,
            mean_util_pct=mean_util,
            peak_util_pct=max(utils),
            peak_mem_gb=peak_mem,
            mem_total_gb=total,
            hints=hints,
        )

    def format_summary(self) -> str:
        s = self.summary()
        lines = [
            f"\n[gpu_monitor] mean_util={s.mean_util_pct:.0f}%  "
            f"peak={s.peak_util_pct}%  "
            f"peak_mem={s.peak_mem_gb:.1f}/{s.mem_total_gb:.0f}GB  "
            f"runtime={s.runtime_s:.0f}s  ({s.n_samples} samples)"
        ]
        for h in s.hints:
            lines.append(f"  • {h}")
        return "\n".join(lines)


# ── stateful threshold tracker ────────────────────────────────────────


@dataclass
class GPUTriageThresholds:
    """Kill-trigger thresholds for an A100-class run.

    Defaults match the CLAUDE.md prescription: hang at <8% util for 5min,
    wasted-compute at <35% util for 15min, undersized config at peak mem
    <50% for 30min, all after a 3-min warmup. Override per-workload if
    your hardware envelope or memory budget differs.
    """

    grace_s: int = 180
    hang_util_pct: int = 8
    hang_window_s: int = 300
    wasted_util_pct: int = 35
    wasted_window_s: int = 900
    undersized_mem_pct: int = 50
    undersized_window_s: int = 1800


class GPUTriage:
    """Stateful kill-trigger over a stream of :class:`GPUSample` snapshots.

    Caller polls ``nvidia-smi`` (e.g. via :class:`GPUMonitor` or directly)
    and feeds each sample to :meth:`update`, which returns a ``kill_reason``
    string when one of the three thresholds latches, or ``None`` to keep
    going.

    Three independent kill classes — each catches a failure mode the other
    two would miss:

    * **Hang** — instantaneous util below ``hang_util_pct`` for
      ``hang_window_s``. Catches deadlocked workloads (DDP allreduce stall,
      CUDA driver wedge) that a wasted-compute detector wouldn't fire on
      until much later.
    * **Wasted compute** — util below ``wasted_util_pct`` for the broader
      ``wasted_window_s``. Catches sub-35% sustained underuse that's
      legitimate compute (so the hang detector ignores it) but burns budget
      at low throughput.
    * **Undersized config** — *peak* memory (monotonic over the whole run)
      below ``undersized_mem_pct`` for ``undersized_window_s``. Uses the
      cumulative peak so a transient eval/checkpoint spike can rescue an
      otherwise memory-thin config — only fires if no phase ever hit the
      memory budget.

    All checks honour a startup ``grace_s`` warmup so model load + first
    eval don't false-trigger.

    Once any kill latches, the same reason is returned indefinitely — the
    caller is expected to act on the first non-None and stop polling.
    """

    def __init__(self, thresholds: GPUTriageThresholds | None = None) -> None:
        self.thresholds = thresholds or GPUTriageThresholds()
        self._started_at: float | None = None
        self._hang_since: float | None = None
        self._wasted_since: float | None = None
        self._undersized_since: float | None = None
        self._peak_mem_pct: float = 0.0
        self._kill_reason: str | None = None

    def update(
        self,
        sample: GPUSample,
        *,
        now: float | None = None,
    ) -> str | None:
        """Feed one GPU sample. Returns a ``kill_reason`` string or ``None``.

        Parameters
        ----------
        sample
            One ``nvidia-smi`` snapshot.
        now
            Override the monotonic clock. Defaults to ``time.monotonic()``.
            Tests pass an explicit value to make the timing windows
            deterministic.
        """
        if self._kill_reason is not None:
            return self._kill_reason

        ts = time.monotonic() if now is None else now
        if self._started_at is None:
            self._started_at = ts
        if ts - self._started_at < self.thresholds.grace_s:
            return None

        util = sample.util_pct
        mem_pct = sample.mem_used_gb / sample.mem_total_gb * 100 if sample.mem_total_gb > 0 else 0.0
        self._peak_mem_pct = max(self._peak_mem_pct, mem_pct)

        t = self.thresholds

        # ── util-based ladder (hang ⊂ wasted) ──────────────────────────
        if util < t.hang_util_pct:
            if self._hang_since is None:
                self._hang_since = ts
            elif ts - self._hang_since >= t.hang_window_s:
                self._kill_reason = (
                    f"GPU util {util}% < {t.hang_util_pct}% "
                    f"for {t.hang_window_s // 60}min+ — likely hang"
                )
                return self._kill_reason
            # Hang zone is also wasted-compute zone — start that latch too.
            if self._wasted_since is None:
                self._wasted_since = ts
        elif util < t.wasted_util_pct:
            self._hang_since = None
            if self._wasted_since is None:
                self._wasted_since = ts
            elif ts - self._wasted_since >= t.wasted_window_s:
                self._kill_reason = (
                    f"GPU util sustained <{t.wasted_util_pct}% "
                    f"for {t.wasted_window_s // 60}min+ — wasted compute "
                    f"(last sample {util}%, "
                    f"{sample.mem_used_gb:.0f}/{sample.mem_total_gb:.0f}GB)"
                )
                return self._kill_reason
        else:
            self._hang_since = None
            self._wasted_since = None

        # ── undersized (peak-mem-based — uses cumulative peak) ─────────
        if self._peak_mem_pct < t.undersized_mem_pct:
            if self._undersized_since is None:
                self._undersized_since = ts
            elif ts - self._undersized_since >= t.undersized_window_s:
                self._kill_reason = (
                    f"peak GPU mem {self._peak_mem_pct:.0f}% "
                    f"< {t.undersized_mem_pct}% "
                    f"for {t.undersized_window_s // 60}min+ — undersized config "
                    f"({sample.mem_used_gb:.0f}/{sample.mem_total_gb:.0f}GB; "
                    "try larger batch / num_generations / max_seq_length)"
                )
                return self._kill_reason
        else:
            self._undersized_since = None

        return None

    @property
    def peak_mem_pct(self) -> float:
        """Cumulative peak memory percentage seen so far across all samples."""
        return self._peak_mem_pct

    @property
    def kill_reason(self) -> str | None:
        """The latched kill reason, or ``None`` if no threshold has fired."""
        return self._kill_reason


__all__ = [
    "DEFAULT_LOW_MEM_PCT",
    "DEFAULT_LOW_UTIL_PCT",
    "DEFAULT_POLL_INTERVAL_S",
    "GPUMonitor",
    "GPUSample",
    "GPUSummary",
    "GPUTriage",
    "GPUTriageThresholds",
]
