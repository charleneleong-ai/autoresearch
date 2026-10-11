"""Launch gate: hold a queued job until the machine has room for it.

On a shared host a chain script that starts its next job the moment the previous one exits
walks into whatever else is running. Two failures on one A100 in one night: a 96-channel U-Net
started while another user's training held 65 of 80 GB and died with CUDA OOM, and a 28 GB
training run started beside a 15 GB build and was taken by the host OOM killer. A BindCraft2
relaunch that dropped its `wait_gpu` gate started into a card it could not fit on.

`autoresearch-wait` blocks until free GPU memory and available host RAM have both stayed above
what the job needs for `--stable` seconds (one free sample between someone else's epochs is not
room), then exits 0, so a chain reads:

    autoresearch-wait --label grid_v3 --gpu-free-gb 24 --ram-free-gb 32 && python train.py ...

While waiting it checks in at least every 30 minutes, like `autoresearch-heartbeat`: a gate
that only speaks once memory frees up leaves a one-line log that reads exactly like a hang.
`--timeout` exits 1 instead of waiting forever.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import typer

from autoresearch.heartbeat import MAX_INTERVAL_S, clamp_interval

MEMINFO = Path("/proc/meminfo")


@dataclass(frozen=True)
class Room:
    """Free memory in GB; None when it cannot be read (no GPU, not Linux)."""

    gpu_free_gb: float | None
    ram_free_gb: float | None

    def shortfall(self, gpu_gb: float, ram_gb: float) -> list[str]:
        """Why the job does not fit yet; empty when it does. An unreadable resource the job
        needs counts as short, so a gate never waves a job through on a failed probe."""
        short = []
        for name, need, have in (
            ("gpu", gpu_gb, self.gpu_free_gb),
            ("ram", ram_gb, self.ram_free_gb),
        ):
            if need > 0 and (have is None or have < need):
                short.append(f"{name} {'?' if have is None else f'{have:.1f}'}/{need:g} GB")
        return short

    def line(self) -> str:
        gpu, ram = (
            ("?" if v is None else f"{v:.1f}") for v in (self.gpu_free_gb, self.ram_free_gb)
        )
        return f"gpu_free={gpu}GB ram_free={ram}GB"


def gpu_free_gb(index: int = 0) -> float | None:
    """Free memory on one GPU, from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.split()[0]) / 1024
    except (IndexError, ValueError):
        return None


def ram_free_gb(meminfo: Path = MEMINFO) -> float | None:
    """MemAvailable (what a new process can take without swapping), from /proc/meminfo."""
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return None


def measure() -> Room:
    return Room(gpu_free_gb(), ram_free_gb())


@dataclass
class Gate:
    """Waits for `stable_s` of continuous room; clock, sleep and probe are injected for tests."""

    gpu_gb: float
    ram_gb: float
    stable_s: int = 120
    poll_s: int = 30
    checkin_s: int = 900
    timeout_s: int = 0
    probe: Callable[[], Room] = measure
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def wait(self, report: Callable[[str], None]) -> bool:
        """True once the job fits; False on timeout. Reports every check-in and the outcome."""
        start, last_report = self.clock(), None
        fit_since: float | None = None
        while True:
            now, room = self.clock(), self.probe()
            short = room.shortfall(self.gpu_gb, self.ram_gb)
            fit_since = None if short else (now if fit_since is None else fit_since)
            waited = int(now - start)
            if fit_since is not None and now - fit_since >= self.stable_s:
                report(f"READY after {waited}s: {room.line()}")
                return True
            if self.timeout_s and now - start >= self.timeout_s:
                report(f"TIMEOUT after {waited}s: short of {', '.join(short) or 'a stable window'}")
                return False
            if last_report is None or now - last_report >= self.checkin_s:
                report(
                    f"WAITING {waited}s: {room.line()}"
                    + (f" (short: {', '.join(short)})" if short else "")
                )
                last_report = now
            self.sleep(self.poll_s)


app = typer.Typer(add_completion=False, help=__doc__.split("\n")[0])


@app.command()
def main(
    label: str = typer.Option(..., "--label", help="name of the queued job in the log"),
    gpu_free_gb: float = typer.Option(0.0, "--gpu-free-gb", help="free GPU memory the job needs"),
    ram_free_gb: float = typer.Option(
        0.0, "--ram-free-gb", help="available host RAM the job needs"
    ),
    stable: int = typer.Option(120, "--stable", help="seconds the room must persist"),
    poll: int = typer.Option(30, "--poll", help="seconds between probes"),
    checkin: int = typer.Option(
        900, "--checkin", help=f"seconds between check-ins (clamped to {MAX_INTERVAL_S})"
    ),
    timeout: int = typer.Option(0, "--timeout", help="give up after this many seconds (0 = never)"),
) -> None:
    """Block until the job fits on this machine, then exit 0 (1 on timeout)."""

    def report(message: str) -> None:
        # flushed: under nohup a buffered check-in never reaches the log, which is the failure
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"{stamp} WAIT {label} {message}", flush=True)

    gate = Gate(gpu_free_gb, ram_free_gb, stable, poll, clamp_interval(checkin), timeout)
    raise typer.Exit(0 if gate.wait(report) else 1)


if __name__ == "__main__":
    app()
