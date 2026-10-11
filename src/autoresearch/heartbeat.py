"""Liveness check-in for any long-running job — the third leg beside gpu_monitor and janitor.

`gpu_monitor` does active triage on a workload it owns; `janitor` does passive triage on
orphans nobody owns. Neither answers the question that actually bites during a long run: is
this job still alive, and has it moved since last time?

A job whose own log only prints at coarse boundaries is indistinguishable from a dead one.
A BindCraft2 campaign died with `CUDA_ERROR_UNKNOWN` and sat dead for roughly four hours before
anyone looked, because its log printed once per trajectory (~35 min) and a crash produces
exactly the same silence as slow progress. A job queued behind someone else's training run was
worse: its GPU-wait helper only printed once memory was free, so the log had one line in it and
then nothing.

So every long-running job gets a sidecar that prints a line on a fixed interval, whatever the
job itself does or does not say. The contract is a **maximum of 30 minutes between check-ins**,
so silence longer than that always means the heartbeat is gone rather than "it might be fine".
`--interval` above that is clamped, deliberately: the ceiling is the point.

Two states are distinguished, because they need different responses:

    ALIVE    the process matches and its log is still growing
    STALLED  the process matches but the log has not grown — alive and doing nothing
    DEAD     nothing matches

Usage (any job, not just an autoresearch sweep):

    autoresearch-heartbeat --label sweep --pattern train.py --log logs/train.log
    autoresearch-heartbeat --label bc2 --pattern "bindcraft design" --interval 900 \
        --state-cmd "python3 -c 'import json;print(json.load(open(\\"state.json\\"))[\\"n\\"])'"

Run it detached, beside the job, per the daemon contract:

    setsid nohup autoresearch-heartbeat --label sweep --pattern train.py \
        </dev/null >>logs/heartbeat.log 2>&1 & disown

The probe is a pure function over injected callables, so the classification is unit-tested
without spawning anything (see tests/test_heartbeat.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console

# The contract: never longer than half an hour between check-ins.
MAX_INTERVAL_S = 1800
DEFAULT_INTERVAL_S = 900
# A log that has not grown in this long means the job is alive but not progressing. Keep it
# longer than the job's coarsest legitimate step (a checkpoint or eval phase) or it false-fires.
DEFAULT_STALL_AFTER_S = 1800


class Status(StrEnum):
    """What the last probe found."""

    ALIVE = "alive"
    STALLED = "stalled"
    DEAD = "dead"

    @property
    def is_healthy(self) -> bool:
        return self is Status.ALIVE


@dataclass(frozen=True)
class Checkin:
    """One check-in, ready to print."""

    label: str
    status: Status
    elapsed_s: int
    log_idle_s: int | None = None
    gpu: str | None = None
    detail: str | None = None

    def line(self, at: str) -> str:
        """The log line. Upper-case status so an unhealthy run greps out of a long file."""
        parts = [at, "HEARTBEAT", self.label, self.status.name, f"elapsed={self.elapsed_s}s"]
        if self.log_idle_s is not None:
            parts.append(f"log_idle={self.log_idle_s}s")
        if self.gpu:
            parts.append(f"gpu={self.gpu}")
        if self.detail:
            parts.append(self.detail)
        return " ".join(parts)


def classify(*, running: bool, log_idle_s: int | None, stall_after_s: int) -> Status:
    """A matching process whose log has gone quiet is stalled, which is not the same as dead."""
    if not running:
        return Status.DEAD
    if log_idle_s is not None and log_idle_s >= stall_after_s:
        return Status.STALLED
    return Status.ALIVE


def process_matches(pattern: str) -> bool:
    """True if a process other than this heartbeat (or the shell that launched it) matches, via
    pgrep -f; the pattern is in the heartbeat's own command line, so it always matches itself."""
    if not shutil.which("pgrep"):
        return False
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
    return bool({int(pid) for pid in found.stdout.split()} - {os.getpid(), os.getppid()})


def log_idle_seconds(log: Path | None, now: float) -> int | None:
    """Seconds since the log was last written, or None when there is no log to watch."""
    if log is None or not log.exists():
        return None
    return max(0, int(now - log.stat().st_mtime))


def gpu_line() -> str | None:
    """`util%,freeMiB` for the first GPU, or None off-GPU — the sidecar must not require one."""
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()[0].replace(" ", "") or None


def state_output(command: str | None) -> str | None:
    """First line of an optional command describing the job's own progress."""
    if not command:
        return None
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    lines = result.stdout.strip().splitlines()
    return lines[0] if lines else None


@dataclass(frozen=True)
class Probe:
    """Everything a check-in needs, with the impure parts injected so they can be faked."""

    label: str
    pattern: str
    log: Path | None = None
    state_cmd: str | None = None
    stall_after_s: int = DEFAULT_STALL_AFTER_S
    running: Callable[[str], bool] = process_matches
    gpu: Callable[[], str | None] = gpu_line
    state: Callable[[str | None], str | None] = state_output

    def check(self, *, started: float, now: float) -> Checkin:
        idle = log_idle_seconds(self.log, now)
        return Checkin(
            label=self.label,
            status=classify(
                running=self.running(self.pattern),
                log_idle_s=idle,
                stall_after_s=self.stall_after_s,
            ),
            elapsed_s=max(0, int(now - started)),
            log_idle_s=idle,
            gpu=self.gpu(),
            detail=self.state(self.state_cmd),
        )


def clamp_interval(seconds: int) -> int:
    """Hold callers to the 30-minute ceiling rather than trusting them to remember it."""
    return min(max(1, seconds), MAX_INTERVAL_S)


app = typer.Typer(add_completion=False, help=__doc__.split("\n")[0])


@app.command()
def main(
    label: str = typer.Option(..., "--label", help="name for this job in the log"),
    pattern: str = typer.Option(..., "--pattern", help="pgrep -f pattern identifying the job"),
    interval: int = typer.Option(
        DEFAULT_INTERVAL_S,
        "--interval",
        help=f"seconds between check-ins (clamped to {MAX_INTERVAL_S})",
    ),
    log: Path | None = typer.Option(
        None, "--log", help="the job's log; a log that stops growing means STALLED"
    ),
    state_cmd: str | None = typer.Option(
        None, "--state-cmd", help="command whose first stdout line is appended"
    ),
    stall_after: int = typer.Option(
        DEFAULT_STALL_AFTER_S,
        "--stall-after",
        help="seconds of log silence before reporting STALLED",
    ),
    once: bool = typer.Option(False, "--once", help="print one check-in and exit"),
) -> None:
    """Check in on a long-running job at a fixed interval, forever, until killed."""
    console = Console()
    every = clamp_interval(interval)
    if every != interval:
        console.print(
            f"[yellow]interval {interval}s clamped to {every}s[/] — check-ins are "
            f"capped at {MAX_INTERVAL_S}s so silence always means the heartbeat died"
        )

    probe = Probe(
        label=label, pattern=pattern, log=log, state_cmd=state_cmd, stall_after_s=stall_after
    )
    started = time.time()
    previous: Status | None = None
    while True:
        checkin = probe.check(started=started, now=time.time())
        console.print(
            checkin.line(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            style=None if checkin.status.is_healthy else "bold red",
            highlight=False,
        )
        # announce the transition too, so a crash is greppable without diffing timestamps
        if previous is Status.ALIVE and not checkin.status.is_healthy:
            console.print(
                f"HEARTBEAT {label} WENT_{checkin.status.name} after {checkin.elapsed_s}s",
                style="bold red",
                highlight=False,
            )
        previous = checkin.status
        if once:
            return
        time.sleep(every)


if __name__ == "__main__":
    app()
