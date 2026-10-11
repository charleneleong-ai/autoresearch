"""Tests for autoresearch.heartbeat — liveness check-in for any long-running job.

Classification is tested with injected probes; no real processes are spawned.
Mirrors test_janitor / test_gpu_monitor (inject data, assert pure behaviour).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoresearch.heartbeat import (
    DEFAULT_STALL_AFTER_S,
    MAX_INTERVAL_S,
    Checkin,
    Probe,
    Status,
    app,
    clamp_interval,
    classify,
    log_idle_seconds,
)

runner = CliRunner()


def _running(_pattern: str, *, answer: bool) -> bool:
    """Picklable stand-in for pgrep, so no test depends on a real process."""
    return answer


def _no_gpu() -> None:
    return None


def _state(_command: str | None, *, text: str | None = None) -> str | None:
    return text


def probe(
    *,
    alive: bool,
    log: Path | None = None,
    state: str | None = None,
    stall_after_s: int = DEFAULT_STALL_AFTER_S,
) -> Probe:
    return Probe(
        label="job",
        pattern="train.py",
        log=log,
        state_cmd="ignored",
        stall_after_s=stall_after_s,
        running=partial(_running, answer=alive),
        gpu=_no_gpu,
        state=partial(_state, text=state),
    )


class TestClassify:
    """Alive-but-stalled is a distinct state from dead; they need different responses."""

    @pytest.mark.parametrize(
        "running,log_idle_s,expected",
        [
            (True, None, Status.ALIVE),  # no log to watch — liveness only
            (True, 0, Status.ALIVE),
            (True, 1799, Status.ALIVE),  # just inside the window
            (True, 1800, Status.STALLED),  # boundary counts as stalled
            (True, 7200, Status.STALLED),
            (False, 0, Status.DEAD),  # a fresh log does not resurrect a dead process
            (False, 9999, Status.DEAD),
        ],
    )
    def test_states(self, running, log_idle_s, expected):
        assert (
            classify(running=running, log_idle_s=log_idle_s, stall_after_s=DEFAULT_STALL_AFTER_S)
            is expected
        )

    def test_only_alive_is_healthy(self):
        assert Status.ALIVE.is_healthy
        assert not Status.STALLED.is_healthy
        assert not Status.DEAD.is_healthy


class TestClampInterval:
    """The 30-minute ceiling is the contract, so it is enforced rather than documented."""

    @pytest.mark.parametrize(
        "given,expected",
        [
            (900, 900),
            (1800, 1800),
            (3600, MAX_INTERVAL_S),
            (0, 1),
            (-5, 1),
        ],
    )
    def test_clamped(self, given, expected):
        assert clamp_interval(given) == expected


class TestLogIdleSeconds:
    """A log that stops growing is the only evidence of a stalled-but-running job."""

    def test_none_without_a_log(self):
        assert log_idle_seconds(None, now=1000.0) is None

    def test_none_when_the_log_does_not_exist_yet(self, tmp_path):
        assert log_idle_seconds(tmp_path / "absent.log", now=1000.0) is None

    def test_measures_age_against_mtime(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("step 1\n")
        mtime = log.stat().st_mtime
        assert log_idle_seconds(log, now=mtime + 120) == 120

    def test_never_negative_when_the_clock_disagrees(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("step 1\n")
        assert log_idle_seconds(log, now=log.stat().st_mtime - 60) == 0


class TestProbe:
    """What a check-in reports, with every impure dependency injected."""

    def test_reports_alive_with_elapsed_and_detail(self):
        checkin = probe(alive=True, state="traj=4 accepted=1").check(started=100.0, now=460.0)
        assert checkin.status is Status.ALIVE
        assert checkin.elapsed_s == 360
        assert checkin.detail == "traj=4 accepted=1"

    def test_reports_dead_when_nothing_matches(self):
        assert probe(alive=False).check(started=0.0, now=10.0).status is Status.DEAD

    def test_stall_threshold_is_configurable(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("x")
        now = log.stat().st_mtime + 600
        assert (
            probe(alive=True, log=log, stall_after_s=300).check(started=0.0, now=now).status
            is Status.STALLED
        )
        assert (
            probe(alive=True, log=log, stall_after_s=1200).check(started=0.0, now=now).status
            is Status.ALIVE
        )

    def test_survives_having_no_gpu(self):
        assert probe(alive=True).check(started=0.0, now=1.0).gpu is None


class TestCheckinLine:
    """The line is what someone greps a long log for."""

    def test_unhealthy_status_is_upper_case_and_greppable(self):
        line = Checkin("bc2", Status.DEAD, elapsed_s=42).line("2026-10-11T06:00:00Z")
        assert "HEARTBEAT bc2 DEAD elapsed=42s" in line

    def test_optional_fields_are_omitted_when_absent(self):
        line = Checkin("bc2", Status.ALIVE, elapsed_s=1).line("T")
        assert "log_idle=" not in line and "gpu=" not in line

    def test_optional_fields_are_included_when_present(self):
        line = Checkin(
            "bc2", Status.ALIVE, elapsed_s=1, log_idle_s=5, gpu="0%,14483MiB", detail="traj=1"
        ).line("T")
        assert "log_idle=5s" in line and "gpu=0%,14483MiB" in line and "traj=1" in line


class TestCli:
    """The sidecar runs as a console script beside the job."""

    def test_once_prints_a_single_checkin(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("step\n")
        result = runner.invoke(
            app,
            [
                "--label",
                "sweep",
                "--pattern",
                "definitely-not-running",
                "--log",
                str(log),
                "--once",
            ],
        )
        assert result.exit_code == 0
        assert "HEARTBEAT sweep DEAD" in result.stdout

    def test_warns_when_the_interval_exceeds_the_ceiling(self):
        result = runner.invoke(
            app, ["--label", "x", "--pattern", "y", "--interval", "3600", "--once"]
        )
        assert result.exit_code == 0
        assert "clamped" in result.stdout
