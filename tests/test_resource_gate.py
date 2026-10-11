"""Tests for autoresearch.resource_gate — hold a queued job until the machine has room."""

from __future__ import annotations

import pytest

from autoresearch.resource_gate import Gate, Room, ram_free_gb


class FakeTime:
    """A clock that moves when the gate sleeps, and a millisecond per read (no two reads agree)."""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        self.now += 0.001
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def gate_over(rooms: list[Room], **kw: int) -> tuple[Gate, list[str]]:
    """A gate whose probe returns `rooms` in turn (the last one forever), plus its report log."""
    t, seen = FakeTime(), iter(rooms)

    def probe() -> Room:
        return next(seen, rooms[-1])

    gate = Gate(gpu_gb=20, ram_gb=30, poll_s=10, probe=probe, clock=t.clock, sleep=t.sleep, **kw)
    return gate, []


FITS, NO_GPU = Room(40.0, 60.0), Room(5.0, 60.0)


class TestShortfall:
    @pytest.mark.parametrize(
        ("room", "short"),
        [
            (FITS, []),
            (NO_GPU, ["gpu 5.0/20 GB"]),
            (Room(None, 10.0), ["gpu ?/20 GB", "ram 10.0/30 GB"]),  # unreadable counts as short
        ],
    )
    def test_names_each_resource_that_does_not_fit(self, room, short):
        assert room.shortfall(20, 30) == short

    def test_a_resource_the_job_does_not_need_is_ignored(self):
        assert Room(None, None).shortfall(0, 0) == []


class TestGate:
    def test_waits_for_a_stable_window_not_one_free_sample(self):
        gate, log = gate_over([NO_GPU, FITS, NO_GPU, FITS], stable_s=30)
        assert gate.wait(log.append)
        assert log[-1].startswith("READY after 60s")  # the fit at 10s broke at 20s; 30..60s holds

    def test_checks_in_while_waiting_and_times_out(self):
        gate, log = gate_over([NO_GPU], checkin_s=100, timeout_s=350)
        assert not gate.wait(log.append)
        waiting = [m for m in log if m.startswith("WAITING")]
        assert [m.split("s:")[0] for m in waiting] == [
            "WAITING 0",
            "WAITING 100",
            "WAITING 200",
            "WAITING 300",
        ]
        assert "short: gpu 5.0/20 GB" in waiting[0] and log[-1].startswith(
            "TIMEOUT after 350s: short of gpu"
        )


def test_ram_free_reads_mem_available(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  98000000 kB\nMemFree:   1000000 kB\nMemAvailable: 31457280 kB\n")
    assert ram_free_gb(meminfo) == 30.0 and ram_free_gb(tmp_path / "absent") is None
