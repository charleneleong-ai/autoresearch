"""Tests for autoresearch.subprocess_utils — kill_gracefully + wait_with_timeout.

Uses real subprocesses (`sleep`, `python -c`) so the kill-escalation ladder
is exercised end-to-end on the actual signal mechanism, not a mock.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import pytest

from autoresearch.subprocess_utils import kill_gracefully, wait_with_timeout

# ── kill_gracefully ────────────────────────────────────────────────────


def test_kill_gracefully_returns_immediately_if_already_exited() -> None:
    proc = subprocess.Popen(["true"])
    proc.wait()  # exit cleanly
    assert kill_gracefully(proc) == 0


def test_kill_gracefully_sigint_clean_exit() -> None:
    """Process that handles SIGINT and exits within the grace window."""
    # Python prints "ready", traps SIGINT, exits with code 7.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time;"
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(7));"
            "print('ready', flush=True);"
            "time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
    )
    # Wait for the trap to be installed
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"ready"

    rc = kill_gracefully(proc, sigint_grace_s=5)
    assert rc == 7


def test_kill_gracefully_escalates_to_sigterm_when_sigint_ignored() -> None:
    """SIGINT-ignoring process should be killed by SIGTERM in the next window."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time;"
            "signal.signal(signal.SIGINT, signal.SIG_IGN);"
            "print('ready', flush=True);"
            "time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"ready"

    escalations: list[str] = []
    rc = kill_gracefully(
        proc,
        sigint_grace_s=2,
        sigterm_grace_s=5,
        on_escalation=escalations.append,
    )
    # SIGTERM = 15; killed-by-signal returncode is -15
    assert rc == -signal.SIGTERM
    assert any("sigterm" in msg.lower() for msg in escalations)


def test_kill_gracefully_escalates_to_sigkill_when_sigterm_also_ignored() -> None:
    """SIGINT+SIGTERM both ignored → SIGKILL final escalation."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time;"
            "signal.signal(signal.SIGINT, signal.SIG_IGN);"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "print('ready', flush=True);"
            "time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"ready"

    escalations: list[str] = []
    rc = kill_gracefully(
        proc,
        sigint_grace_s=2,
        sigterm_grace_s=2,
        on_escalation=escalations.append,
    )
    assert rc == -signal.SIGKILL
    assert any("sigkill" in msg.lower() for msg in escalations)


# ── wait_with_timeout ──────────────────────────────────────────────────


def test_wait_with_timeout_natural_exit() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    rc, kill_reason = wait_with_timeout(proc, timeout_s=10)
    assert rc == 3
    assert kill_reason is None


def test_wait_with_timeout_wall_clock_kill() -> None:
    """Process that won't exit on its own should be killed by the wall clock."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time;"
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0));"
            "print('ready', flush=True);"
            "time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"ready"

    start = time.monotonic()
    rc, kill_reason = wait_with_timeout(proc, timeout_s=2, poll_s=0.5)
    elapsed = time.monotonic() - start
    assert kill_reason is not None
    assert "timeout" in kill_reason.lower()
    assert "2" in kill_reason  # the timeout value
    assert elapsed < 10  # actually killed, not waited 60s
    assert rc == 0  # SIGINT handler exited cleanly


def test_wait_with_timeout_should_kill_callback_fires() -> None:
    """should_kill returning a string triggers kill with that reason."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time;"
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0));"
            "print('ready', flush=True);"
            "time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"ready"

    poll_count = [0]

    def should_kill() -> str | None:
        poll_count[0] += 1
        if poll_count[0] >= 3:
            return "triage: synthetic plateau"
        return None

    rc, kill_reason = wait_with_timeout(proc, timeout_s=60, poll_s=0.2, should_kill=should_kill)
    assert kill_reason == "triage: synthetic plateau"
    assert poll_count[0] >= 3


def test_wait_with_timeout_should_kill_returning_none_keeps_waiting() -> None:
    """should_kill that returns None should never trigger a kill."""
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    rc, kill_reason = wait_with_timeout(proc, timeout_s=10, poll_s=0.2, should_kill=lambda: None)
    assert rc == 7
    assert kill_reason is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
