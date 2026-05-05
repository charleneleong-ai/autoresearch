"""Tests for autoresearch.subprocess_utils — kill_gracefully + wait_with_timeout
+ crash_reason_from_stdout.

Uses real subprocesses (`sleep`, `python -c`) so the kill-escalation ladder
is exercised end-to-end on the actual signal mechanism, not a mock.
"""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import time
from collections import deque

import pytest

from autoresearch.subprocess_utils import (
    crash_reason_from_stdout,
    kill_gracefully,
    wait_with_timeout,
)

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


# ── crash_reason_from_stdout ───────────────────────────────────────────


def test_crash_reason_empty_buffer_returns_unknown() -> None:
    assert crash_reason_from_stdout([]) == "unknown crash"


def test_crash_reason_no_pattern_match_returns_unknown_with_last_line() -> None:
    """No builtin pattern matches → falls back to the last non-blank line."""
    lines = ["doing fine", "still fine", "exit 1 but no traceback", ""]
    assert crash_reason_from_stdout(lines) == "unknown: exit 1 but no traceback"


def test_crash_reason_long_unknown_truncated_to_120_chars() -> None:
    long = "x" * 200
    assert crash_reason_from_stdout([long]) == f"unknown: {'x' * 120}"


def test_crash_reason_cuda_oom() -> None:
    assert crash_reason_from_stdout(["...", "torch.OutOfMemoryError: CUDA OOM here"]) == "CUDA OOM"
    # Both spellings hit the same mapper.
    assert crash_reason_from_stdout(["CUDA out of memory near step 42"]) == "CUDA OOM"


def test_crash_reason_host_killed() -> None:
    """Bare 'Killed' on its own line is the cgroup-OOM signature.

    Lines are expected to retain trailing ``\\n`` (matching what
    ``proc.stdout.readline()`` emits) so the ``re.MULTILINE`` ``^/$`` anchors
    actually anchor.
    """
    lines = ["...running...\n", "Killed\n"]
    assert crash_reason_from_stdout(lines) == "killed by host (likely cgroup OOM)"


def test_crash_reason_assertion_runtime_value_filenotfound() -> None:
    runtime_msg = "RuntimeError: shape mismatch (2,3) vs (4,5)"
    cases = [
        ("AssertionError: shapes don't match", "AssertionError: shapes don't match"),
        (runtime_msg, runtime_msg),
        ("ValueError: not enough rows", "ValueError: not enough rows"),
        (
            "FileNotFoundError: [Errno 2] No such file or directory: 'foo.yaml'",
            "FileNotFoundError: [Errno 2] No such file or directory: 'foo.yaml'",
        ),
    ]
    for line, expected in cases:
        assert crash_reason_from_stdout([line]) == expected


def test_crash_reason_truncates_messages_to_80_chars() -> None:
    long = "RuntimeError: " + ("x" * 200)
    out = crash_reason_from_stdout([long])
    assert out.startswith("RuntimeError: ")
    assert out == f"RuntimeError: {'x' * 80}"


def test_crash_reason_generic_named_error_fallback() -> None:
    """A named *Error not in the explicit list still gets categorised by the
    generic fallback."""
    out = crash_reason_from_stdout(["KeyError: 'missing_key'"])
    assert out == "KeyError: 'missing_key'"


def test_crash_reason_specific_pattern_wins_over_generic() -> None:
    """RuntimeError-specific mapper runs before the generic *Error fallback,
    so the more specific label wins."""
    lines = ["RuntimeError: foo", "AssertionError: also bar"]
    # AssertionError comes after RuntimeError in the buffer, but the regex
    # scans the whole text and AssertionError is checked before RuntimeError
    # in the pattern list — that's intentional, the order *of the list*
    # wins over the order *in the buffer*.
    assert crash_reason_from_stdout(lines) == "AssertionError: also bar"


def test_crash_reason_tail_window_caps_scan() -> None:
    """Patterns outside the tail window are ignored."""
    lines = ["RuntimeError: ancient\n"] + ["filler\n"] * 500
    # Default tail=200 — the RuntimeError is well outside it.
    out = crash_reason_from_stdout(lines)
    assert "RuntimeError" not in out
    assert out.startswith("unknown:")
    # Bumping tail to cover everything finds it.
    assert crash_reason_from_stdout(lines, tail=1000) == "RuntimeError: ancient"


def test_crash_reason_accepts_deque() -> None:
    """deque is what gemma4_rl actually uses for recent_lines."""
    buf: deque[str] = deque(maxlen=10)
    for ln in ["x", "y", "RuntimeError: from a deque"]:
        buf.append(ln)
    # We need to pass a sequence that supports negative slicing — deque
    # doesn't, but list(deque) does, so callers can splat it; here we just
    # confirm the API doesn't choke on the typical use shape.
    assert crash_reason_from_stdout(list(buf)) == "RuntimeError: from a deque"


def test_crash_reason_extra_patterns_run_first() -> None:
    """Project-supplied patterns win over builtins on conflict."""
    lines = ["RuntimeError: throttled by wandb"]
    extra = [(re.compile(r"throttled by wandb"), "wandb_throttle")]
    assert crash_reason_from_stdout(lines, extra_patterns=extra) == "wandb_throttle"


def test_crash_reason_extra_pattern_callable_mapper() -> None:
    """Callable mapper receives the regex match for extracting numerics."""
    lines = ["wandb 429 retry-after=42.0 sec"]
    extra = [
        (
            re.compile(r"wandb 429 retry-after=(\d+(?:\.\d+)?)"),
            lambda m: f"wandb_throttle: retry={m.group(1)}s",
        )
    ]
    assert crash_reason_from_stdout(lines, extra_patterns=extra) == "wandb_throttle: retry=42.0s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
