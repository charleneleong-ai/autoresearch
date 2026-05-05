"""Subprocess lifecycle helpers for the sweep loop.

Both downstream consumers (orak-2025-starter-kit, gemma4-rlvr) had near-
verbatim implementations of the SIGINT → SIGTERM → SIGKILL escalation
ladder used to gracefully kill an in-flight iter on triage trigger.
This module is the extracted reusable form.

The pattern: try cooperative shutdown first (SIGINT lets the subprocess
clean up open file handles, network sockets, wandb runs, GPU memory),
escalate after a grace window if it doesn't exit, finally hard-kill
after a second grace window. Without the escalation ladder, half the
real-world kills end up with leaked GPU memory or zombie helper
processes (wandb-core, game servers, etc.).

``crash_reason_from_stdout`` is the matching post-mortem helper: when a
subprocess exits non-zero (and was not killed by triage), scan its tail
buffer for a structured failure-mode label.
"""

from __future__ import annotations

import re
import signal
import subprocess
import time
from collections.abc import Callable, Sequence


def kill_gracefully(
    proc: subprocess.Popen,
    *,
    sigint_grace_s: int = 60,
    sigterm_grace_s: int = 30,
    on_escalation: Callable[[str], None] | None = None,
) -> int:
    """Escalate kill signals to ``proc`` until it exits.

    Sequence:
      1. Send SIGINT (Ctrl-C). Wait up to ``sigint_grace_s`` seconds.
      2. If still alive, send SIGTERM. Wait up to ``sigterm_grace_s`` seconds.
      3. If still alive, send SIGKILL. Wait up to 5 seconds (it should always
         take). Returns the proc's returncode regardless.

    No-op if the process has already exited (returncode already set).

    Parameters
    ----------
    proc
        The ``subprocess.Popen`` instance to kill.
    sigint_grace_s
        How long to wait after SIGINT before escalating to SIGTERM. Defaults
        to 60 seconds — long enough for typical training subprocesses to
        flush wandb runs, save checkpoints, close inference connections.
    sigterm_grace_s
        How long to wait after SIGTERM before escalating to SIGKILL. Defaults
        to 30 seconds.
    on_escalation
        Optional callback invoked with the escalation reason as a string,
        e.g. ``"sigint timeout, escalating to sigterm"``. Useful for log
        breadcrumbs. Called at most twice per kill (sigint→term, term→kill).

    Returns
    -------
    int
        ``proc.returncode``. Will be a negative number (negated signal value)
        when the process was killed by signal; positive when it exited with
        a Python-level error code; zero on cooperative clean exit.
    """
    if proc.poll() is not None:
        return proc.returncode

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=sigint_grace_s)
        return proc.returncode
    except subprocess.TimeoutExpired:
        if on_escalation is not None:
            on_escalation(f"sigint grace ({sigint_grace_s}s) expired, escalating to SIGTERM")

    proc.terminate()
    try:
        proc.wait(timeout=sigterm_grace_s)
        return proc.returncode
    except subprocess.TimeoutExpired:
        if on_escalation is not None:
            on_escalation(f"sigterm grace ({sigterm_grace_s}s) expired, escalating to SIGKILL")

    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # SIGKILL not exiting in 5s = kernel-level zombie, nothing more we can do.
        pass
    return proc.returncode if proc.returncode is not None else -signal.SIGKILL


def wait_with_timeout(
    proc: subprocess.Popen,
    *,
    timeout_s: int,
    poll_s: float = 1.0,
    should_kill: Callable[[], str | None] | None = None,
) -> tuple[int, str | None]:
    """Wait for ``proc`` to exit, polling for an external kill condition.

    Returns ``(returncode, kill_reason)``. If the process exits naturally,
    ``kill_reason`` is None. If ``timeout_s`` is hit OR ``should_kill()``
    returns a non-None string, kills the process gracefully and returns
    that reason.

    Both downstream projects' iter loops have this exact poll-and-kill shape;
    this hides the ``while proc.poll() is None`` boilerplate.

    Parameters
    ----------
    proc
        The ``subprocess.Popen`` instance to wait on.
    timeout_s
        Wall-clock cap. Pass a large number (or ``sys.maxsize``) to disable.
    poll_s
        How often to call ``should_kill()``. Defaults to 1 second.
    should_kill
        Optional callable invoked every ``poll_s`` seconds. Return None to
        keep waiting; return a string to kill the process and use that
        string as the ``kill_reason``. Useful for triage checks (e.g.
        plateau detection from `game_states.jsonl`, KL spike from stdout).
    """
    start = time.monotonic()
    kill_reason: str | None = None

    while proc.poll() is None:
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            kill_reason = f"wall-clock timeout ({timeout_s}s)"
            break
        if should_kill is not None:
            reason = should_kill()
            if reason is not None:
                kill_reason = reason
                break
        time.sleep(poll_s)

    if kill_reason is not None:
        kill_gracefully(proc)
    return proc.returncode if proc.returncode is not None else 0, kill_reason


# ── crash_reason_from_stdout ───────────────────────────────────────────

# (regex, mapper). ``mapper`` is either a literal reason string or a callable
# that takes the regex match and returns one. First match wins, so order from
# specific → generic. Patterns are anchored on the most distinctive token of
# each failure mode rather than the full traceback so they survive ANSI
# colour codes and progress-bar reflows that often pollute the buffer.
_BUILTIN_CRASH_PATTERNS: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    (
        re.compile(r"torch\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError"),
        "CUDA OOM",
    ),
    (
        re.compile(r"^Killed\s*$", re.MULTILINE),
        "killed by host (likely cgroup OOM)",
    ),
    (
        re.compile(r"AssertionError:?\s*(.*)"),
        lambda m: f"AssertionError: {m.group(1).strip()[:80]}",
    ),
    (
        re.compile(r"RuntimeError:?\s*(.*)"),
        lambda m: f"RuntimeError: {m.group(1).strip()[:80]}",
    ),
    (
        re.compile(r"ValueError:?\s*(.*)"),
        lambda m: f"ValueError: {m.group(1).strip()[:80]}",
    ),
    (
        re.compile(r"FileNotFoundError:?\s*(.*)"),
        lambda m: f"FileNotFoundError: {m.group(1).strip()[:80]}",
    ),
    # Generic *Error fallback — runs after named handlers so a more specific
    # mapper wins when both match.
    (
        re.compile(r"^([A-Z][A-Za-z]+Error):?\s*(.*)", re.MULTILINE),
        lambda m: f"{m.group(1)}: {m.group(2).strip()[:80]}",
    ),
]

CrashPattern = tuple[re.Pattern[str], "str | Callable[[re.Match[str]], str]"]


def crash_reason_from_stdout(
    lines: Sequence[str],
    *,
    tail: int = 200,
    extra_patterns: Sequence[CrashPattern] | None = None,
) -> str:
    """Infer a one-line crash reason from a subprocess's stdout/stderr buffer.

    Both downstream sweep loops (gemma4-rlvr and orak-2025-starter-kit) had
    a near-identical helper that scans the tail of a non-zero-exit run for
    structured failure modes (OOM, host SIGKILL, common Python error classes)
    so each iter's row gets a meaningful ``crash_reason`` instead of just
    ``exit_code=-9``.

    Parameters
    ----------
    lines
        Sequence of stdout/stderr lines (newlines optional). Order is preserved.
        A ``deque`` works too — anything supporting negative slicing.
    tail
        How many lines from the end to scan. Defaults to 200 — enough to catch
        a typical Python traceback without re-reading multi-MB log files.
    extra_patterns
        Project-specific ``(regex, mapper)`` pairs to try **before** the
        builtins. ``mapper`` is either a literal reason string or a callable
        that receives the regex match and returns one. Lets a project register
        custom failure modes (eg. wandb 429s, GPU ECC errors, custom exception
        types) without forking the package.

    Returns
    -------
    str
        A short reason string. Always non-empty — falls back to
        ``"unknown: <last non-blank line, truncated to 120 chars>"`` or
        ``"unknown crash"`` if the buffer is empty.

    Examples
    --------
    >>> crash_reason_from_stdout(["...", "torch.OutOfMemoryError: CUDA OOM"])
    'CUDA OOM'
    >>> crash_reason_from_stdout(["RuntimeError: shape mismatch (2, 3) vs (4, 5)"])
    'RuntimeError: shape mismatch (2, 3) vs (4, 5)'
    >>> crash_reason_from_stdout(["everything fine but exit 1"])
    'unknown: everything fine but exit 1'
    >>> crash_reason_from_stdout([])
    'unknown crash'

    Project-specific extension:

    >>> import re
    >>> wandb_429 = (re.compile(r"wandb 429"), "wandb_throttle")
    >>> crash_reason_from_stdout(["wandb 429 retry-after=30"], extra_patterns=[wandb_429])
    'wandb_throttle'
    """
    if not lines:
        return "unknown crash"
    text = "".join(lines[-tail:])
    patterns = list(extra_patterns or []) + _BUILTIN_CRASH_PATTERNS
    for pat, mapper in patterns:
        m = pat.search(text)
        if m:
            return mapper(m) if callable(mapper) else mapper
    last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
    return f"unknown: {last[:120]}" if last else "unknown crash"


__all__ = [
    "CrashPattern",
    "crash_reason_from_stdout",
    "kill_gracefully",
    "wait_with_timeout",
]
