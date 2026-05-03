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
"""

from __future__ import annotations

import signal
import subprocess
import time
from collections.abc import Callable


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


__all__ = ["kill_gracefully", "wait_with_timeout"]
