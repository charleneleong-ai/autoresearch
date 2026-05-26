"""Tests for autoresearch.parallel_batch — parallel multi-slot orchestration
with retries, capacity caps, and results.jsonl writing.

Each test uses tiny poll intervals + cheap subprocesses (`true`, `false`,
`sleep 0.05`) so the full suite finishes in well under a second.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from autoresearch.parallel_batch import (
    BatchResult,
    SlotSpec,
    default_launch,
    run_parallel_batch,
)
from autoresearch.results import load_results

# ── helpers ────────────────────────────────────────────────────────────


def _always_complete(score: float = 42.0):
    """Completion check that returns immediately with a fixed score."""

    def check(slot: SlotSpec, run_idx: int) -> dict | None:
        return {"score": score, "extra": {"run_id": f"{slot.label}_r{run_idx}"}}

    return check


def _complete_after_n_polls(n: int, score: float = 42.0):
    """Completion check that returns None for `n` polls then a dict on the (n+1)th."""
    state: dict[tuple[str, int], int] = {}

    def check(slot: SlotSpec, run_idx: int) -> dict | None:
        key = (slot.label, run_idx)
        state[key] = state.get(key, 0) + 1
        if state[key] > n:
            return {"score": score, "extra": {"run_id": f"{slot.label}_r{run_idx}"}}
        return None

    return check


def _launch_true(slot: SlotSpec, run_idx: int) -> subprocess.Popen:
    """Launch fn that runs `true` — exits immediately with code 0."""
    return subprocess.Popen(["true"])


def _launch_false(slot: SlotSpec, run_idx: int) -> subprocess.Popen:
    """Launch fn that runs `false` — exits immediately with code 1."""
    return subprocess.Popen(["false"])


# ── SlotSpec ───────────────────────────────────────────────────────────


def test_slot_spec_basic() -> None:
    spec = SlotSpec(
        label="test_slot",
        tag="test_tag",
        command=["echo", "hello"],
        n_runs=3,
    )
    assert spec.label == "test_slot"
    assert spec.tag == "test_tag"
    assert spec.n_runs == 3
    assert spec.game is None


def test_slot_spec_with_game_and_extra() -> None:
    spec = SlotSpec(
        label="mario_baseline",
        tag="my_sweep",
        command=["python", "run.py"],
        n_runs=2,
        game="super_mario",
        extra={"side": "baseline"},
    )
    assert spec.game == "super_mario"
    assert spec.extra == {"side": "baseline"}


# ── run_parallel_batch — happy path ────────────────────────────────────


def test_run_parallel_batch_writes_all_results(tmp_path: Path) -> None:
    """All n_runs of a successful slot get logged to results.jsonl."""
    slot = SlotSpec(
        label="happy",
        tag="happy_tag",
        command=["true"],
        n_runs=3,
        completion_check=_always_complete(score=42.0),
    )
    result = run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=4,
        max_attempts_per_run=1,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=_launch_true,
    )

    assert isinstance(result, BatchResult)
    assert len(result.completed) == 3
    assert result.gave_up == []

    rows = load_results(experiments_dir=tmp_path, tag="happy_tag")
    assert len(rows) == 3
    assert all(r["score"] == 42.0 for r in rows)
    assert all(r["status"] == "KEEP" for r in rows)


def test_run_parallel_batch_status_override_from_completion(tmp_path: Path) -> None:
    """``completion_check`` can return a ``status`` key that overrides the
    default KEEP — lets callers tag specific rows (e.g. BASELINE) when needed."""
    counter = {"i": 0}

    def alternating_status(slot: SlotSpec, run_idx: int) -> dict:
        counter["i"] += 1
        return {
            "score": 1.0,
            # First completion is BASELINE, rest are KEEP
            "status": "BASELINE" if run_idx == 0 else "KEEP",
        }

    slot = SlotSpec(
        label="mario",
        tag="my_sweep",
        command=["true"],
        n_runs=3,
        game="super_mario",
        completion_check=alternating_status,
    )
    run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=1,  # serialize so order is deterministic
        max_attempts_per_run=1,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=_launch_true,
    )

    rows = load_results(experiments_dir=tmp_path, tag="my_sweep")
    assert len(rows) == 3
    statuses = [r["status"] for r in rows]
    assert statuses.count("BASELINE") == 1
    assert statuses.count("KEEP") == 2


def test_run_parallel_batch_multiple_slots(tmp_path: Path) -> None:
    """Multiple slots with different tags each get their own results.jsonl."""
    slots = [
        SlotSpec(
            label="a",
            tag="tag_a",
            command=["true"],
            n_runs=2,
            completion_check=_always_complete(score=10.0),
        ),
        SlotSpec(
            label="b",
            tag="tag_b",
            command=["true"],
            n_runs=2,
            completion_check=_always_complete(score=20.0),
        ),
    ]
    run_parallel_batch(
        slots,
        experiments_dir=tmp_path,
        max_parallel=4,
        max_attempts_per_run=1,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=_launch_true,
    )

    rows_a = load_results(experiments_dir=tmp_path, tag="tag_a")
    rows_b = load_results(experiments_dir=tmp_path, tag="tag_b")
    assert len(rows_a) == 2 and all(r["score"] == 10.0 for r in rows_a)
    assert len(rows_b) == 2 and all(r["score"] == 20.0 for r in rows_b)


# ── max_parallel ───────────────────────────────────────────────────────


def test_run_parallel_batch_respects_max_parallel(tmp_path: Path) -> None:
    """No more than max_parallel processes are in-flight at once."""
    peak_concurrent = 0

    def slow_launch(slot: SlotSpec, run_idx: int) -> subprocess.Popen:
        return subprocess.Popen(["sleep", "0.15"])

    completion_state: dict[tuple[str, int], float] = {}

    def slow_complete(slot: SlotSpec, run_idx: int) -> dict | None:
        # Only return complete once the subprocess would have finished
        key = (slot.label, run_idx)
        if key not in completion_state:
            completion_state[key] = time.monotonic()
            return None
        if time.monotonic() - completion_state[key] < 0.15:
            return None
        return {"score": 1.0}

    slot = SlotSpec(
        label="capped",
        tag="capped_tag",
        command=["sleep", "0.15"],
        n_runs=6,
        completion_check=slow_complete,
    )

    # We use a wrapper launch_fn to observe concurrency
    in_flight = [0]

    def observed_launch(slot: SlotSpec, run_idx: int) -> subprocess.Popen:
        in_flight[0] += 1
        nonlocal peak_concurrent
        peak_concurrent = max(peak_concurrent, in_flight[0])
        proc = subprocess.Popen(["sleep", "0.15"])
        return proc

    # Hook into completion to decrement
    completion_seen: set[tuple[str, int]] = set()

    def observed_complete(slot: SlotSpec, run_idx: int) -> dict | None:
        result = slow_complete(slot, run_idx)
        key = (slot.label, run_idx)
        if result is not None and key not in completion_seen:
            completion_seen.add(key)
            in_flight[0] -= 1
        return result

    slot = SlotSpec(
        label="capped",
        tag="capped_tag",
        command=["sleep", "0.15"],
        n_runs=6,
        completion_check=observed_complete,
    )

    run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=2,
        max_attempts_per_run=1,
        poll_interval_s=0.02,
        launch_stagger_s=0,
        launch_fn=observed_launch,
    )

    assert peak_concurrent <= 2, f"peak was {peak_concurrent}, expected <= 2"
    rows = load_results(experiments_dir=tmp_path, tag="capped_tag")
    assert len(rows) == 6


# ── retry on failure ───────────────────────────────────────────────────


def test_run_parallel_batch_retries_failed_run(tmp_path: Path) -> None:
    """A failed run is retried up to max_attempts_per_run."""
    attempt_count: dict[tuple[str, int], int] = {}

    def flaky_launch(slot: SlotSpec, run_idx: int) -> subprocess.Popen:
        key = (slot.label, run_idx)
        attempt_count[key] = attempt_count.get(key, 0) + 1
        # First attempt fails, second succeeds
        cmd = ["false"] if attempt_count[key] == 1 else ["true"]
        return subprocess.Popen(cmd)

    def complete_on_success(slot: SlotSpec, run_idx: int) -> dict | None:
        # Only return complete after the 2nd attempt
        key = (slot.label, run_idx)
        if attempt_count.get(key, 0) >= 2:
            return {"score": 99.0}
        return None

    slot = SlotSpec(
        label="flaky",
        tag="flaky_tag",
        command=["true"],
        n_runs=1,
        completion_check=complete_on_success,
    )

    result = run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=1,
        max_attempts_per_run=3,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=flaky_launch,
    )

    assert ("flaky", 0) in result.completed
    assert attempt_count[("flaky", 0)] == 2
    rows = load_results(experiments_dir=tmp_path, tag="flaky_tag")
    assert len(rows) == 1
    assert rows[0]["score"] == 99.0


def test_run_parallel_batch_gives_up_after_max_attempts(tmp_path: Path) -> None:
    """Persistently failing runs end up in gave_up after max_attempts."""
    slot = SlotSpec(
        label="cursed",
        tag="cursed_tag",
        command=["false"],
        n_runs=2,
        completion_check=lambda s, i: None,  # never succeeds
    )

    result = run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=2,
        max_attempts_per_run=2,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=_launch_false,
    )

    assert len(result.gave_up) == 2
    assert ("cursed", 0) in result.gave_up
    assert ("cursed", 1) in result.gave_up
    assert result.completed == {}


# ── idempotency ────────────────────────────────────────────────────────


def test_run_parallel_batch_skips_existing_results(tmp_path: Path) -> None:
    """Re-running with the same slot skips runs that already have a row.

    Identity is by `extra.run_id` if the completion provides one;
    otherwise by (tag, game, run_idx).
    """
    slot = SlotSpec(
        label="skippable",
        tag="skip_tag",
        command=["true"],
        n_runs=3,
        completion_check=_always_complete(),
    )

    # First run — writes 3 rows
    run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=4,
        max_attempts_per_run=1,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=_launch_true,
    )
    assert len(load_results(experiments_dir=tmp_path, tag="skip_tag")) == 3

    # Second run — should write 0 new rows
    launch_called: list[int] = []

    def counting_launch(slot: SlotSpec, run_idx: int) -> subprocess.Popen:
        launch_called.append(run_idx)
        return subprocess.Popen(["true"])

    run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=4,
        max_attempts_per_run=1,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=counting_launch,
    )
    # All 3 runs already exist; nothing should be launched
    assert launch_called == []
    assert len(load_results(experiments_dir=tmp_path, tag="skip_tag")) == 3


# ── callbacks ──────────────────────────────────────────────────────────


def test_on_completion_callback_fires(tmp_path: Path) -> None:
    """on_completion fires once per successful (slot, run_idx)."""
    seen: list[tuple[str, int, float]] = []

    def on_complete(slot: SlotSpec, run_idx: int, data: dict) -> None:
        seen.append((slot.label, run_idx, data["score"]))

    slot = SlotSpec(
        label="cb_slot",
        tag="cb_tag",
        command=["true"],
        n_runs=2,
        completion_check=_always_complete(score=7.0),
    )
    run_parallel_batch(
        [slot],
        experiments_dir=tmp_path,
        max_parallel=2,
        max_attempts_per_run=1,
        poll_interval_s=0.01,
        launch_stagger_s=0,
        launch_fn=_launch_true,
        on_completion=on_complete,
    )

    assert sorted(seen) == [("cb_slot", 0, 7.0), ("cb_slot", 1, 7.0)]


# ── default_launch ─────────────────────────────────────────────────────


def test_default_launch_writes_logs(tmp_path: Path) -> None:
    """default_launch redirects stdout to a per-run log file."""
    log_dir = tmp_path / "logs"
    slot = SlotSpec(
        label="logged",
        tag="logged_tag",
        command=["sh", "-c", "echo hello-from-stdout; echo woops >&2"],
        n_runs=1,
        log_dir=log_dir,
    )
    proc = default_launch(slot, run_idx=0)
    proc.wait(timeout=5)

    log_files = list(log_dir.glob("logged_*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text()
    assert "hello-from-stdout" in content
    assert "woops" in content  # stderr merged into stdout


def test_default_launch_handles_callable_command(tmp_path: Path) -> None:
    """command can be a callable receiving run_idx → list[str]."""
    log_dir = tmp_path / "logs"
    slot = SlotSpec(
        label="dynamic",
        tag="dyn_tag",
        command=lambda run_idx: ["sh", "-c", f"echo run-was-{run_idx}"],
        n_runs=1,
        log_dir=log_dir,
    )
    proc = default_launch(slot, run_idx=4)
    proc.wait(timeout=5)

    log_files = list(log_dir.glob("dynamic_*.log"))
    assert len(log_files) == 1
    assert "run-was-4" in log_files[0].read_text()
