"""Parallel multi-slot batch orchestrator.

Launches a fixed set of *slots* in parallel, where each slot runs the same
command ``n_runs`` times (each run is a replicate). Retries failed runs up
to ``max_attempts_per_run``, caps in-flight subprocesses at ``max_parallel``,
and writes each successful completion to a per-tag ``results.jsonl`` via
``autoresearch.results.log_experiment``.

Use cases
---------
- **Reproducibility studies** — one slot per condition, n_runs=K replicates.
- **Ablation grids** — N slots × K seeds, each slot a different config.
- **Cross-condition matrices** — slots span the cartesian product of conditions
  (e.g. baseline-vs-detector × N games), each with K replicates.

The orchestrator is *project-agnostic*. The caller provides:

- ``command`` — what to launch (list, or callable ``run_idx -> list``)
- ``completion_check`` — callable returning a dict (with ``score`` etc.) when
  the run has produced its output, ``None`` while still in-flight
- (optionally) ``launch_fn`` — full control over subprocess spawning, e.g. to
  detach via ``setsid nohup`` for daemon-mode runs

The orchestrator polls ``completion_check`` every ``poll_interval_s``. A run
is considered failed when its subprocess exits non-zero AND
``completion_check`` returns ``None`` (i.e. the process died before producing
its output file).

Idempotency: before launching, the orchestrator inspects existing rows under
``experiments/<tag>/results.jsonl`` and skips runs whose ``(slot.label,
run_idx)`` already have a row. Batch identity is stored in the row's
``_batch_slot_label`` / ``_batch_run_idx`` fields so it's robust against
completion data overriding ``run_id`` with the rollout's own ID.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .results import load_results, log_experiment

CompletionCheck = Callable[["SlotSpec", int], dict[str, Any] | None]
LaunchFn = Callable[["SlotSpec", int], subprocess.Popen]
OnCompletion = Callable[["SlotSpec", int, dict[str, Any]], None]


@dataclass(frozen=True)
class SlotSpec:
    """A single batch slot — runs ``command`` ``n_runs`` times in parallel.

    Parameters
    ----------
    label
        Human-readable slot name (used for logs, log-file naming).
    tag
        Per-tag bucket for ``results.jsonl`` — rows for this slot land at
        ``experiments/<tag>/results.jsonl``.
    command
        Either a static ``list[str]`` (same command for every replicate) or
        a callable ``(run_idx) -> list[str]`` to vary per replicate (e.g.
        injecting a seed or run-id into the args).
    n_runs
        How many replicates to run.
    game
        Optional ``game`` field for the results row (lets one tag carry rows
        for multiple games, e.g. cross-game lift studies).
    cwd
        Working directory for the subprocess. Default: inherit caller's cwd.
    env
        Environment mapping passed to subprocess. Default: inherit os.environ.
    log_dir
        Where stdout/stderr land (one log per replicate). Default: cwd.
    completion_check
        Callable ``(slot, run_idx) -> dict | None``. Return ``None`` while the
        replicate is still in-flight; return a dict with at minimum ``score``
        when complete. Optional keys: ``game_score``, ``steps``, ``runtime_min``,
        ``wandb_url``, ``description``, ``extra``. The dict is forwarded directly
        to ``log_experiment``.
    extra
        Static extra fields merged into every row's ``extra``. Per-run keys
        from ``completion_check``'s ``extra`` take precedence.
    """

    label: str
    tag: str
    command: list[str] | Callable[[int], list[str]]
    n_runs: int = 3
    game: str | None = None
    cwd: Path | None = None
    env: dict[str, str] | None = None
    log_dir: Path | None = None
    completion_check: CompletionCheck | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    """Outcome of ``run_parallel_batch``.

    Attributes
    ----------
    slots
        The original slots passed in.
    completed
        Mapping ``(slot_label, run_idx) -> completion_data`` for every
        successful run.
    gave_up
        ``(slot_label, run_idx)`` tuples that hit ``max_attempts_per_run``
        without ever completing.
    attempt_counts
        Mapping ``(slot_label, run_idx) -> int`` showing how many attempts
        were made.
    elapsed_s
        Total wall-clock time.
    """

    slots: list[SlotSpec]
    completed: dict[tuple[str, int], dict[str, Any]]
    gave_up: list[tuple[str, int]]
    attempt_counts: dict[tuple[str, int], int]
    elapsed_s: float


def default_launch(slot: SlotSpec, run_idx: int) -> subprocess.Popen:
    """Default launch_fn — subprocess.Popen with stdout+stderr → per-run log.

    Log file: ``<slot.log_dir or cwd>/<label>_run<idx>_<ts>.log``. The file
    is opened in append mode so a retried run appends to the same file as
    earlier attempts (keeps the full history per replicate together).
    """
    cmd = slot.command(run_idx) if callable(slot.command) else slot.command
    log_dir = slot.log_dir or (slot.cwd or Path.cwd())
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = log_dir / f"{slot.label}_run{run_idx}_{ts}.log"
    log_file = log_path.open("ab")
    return subprocess.Popen(
        cmd,
        cwd=slot.cwd,
        env=slot.env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def _existing_batch_keys(experiments_dir: Path | str, tag: str) -> set[tuple[str, int]]:
    """``(slot_label, run_idx)`` tuples already represented in the tag's
    ``results.jsonl``. Used for idempotency.

    This is the batch's own concept of identity — independent of whatever
    ``run_id`` the completion data provided (which is typically the
    rollout's own natural ID, e.g. a wandb run id or output directory).
    """
    rows = load_results(experiments_dir=experiments_dir, tag=tag)
    return {
        (r["_batch_slot_label"], r["_batch_run_idx"])
        for r in rows
        if r.get("_batch_slot_label") is not None and r.get("_batch_run_idx") is not None
    }


def run_parallel_batch(
    slots: list[SlotSpec],
    *,
    experiments_dir: Path | str,
    max_parallel: int = 8,
    max_attempts_per_run: int = 3,
    poll_interval_s: float = 60.0,
    launch_stagger_s: float = 10.0,
    launch_fn: LaunchFn | None = None,
    on_completion: OnCompletion | None = None,
    write_results: bool = True,
    log: Callable[[str], None] | None = None,
) -> BatchResult:
    """Orchestrate a parallel multi-slot batch.

    Each slot's ``n_runs`` replicates are queued; up to ``max_parallel`` are
    in-flight at any time. A run is considered:

    - **complete** when ``slot.completion_check(slot, run_idx)`` returns a dict;
      the dict is forwarded to ``log_experiment`` (status defaults to ``KEEP``
      unless the completion data provides an explicit ``status``).
    - **failed** when its subprocess exits non-zero AND ``completion_check``
      still returns ``None``. The replicate is requeued (up to
      ``max_attempts_per_run`` total tries).
    - **gave_up** when the attempt count exceeds ``max_attempts_per_run``.

    Idempotency: at startup, replicates whose ``(slot.label, run_idx)`` already
    have a row in ``experiments/<tag>/results.jsonl`` are skipped. The orchestrator
    stores its own batch identity in each row's ``_batch_slot_label`` /
    ``_batch_run_idx`` fields, independent of whatever ``run_id`` the completion
    provided (which is the rollout's natural ID — wandb run, output dir name, etc.).

    Parameters
    ----------
    slots
        The batch's slots.
    experiments_dir
        Root for per-tag ``results.jsonl`` files.
    max_parallel
        Hard cap on in-flight subprocesses across the whole batch.
    max_attempts_per_run
        Total attempts (including the first) before a replicate is given up on.
    poll_interval_s
        Sleep between poll cycles. Reduce for tests; default 60s for real use.
    launch_stagger_s
        Sleep between successive launches *within a single poll cycle* — gives
        each subprocess a head-start on resource acquisition before the next
        joins.
    launch_fn
        Override the subprocess-spawning strategy. Default: ``default_launch``.
        Useful for daemon-mode (setsid nohup), Docker exec, K8s pods, etc.
    on_completion
        Optional callback invoked once per successful replicate, after the
        row is written. Signature: ``(slot, run_idx, completion_data)``.
    write_results
        If False, skip writing to ``results.jsonl``. Useful when the caller
        wants to handle persistence themselves via ``on_completion``.
    log
        Optional logging callback. Default: silent.

    Returns
    -------
    BatchResult
        Detailed outcome — completed/gave_up/attempt_counts/elapsed_s.
    """
    launch_fn = launch_fn or default_launch
    log = log or (lambda _msg: None)
    start = time.monotonic()

    pending: list[tuple[str, int]] = []
    in_flight: dict[tuple[str, int], subprocess.Popen] = {}
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    gave_up: list[tuple[str, int]] = []
    attempt_counts: dict[tuple[str, int], int] = {}

    slots_by_label = {s.label: s for s in slots}

    for slot in slots:
        existing = _existing_batch_keys(experiments_dir, slot.tag) if write_results else set()
        for run_idx in range(slot.n_runs):
            if (slot.label, run_idx) in existing:
                log(f"skip {slot.label}/r{run_idx} — already in results.jsonl")
                continue
            pending.append((slot.label, run_idx))
            attempt_counts[(slot.label, run_idx)] = 0

    log(
        f"parallel_batch start: {len(pending)} pending, "
        f"max_parallel={max_parallel}, max_attempts={max_attempts_per_run}"
    )

    while pending or in_flight:
        for key in list(in_flight.keys()):
            slot = slots_by_label[key[0]]
            proc = in_flight[key]

            completion = (
                slot.completion_check(slot, key[1]) if slot.completion_check is not None else None
            )
            if completion is not None:
                _on_complete(
                    slot,
                    key[1],
                    completion,
                    experiments_dir=experiments_dir,
                    write_results=write_results,
                    on_completion=on_completion,
                )
                completed[key] = completion
                del in_flight[key]
                log(f"complete {key[0]}/r{key[1]} score={completion.get('score')}")
                continue

            if proc.poll() is not None and proc.returncode != 0:
                del in_flight[key]
                if attempt_counts[key] < max_attempts_per_run:
                    log(
                        f"failed {key[0]}/r{key[1]} (rc={proc.returncode}), "
                        f"requeue ({attempt_counts[key]}/{max_attempts_per_run})"
                    )
                    pending.append(key)
                else:
                    log(f"gave up {key[0]}/r{key[1]} after {attempt_counts[key]} attempts")
                    gave_up.append(key)

        while len(in_flight) < max_parallel and pending:
            key = pending.pop(0)
            slot = slots_by_label[key[0]]
            attempt_counts[key] += 1
            proc = launch_fn(slot, key[1])
            in_flight[key] = proc
            log(f"launch {key[0]}/r{key[1]} attempt {attempt_counts[key]}/{max_attempts_per_run}")
            if launch_stagger_s > 0:
                time.sleep(launch_stagger_s)

        if in_flight:
            time.sleep(poll_interval_s)

    elapsed = time.monotonic() - start
    log(
        f"parallel_batch done: completed={len(completed)} "
        f"gave_up={len(gave_up)} elapsed={elapsed:.1f}s"
    )

    return BatchResult(
        slots=slots,
        completed=completed,
        gave_up=gave_up,
        attempt_counts=attempt_counts,
        elapsed_s=elapsed,
    )


def _derive_run_id(slot: SlotSpec, run_idx: int) -> str:
    """If the slot's extra carries a run_id, use that; else derive ``<label>_r<idx>``."""
    if "run_id" in slot.extra:
        return f"{slot.extra['run_id']}_r{run_idx}"
    return f"{slot.label}_r{run_idx}"


def _on_complete(
    slot: SlotSpec,
    run_idx: int,
    completion: dict[str, Any],
    *,
    experiments_dir: Path | str,
    write_results: bool,
    on_completion: OnCompletion | None,
) -> None:
    """Write the results.jsonl row + fire on_completion. Default status KEEP;
    completion can override via a ``status`` key."""
    if write_results:
        run_extra = {**slot.extra, **(completion.get("extra") or {})}
        run_extra.setdefault("run_id", _derive_run_id(slot, run_idx))
        # Batch's own (slot, run_idx) identity — used for idempotent rerun.
        # Independent of `run_id` (which is the rollout's natural ID and may
        # come from the completion data, e.g. a wandb run id or output dir).
        run_extra["_batch_slot_label"] = slot.label
        run_extra["_batch_run_idx"] = run_idx
        log_experiment(
            experiments_dir=experiments_dir,
            tag=slot.tag,
            game=slot.game,
            score=completion.get("score", 0.0),
            game_score=completion.get("game_score", 0.0),
            steps=completion.get("steps", 0),
            runtime_min=completion.get("runtime_min", 0.0),
            status=completion.get("status", "KEEP"),
            description=completion.get("description", f"{slot.label} r{run_idx}"),
            wandb_url=completion.get("wandb_url", ""),
            notes=completion.get("notes", ""),
            extra=run_extra,
        )
    if on_completion is not None:
        on_completion(slot, run_idx, completion)


__all__ = [
    "BatchResult",
    "CompletionCheck",
    "LaunchFn",
    "OnCompletion",
    "SlotSpec",
    "default_launch",
    "run_parallel_batch",
]
