"""In-package sweep runner — the reusable orchestration loop.

Replaces the ~600–1300 LOC copy-pasted iter-loop in per-project
``experiments/autoresearch.py`` files with a composable runner backed by
three Protocols (``IterPlanner``, ``TriageMonitor``, ``ResultExtractor``).

See autoresearch#20 and ``docs/sweep_runner_design.md`` for the design
discussion.

Minimal per-project glue (~30 lines)::

    runner = SweepRunner(
        tag="my_sweep",
        planner=MyPlanner(...),
        triage=MyTriage(...),
        extractor=MyExtractor(...),
        retrospective_spec=load_retrospective_spec("configs/schedules/my_sweep.yaml"),
    )
    result = runner.run()
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from autoresearch.current_run import sidecar
from autoresearch.results import (
    get_score,
    load_results,
    log_experiment,
    relabel_last_as_early_kill,
)
from autoresearch.retrospective import (
    BUILTIN_DETECTORS,
    Finding,
    RetrospectiveSpec,
    audit_iter,
    filter_by_severity,
    format_markdown,
)
from autoresearch.subprocess_utils import wait_with_timeout

logger = logging.getLogger(__name__)


# ── data shapes ────────────────────────────────────────────────────────


@dataclass
class IterPlan:
    """What the planner wants the runner to execute for one iteration.

    Only ``cmd`` and ``description`` are required.  Everything else has
    sensible defaults or falls back to ``SweepRunner``-level settings.
    """

    cmd: list[str]
    description: str
    config_name: str | None = None
    notes: str = ""
    timeout_min: int | None = None
    env: dict[str, str] | None = None
    cwd: str | Path | None = None


@dataclass
class IterOutcome:
    """Post-mortem of a single iteration."""

    plan: IterPlan
    run_id: str | None
    exit_code: int
    kill_reason: str | None
    rows: list[dict[str, Any]]
    findings: list[Finding]
    elapsed_s: float = 0.0


@dataclass
class SweepResult:
    """Summary returned by ``SweepRunner.run()``."""

    tag: str
    iterations: int
    kills: int
    blocked: bool
    outcomes: list[IterOutcome] = field(default_factory=list)


# ── Protocols ──────────────────────────────────────────────────────────


class IterPlanner(Protocol):
    """Yields one ``IterPlan`` per iteration.

    The generator stops when the sweep is done — all planned iters
    exhausted, convergence reached, or any other project-specific
    stopping criterion.

    ``history`` is the list of *already-logged* result rows for the tag,
    as returned by ``load_results()``.  A schedule-driven planner may
    ignore it; a feedback-driven planner reads it to decide what to
    propose next.
    """

    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]: ...


class TriageMonitor(Protocol):
    """Polled while a subprocess runs.  Returns a kill-reason on trigger,
    ``None`` to keep waiting.

    Lifecycle per iteration: ``setup()`` → ``check()`` × N → ``teardown()``.
    """

    def setup(
        self,
        plan: IterPlan,
        proc: subprocess.Popen[bytes],
        baseline: float,
    ) -> str | None:
        """Called right after ``Popen``.

        Use this to latch onto whatever signal channel the project needs
        (file path for game-state polling, ``proc.stdout`` for line-grep,
        etc.).  Return a ``run_id`` if discoverable, else ``None``.
        """
        ...

    def check(self, elapsed_s: float) -> str | None:
        """Return a kill-reason string to abort, or ``None`` to keep going."""
        ...

    def teardown(self) -> None:
        """Cleanup after the subprocess exits (regardless of how)."""
        ...


class ResultExtractor(Protocol):
    """Turns a finished (or killed) subprocess into result-row dicts
    suitable for ``log_experiment``.

    Return one dict per logical result (e.g. one per game in orak, one
    per iter in gemma4-rlvr).  Recognised keys::

        score, steps, status, description, notes, game,
        config_name, wandb_url, game_score, runtime_min

    Any unrecognised keys are passed through as ``extra``.
    """

    def extract(
        self,
        plan: IterPlan,
        run_id: str | None,
        exit_code: int,
    ) -> list[dict[str, Any]]: ...


# ── internal helpers ───────────────────────────────────────────────────

_LOG_FIELDS: frozenset[str] = frozenset({
    "config_name",
    "game",
    "score",
    "steps",
    "status",
    "description",
    "wandb_url",
    "notes",
    "game_score",
    "runtime_min",
})

# Keys that SweepRunner controls — silently dropped from extractor output
# so they can't conflict with the runner's own values.
_RESERVED_FIELDS: frozenset[str] = frozenset({
    "tag",
    "experiments_dir",
    "experiment",
    "timestamp",
    "tags",
    "extra",
})


# ── SweepRunner ───────────────────────────────────────────────────────


class SweepRunner:
    """The main loop — plan → launch → monitor → log → retrospective → repeat."""

    def __init__(
        self,
        *,
        tag: str,
        planner: IterPlanner,
        triage: TriageMonitor,
        extractor: ResultExtractor,
        retrospective_spec: RetrospectiveSpec | None = None,
        experiments_dir: str | Path = "experiments",
        iter_timeout_min: int = 30,
        triage_poll_s: int = 5,
        pause_between_iters_s: int = 15,
        sigint_grace_s: int = 60,
        sigterm_grace_s: int = 30,
    ) -> None:
        self.tag = tag
        self.planner = planner
        self.triage = triage
        self.extractor = extractor
        self.retrospective_spec = retrospective_spec
        self.experiments_dir = Path(experiments_dir)
        self.iter_timeout_min = iter_timeout_min
        self.triage_poll_s = triage_poll_s
        self.pause_between_iters_s = pause_between_iters_s
        self.sigint_grace_s = sigint_grace_s
        self.sigterm_grace_s = sigterm_grace_s

    # ── public ────────────────────────────────────────────────────────

    def run(self) -> SweepResult:
        """Execute the sweep loop.  Returns a :class:`SweepResult`."""
        history = load_results(self.experiments_dir, self.tag)
        best_score = max((get_score(r) for r in history), default=0.0)
        outcomes: list[IterOutcome] = []
        kills = 0
        blocked = False

        for plan in self.planner.plan_iters(history):
            iter_num = len(outcomes) + 1
            logger.info("Iter %d: %s", iter_num, plan.description)

            outcome = self._run_iter(plan, best_score)
            outcomes.append(outcome)

            if outcome.kill_reason is not None:
                kills += 1

            # Update best score from newly logged rows.
            for row in outcome.rows:
                best_score = max(best_score, get_score(row))

            # Retrospective — check for blockers.
            blockers = filter_by_severity(outcome.findings, "block")
            if blockers:
                logger.error(
                    "Blocked by retrospective: %s",
                    "; ".join(f.summary for f in blockers),
                )
                blocked = True
                break

            # Refresh history for the planner's next yield.
            history = load_results(self.experiments_dir, self.tag)

            # Pause between iters (GPU memory release, etc.).
            if self.pause_between_iters_s > 0:
                logger.debug(
                    "Pausing %ds before next iter",
                    self.pause_between_iters_s,
                )
                time.sleep(self.pause_between_iters_s)

        return SweepResult(
            tag=self.tag,
            iterations=len(outcomes),
            kills=kills,
            blocked=blocked,
            outcomes=outcomes,
        )

    # ── private ───────────────────────────────────────────────────────

    def _run_iter(self, plan: IterPlan, best_score: float) -> IterOutcome:
        """Run a single iteration: launch → monitor → log → retrospective."""
        timeout_s = (plan.timeout_min or self.iter_timeout_min) * 60
        config_rows = load_results(
            self.experiments_dir, self.tag, plan.config_name
        )

        sidecar_payload = {
            "experiment": len(config_rows),
            "config_name": plan.config_name or "",
            "description": plan.description,
            "notes": plan.notes,
            "started_at": datetime.now(tz=UTC).isoformat(),
        }

        with sidecar(
            sidecar_payload,
            tag=self.tag,
            config_name=plan.config_name,
            experiments_dir=self.experiments_dir,
        ):
            popen_env = {**os.environ, **plan.env} if plan.env else None
            proc = subprocess.Popen(
                plan.cmd,
                env=popen_env,
                cwd=plan.cwd,
            )
            run_id = self.triage.setup(plan, proc, best_score)

            launch_time = time.monotonic()
            returncode, kill_reason = wait_with_timeout(
                proc,
                timeout_s=timeout_s,
                poll_s=self.triage_poll_s,
                should_kill=lambda: self.triage.check(
                    time.monotonic() - launch_time
                ),
            )
            elapsed_s = time.monotonic() - launch_time
            self.triage.teardown()

        exit_code = returncode if returncode is not None else -1

        # ── extract and log results ───────────────────────────────────
        rows = self.extractor.extract(plan, run_id, exit_code)
        for row in rows:
            self._log_row(row, plan)

        # ── relabel if killed ─────────────────────────────────────────
        if kill_reason is not None:
            logger.warning("Iter killed: %s", kill_reason)
            relabel_last_as_early_kill(
                experiments_dir=self.experiments_dir,
                tag=self.tag,
                config_name=plan.config_name,
                kill_reason=kill_reason,
                last_n=max(len(rows), 1),
            )

        # ── retrospective ─────────────────────────────────────────────
        findings = self._run_retrospective(plan, len(rows))

        return IterOutcome(
            plan=plan,
            run_id=run_id,
            exit_code=exit_code,
            kill_reason=kill_reason,
            rows=rows,
            findings=findings,
            elapsed_s=elapsed_s,
        )

    def _log_row(self, row: dict[str, Any], plan: IterPlan) -> None:
        """Map an extractor row-dict to :func:`log_experiment` kwargs."""
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for k, v in row.items():
            if k in _RESERVED_FIELDS:
                continue
            if k in _LOG_FIELDS:
                kwargs[k] = v
            else:
                extra[k] = v

        # Accept evaluation_score as an alias for score.
        if "score" not in kwargs and "evaluation_score" in extra:
            kwargs["score"] = extra.pop("evaluation_score")

        # Defaults from the plan.
        kwargs.setdefault("config_name", plan.config_name)
        kwargs.setdefault("description", plan.description)
        kwargs.setdefault("notes", plan.notes)
        kwargs.setdefault("status", "KEEP")

        log_experiment(
            experiments_dir=self.experiments_dir,
            tag=self.tag,
            extra=extra or None,
            **kwargs,
        )

    def _run_retrospective(
        self, plan: IterPlan, n_rows: int
    ) -> list[Finding]:
        """Run detectors on the just-logged rows.  Returns all findings."""
        spec = self.retrospective_spec
        if spec is None or not spec.enabled or n_rows == 0:
            return []

        detectors = [
            BUILTIN_DETECTORS[n]
            for n in spec.detectors
            if n in BUILTIN_DETECTORS
        ]
        if not detectors:
            return []

        # Re-read history (includes the just-logged rows).
        history = load_results(
            self.experiments_dir, self.tag, plan.config_name
        )
        logged_rows = history[-n_rows:]

        all_findings: list[Finding] = []
        out_dir = self.experiments_dir / self.tag
        if plan.config_name:
            out_dir = out_dir / plan.config_name

        for row in logged_rows:
            row_findings = audit_iter(
                results_row=row,
                history=history,
                detectors=detectors,
                detector_kwargs=spec.detector_kwargs or {},
            )
            all_findings.extend(row_findings)

            # Write per-iter retrospective markdown alongside results.jsonl.
            if row_findings:
                iter_id = row.get("experiment", "?")
                md_path = out_dir / f"retrospective_E{iter_id}.md"
                md_path.write_text(
                    format_markdown(row_findings, iter_id=iter_id)
                )

        return all_findings


__all__ = [
    "IterOutcome",
    "IterPlan",
    "IterPlanner",
    "ResultExtractor",
    "SweepResult",
    "SweepRunner",
    "TriageMonitor",
]
