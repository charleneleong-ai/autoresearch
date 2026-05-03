"""Post-iter retrospective — automated failure-mode detection for sweep iterations.

After each sweep iteration, run a panel of `FailureDetector` callables against
the iter's standard artefacts (`results.jsonl` row, the per-iter log, optional
`*.per_row.jsonl` rubric dump). Each detector returns an optional `Finding`;
the collected findings are attached to the iter's `results.jsonl` row under a
``retrospective`` key and rendered into a sibling ``retrospective_<iter>.md``.

Wiring into a sweep loop (see autoresearch#16 for the design discussion):

    from autoresearch.retrospective import (
        BUILTIN_DETECTORS,
        attach_findings_to_row,
        audit_iter,
        format_markdown,
    )

    findings = audit_iter(
        results_row=row,
        log_path=Path("logs/sweep.log"),
        per_row_jsonl_path=Path("experiments/<tag>/per_row_E27.jsonl"),
        history=load_results(...),  # all rows for this tag, for plateau detectors
        detectors=[BUILTIN_DETECTORS[n] for n in cfg.detectors],
    )
    attach_findings_to_row(row, findings)
    (out_dir / f"retrospective_E{row['experiment']}.md").write_text(
        format_markdown(findings, iter_id=row["experiment"])
    )

After-the-fact CLI:

    autoresearch-retrospective audit \\
        --results-jsonl experiments/<tag>/<config>/results.jsonl \\
        --log logs/sweep.log \\
        --iter latest

Severity ladder (`info` < `warn` < `block`) drives the per-project loop's
on-finding action — typically `warn` → append to next iter's notes (so the
context propagates), `block` → stop the sweep early (don't burn compute on a
known-broken setup).
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import typer
import yaml
from rich import print as rprint

from autoresearch.results import get_score

Severity = Literal["info", "warn", "block"]
SEVERITY_ORDER: dict[Severity, int] = {"info": 0, "warn": 1, "block": 2}

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ── data shapes ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """One pattern observed by a FailureDetector. Attached to results.jsonl."""

    detector: str
    severity: Severity
    summary: str  # one-line, written into results.jsonl
    detail: str  # markdown for retrospective.md, with concrete row indices / log refs
    suggested_action: str  # "fix rubric", "tighten prompt", "tune hparam X", etc

    def at_least(self, level: Severity) -> bool:
        return SEVERITY_ORDER[self.severity] >= SEVERITY_ORDER[level]


@dataclass
class IterContext:
    """Everything a detector needs to inspect a single iter."""

    results_row: dict[str, Any]
    log_path: Path | None = None
    per_row_jsonl_path: Path | None = None
    # Prior rows for the same (tag, game), oldest first. Plateau / streak
    # detectors need this; single-iter detectors can ignore it.
    history: Sequence[dict[str, Any]] = field(default_factory=list)
    # Free-form per-detector overrides (e.g. plateau epsilon). Detectors look up
    # their own name in this dict for tunable params.
    detector_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


class FailureDetector(Protocol):
    """A callable that inspects an IterContext and returns 0 or 1 Finding.

    Detectors should be cheap (no network, no GPU) — the loop calls them
    synchronously after each iter. Returning ``None`` means "no pattern
    observed"; returning a Finding means "this pattern fired, here's what +
    why".
    """

    name: str

    def __call__(self, ctx: IterContext) -> Finding | None: ...


# ── helpers shared by built-in detectors ───────────────────────────────


def _read_log_lines(log_path: Path | None) -> list[str]:
    if log_path is None or not log_path.exists():
        return []
    try:
        return log_path.read_text(errors="replace").splitlines()
    except OSError:
        return []


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _detector_kwargs(ctx: IterContext, detector_name: str) -> dict[str, Any]:
    return ctx.detector_kwargs.get(detector_name, {})


# ── built-in detectors ────────────────────────────────────────────────


def _silent_kill(ctx: IterContext) -> Finding | None:
    """``status`` indicates a kill, but the log shows no traceback → hang vs crash ambiguity."""
    status = str(ctx.results_row.get("status", "")).upper()
    kill_markers = ("EARLY_KILL", "TIMEOUT", "TRIAGE_KILL")
    if not any(m in status for m in kill_markers):
        return None
    lines = _read_log_lines(ctx.log_path)
    has_traceback = any("Traceback" in ln or "Error:" in ln for ln in lines)
    if has_traceback:
        return None
    iter_id = ctx.results_row.get("experiment", "?")
    return Finding(
        detector=_silent_kill.name,
        severity="warn",
        summary=(
            f"E{iter_id} killed with status={status} but no traceback in log "
            f"— hang vs crash unclear."
        ),
        detail=(
            f"### silent_kill (warn)\n\n"
            f"Iter `E{iter_id}` exited with `status={status}` but the log "
            f"({ctx.log_path}) contains no `Traceback` or `Error:` lines.\n\n"
            f"Likely a hang (deadlocked subprocess, OOM partial-recovery, "
            f"stuck network call) rather than a crash. Add traceback capture "
            f"to the subprocess wrapper (`signal.SIGINT` then `traceback.print_stack` "
            f"on a watchdog thread) so the next occurrence diagnoses itself."
        ),
        suggested_action=(
            "Add traceback capture to the subprocess; instrument the suspected hang point."
        ),
    )


_silent_kill.name = "silent_kill"  # type: ignore[attr-defined]


def _triage_threshold_mismatch(ctx: IterContext) -> Finding | None:
    """Triage threshold fired before the task's first realistic scoring step.

    Detected by: status == EARLY_KILL with reason mentioning `score plateau`,
    AND the iter's step count is below the detector's `min_first_score_step`
    (default 100). That combination means the kill is firing *before* the task
    has had a chance to reach its first scoring opportunity.
    """
    status = str(ctx.results_row.get("status", "")).upper()
    if "KILL" not in status and "TIMEOUT" not in status:
        return None

    lines = _read_log_lines(ctx.log_path)
    plateau_kill_re = re.compile(r"score plateau .*\b(\d+)\b\s*steps")
    plateau_match = next(
        (m for line in lines for m in [plateau_kill_re.search(line)] if m),
        None,
    )
    if plateau_match is None:
        return None

    kwargs = _detector_kwargs(ctx, _triage_threshold_mismatch.name)
    min_first_score = int(kwargs.get("min_first_score_step", 100))
    steps = int(ctx.results_row.get("steps", 0) or 0)
    plateau_steps = int(plateau_match.group(1))

    if steps >= min_first_score:
        return None

    iter_id = ctx.results_row.get("experiment", "?")
    return Finding(
        detector=_triage_threshold_mismatch.name,
        severity="warn",
        summary=(
            f"E{iter_id} triage-killed at step {steps} "
            f"(< {min_first_score} expected first-score step). "
            f"Plateau threshold {plateau_steps} fires before this task can score."
        ),
        detail=(
            f"### triage_threshold_mismatch (warn)\n\n"
            f"Iter `E{iter_id}` was killed by `score plateau` triage at step "
            f"{steps}, but this task typically takes ≥ {min_first_score} steps "
            f"to reach its first scoring event. The plateau threshold of "
            f"{plateau_steps} is firing before the task has a fair chance.\n\n"
            f"Bump the per-task threshold:\n\n"
            f"```python\n"
            f"TRIAGE_SCORE_PLATEAU_STEPS_PER_TASK = {{'<task>': {plateau_steps * 2 + 50}}}\n"
            f"```"
        ),
        suggested_action=(
            f"Raise TRIAGE_SCORE_PLATEAU_STEPS for this task to ≥ {plateau_steps * 2 + 50}."
        ),
    )


_triage_threshold_mismatch.name = "triage_threshold_mismatch"  # type: ignore[attr-defined]


def _eval_score_plateau(ctx: IterContext) -> Finding | None:
    """KEEP iter, but heldout score within ε of last K iters → param sweep is doing nothing.

    Detector params (under ``detector_kwargs["eval_score_plateau"]``):

    * ``window`` (int, default 5) — how many recent iters (excluding the
      current one) define the plateau baseline.
    * ``epsilon`` (float, default 0.5) — how close the current score must be
      to the plateau median, in the same units as ``score``, to fire.
    """
    if not ctx.history:
        return None
    kwargs = _detector_kwargs(ctx, _eval_score_plateau.name)
    window = int(kwargs.get("window", 5))
    epsilon = float(kwargs.get("epsilon", 0.5))

    if len(ctx.history) < window:
        return None

    recent = ctx.history[-window:]
    recent_scores = [get_score(r) for r in recent]
    cur_score = get_score(ctx.results_row)
    plateau_score = statistics.median(recent_scores)

    spread = max(recent_scores) - min(recent_scores)
    if spread > epsilon:
        return None
    if abs(cur_score - plateau_score) > epsilon:
        return None

    iter_id = ctx.results_row.get("experiment", "?")
    return Finding(
        detector=_eval_score_plateau.name,
        severity="warn",
        summary=(
            f"E{iter_id} score {cur_score:.2f} within ±{epsilon} of the last {window} iters "
            f"(plateau ≈ {plateau_score:.2f}). Param sweep is not moving the metric."
        ),
        detail=(
            f"### eval_score_plateau (warn)\n\n"
            f"Across the last {window} iters and the current one, score has "
            f"sat at {plateau_score:.2f} ± {epsilon}. The autoresearch parameter "
            f"proposer is firing (see `notes` field for the deltas) but the "
            f"swept axis isn't budging the metric.\n\n"
            f"Recent scores: {', '.join(f'{s:.2f}' for s in recent_scores + [cur_score])}\n\n"
            f"Likely the bottleneck is *outside* the swept hyperparameter — "
            f"capability limit, prompt design, or reward shaping. Stop sweeping "
            f"the same axis and inspect those layers."
        ),
        suggested_action=(
            "Stop sweeping the current hparam axis; inspect prompt / reward / "
            "capability layers for the actual bottleneck."
        ),
    )


_eval_score_plateau.name = "eval_score_plateau"  # type: ignore[attr-defined]


def _bucketed_failure(ctx: IterContext) -> Finding | None:
    """Heldout failures concentrate in a single ground-truth bucket → rubric mismatch.

    Detector params (under ``detector_kwargs["bucketed_failure"]``):

    * ``bucket_field`` (str, default ``"ground_truth_bucket"``) — JSONL field
      to group failures by. Common alternatives: ``"label"``, ``"class"``,
      ``"category"``, ``"map_name"``, ``"trigger"``.
    * ``passed_field`` (str, default ``"passed"``) — boolean field marking
      pass/fail rows.
    * ``concentration_threshold`` (float, default 0.7) — fire when the top
      bucket holds at least this fraction of all failures.
    * ``min_failures`` (int, default 5) — require at least this many failure
      rows before firing (avoid noise on tiny eval sets).
    """
    rows = _read_jsonl(ctx.per_row_jsonl_path)
    if not rows:
        return None

    kwargs = _detector_kwargs(ctx, _bucketed_failure.name)
    bucket_field = str(kwargs.get("bucket_field", "ground_truth_bucket"))
    passed_field = str(kwargs.get("passed_field", "passed"))
    threshold = float(kwargs.get("concentration_threshold", 0.7))
    min_failures = int(kwargs.get("min_failures", 5))

    fails = [r for r in rows if r.get(passed_field) is False]
    if len(fails) < min_failures:
        return None

    bucket_counts: dict[str, int] = {}
    fail_indices_by_bucket: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        if r.get(passed_field) is not False:
            continue
        b = r.get(bucket_field)
        if b is None:
            continue
        key = str(b)
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
        fail_indices_by_bucket.setdefault(key, []).append(idx)

    if not bucket_counts:
        return None

    top_bucket, top_count = max(bucket_counts.items(), key=lambda kv: kv[1])
    fraction = top_count / len(fails)
    if fraction < threshold:
        return None

    iter_id = ctx.results_row.get("experiment", "?")
    sample = fail_indices_by_bucket[top_bucket][:8]
    sample_str = ", ".join(f"i={i}" for i in sample)
    return Finding(
        detector=_bucketed_failure.name,
        severity="warn",
        summary=(
            f"E{iter_id}: {fraction:.1%} of failures concentrate in {bucket_field}={top_bucket!r} "
            f"({top_count}/{len(fails)}). Rubric or prompt likely mismatched on this bucket."
        ),
        detail=(
            f"### bucketed_failure (warn)\n\n"
            f"{fraction:.1%} of failure rows ({top_count}/{len(fails)}) share "
            f"`{bucket_field}={top_bucket!r}`. Sample fail rows: {sample_str} …\n\n"
            f"All other buckets account for ≤ {1 - fraction:.1%} combined "
            f"({len(bucket_counts) - 1} other buckets observed). When failures "
            f"cluster this tightly, the rubric or prompt is usually missing a "
            f"case for this specific bucket — extend allowed-facts, add a "
            f"few-shot example, or relax the strict-match constraint for it.\n\n"
            f"Top bucket counts: "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(bucket_counts.items(), key=lambda kv: -kv[1])[:5]
            )
        ),
        suggested_action=(
            f"Inspect rows {sample_str} and extend the rubric / prompt for "
            f"`{bucket_field}={top_bucket}`."
        ),
    )


_bucketed_failure.name = "bucketed_failure"  # type: ignore[attr-defined]


def _gradient_collapse(ctx: IterContext) -> Finding | None:
    """RL/RLVR sweep: train/loss → 0 AND train/reward flat → optimizer stuck.

    Reads the iter's wandb history for ``loss_key`` and ``reward_key``, then
    fires when both:

    * ``mean(loss[-window:]) < loss_near_zero_threshold`` (loss collapsed), AND
    * ``stddev(reward[-window:]) / |mean(reward[-window:])| < flat_cv_threshold``
      (reward isn't moving — gradient signal is dead)

    Silently skips when:

    * ``wandb_url`` isn't in the row (project doesn't use wandb),
    * the ``[wandb]`` extra isn't installed (lazy ImportError → None),
    * either series is missing or has fewer than ``window`` samples,
    * the wandb API call fails (logged via the exception, but not a finding).

    Detector params (under ``detector_kwargs["gradient_collapse"]``):

    * ``loss_key`` (str, default ``"train/loss"``)
    * ``reward_key`` (str, default ``"train/reward"``)
    * ``window`` (int, default 50) — how many recent samples define "recent"
    * ``loss_near_zero_threshold`` (float, default 0.05)
    * ``flat_cv_threshold`` (float, default 0.02) — coefficient-of-variation
      below which the reward series counts as "flat"
    * ``samples`` (int, default 500) — passed through to wandb history sub-sampling
    """
    wandb_url = ctx.results_row.get("wandb_url")
    if not wandb_url:
        return None

    try:
        from autoresearch.wandb_history import fetch_history
    except ImportError:
        return None  # [wandb] extra not installed — no-op gracefully

    kwargs = _detector_kwargs(ctx, _gradient_collapse.name)
    loss_key = str(kwargs.get("loss_key", "train/loss"))
    reward_key = str(kwargs.get("reward_key", "train/reward"))
    window = int(kwargs.get("window", 50))
    loss_zero = float(kwargs.get("loss_near_zero_threshold", 0.05))
    flat_cv = float(kwargs.get("flat_cv_threshold", 0.02))
    samples = int(kwargs.get("samples", 500))

    try:
        series = fetch_history(run_url=str(wandb_url), keys=[loss_key, reward_key], samples=samples)
    except Exception:
        # Bad URL, API failure, missing credentials (wandb.errors.UsageError),
        # network timeout — anything fetch_history might raise. The detector's
        # contract is "silently skip if I can't run", not "crash the sweep".
        return None

    loss = series.get(loss_key, [])
    reward = series.get(reward_key, [])
    if len(loss) < window or len(reward) < window:
        return None

    recent_loss = loss[-window:]
    recent_reward = reward[-window:]
    mean_loss = sum(recent_loss) / window
    mean_reward = sum(recent_reward) / window

    if mean_loss >= loss_zero:
        return None

    if abs(mean_reward) < 1e-9:
        # Reward effectively zero throughout — CV is undefined but the symptom
        # (no signal) is exactly what we're catching. Treat as "flat".
        flat = True
        cv = float("inf")
    else:
        var = sum((r - mean_reward) ** 2 for r in recent_reward) / window
        std = var**0.5
        cv = std / abs(mean_reward)
        flat = cv < flat_cv

    if not flat:
        return None

    iter_id = ctx.results_row.get("experiment", "?")
    return Finding(
        detector=_gradient_collapse.name,
        severity="block",
        summary=(
            f"E{iter_id}: {loss_key} mean={mean_loss:.4f} (< {loss_zero}) and "
            f"{reward_key} CV={cv:.4f} (< {flat_cv}) over last {window} steps "
            f"— optimizer appears collapsed."
        ),
        detail=(
            f"### gradient_collapse (block)\n\n"
            f"Iter `E{iter_id}` shows the joint pattern that indicates a stuck "
            f"optimizer:\n\n"
            f"* `{loss_key}` mean over last {window} samples = **{mean_loss:.6f}** "
            f"(threshold < {loss_zero})\n"
            f"* `{reward_key}` mean = **{mean_reward:.6f}**, "
            f"stddev/|mean| = **{cv:.6f}** (threshold < {flat_cv})\n\n"
            f"Loss has collapsed near zero while reward isn't moving — "
            f"gradients are no longer driving learning. Check (in order): "
            f"learning-rate schedule (collapsed too far?), gradient clipping "
            f"(over-aggressive?), KL coefficient (β too high?), reward scale "
            f"(numerical underflow?). Sweep should stop until the optimizer "
            f"is unstuck."
        ),
        suggested_action=(
            "Inspect LR / grad-clip / KL coefficient / reward scale; the "
            "optimizer is no longer learning. Recommend stopping the sweep."
        ),
    )


_gradient_collapse.name = "gradient_collapse"  # type: ignore[attr-defined]


# ── value_transform_mismatch (and its sign_flip_in_rubric alias) ──────
#
# Generic "rubric strict-equals on a transformed value" detector. Catches
# the family of rubric mismatches where the model presents a value through
# a transform (sign flip, unit scale, sign negation, etc.) that the rubric
# refuses to invert. The original sign_flip_in_rubric case from #19 is one
# instance: cited == abs(truth) — model wrote magnitude, truth was signed.
#
# Users can extend BUILTIN_TRANSFORMS by passing their own callables in the
# `transforms` list (works in-process; for YAML schedule blocks, register
# a name in BUILTIN_TRANSFORMS first).

# Map of name → callable. YAML schedule entries reference these by name.
BUILTIN_TRANSFORMS: dict[str, Callable[[float], float]] = {
    "abs": abs,
    "negate": lambda x: -x,
    # `scale_100` is for rubrics that store decimals (0.12) but the model writes
    # percent (12). `scale_0.01` is the inverse — for rubrics that store
    # percents but the model writes decimals.
    "scale_100": lambda x: x * 100,
    "scale_0.01": lambda x: x * 0.01,
}


def _build_value_transform_detector(
    detector_name: str,
    default_transforms: list[str] | None,
) -> FailureDetector:
    """Factory for the value_transform_mismatch family.

    `value_transform_mismatch` is the generic form: the user must pass
    `transforms=[...]` in detector_kwargs or the detector is a no-op.

    `sign_flip_in_rubric` is the same logic with `default_transforms=["abs"]`,
    matching the spec from autoresearch#19. Calling them by either name
    produces a Finding tagged with that name (so the YAML schedule and
    results.jsonl entries stay self-documenting).
    """

    def fn(ctx: IterContext) -> Finding | None:
        rows = _read_jsonl(ctx.per_row_jsonl_path)
        if not rows:
            return None

        kwargs = _detector_kwargs(ctx, detector_name)
        cited_field = str(kwargs.get("cited_value_field", "cited_value"))
        truth_field = str(kwargs.get("ground_truth_value_field", "ground_truth_value"))
        passed_field = str(kwargs.get("passed_field", "passed"))
        min_pairs = int(kwargs.get("min_value_pairs", 10))
        threshold = float(kwargs.get("mismatch_threshold", 0.5))
        epsilon = float(kwargs.get("epsilon", 0.001))
        transforms_kw = kwargs.get("transforms", default_transforms)
        if not transforms_kw:
            return None  # generic detector with no transforms is a no-op

        # Resolve names → callables. Accept callables directly for in-process
        # use; YAML schedules pass strings.
        resolved: list[tuple[str, Callable[[float], float]]] = []
        for t in transforms_kw:
            if isinstance(t, str):
                if t not in BUILTIN_TRANSFORMS:
                    raise KeyError(
                        f"Unknown transform {t!r}. Built-in: {sorted(BUILTIN_TRANSFORMS)}. "
                        f"Pass a callable directly to use a custom transform."
                    )
                resolved.append((t, BUILTIN_TRANSFORMS[t]))
            elif callable(t):
                resolved.append((getattr(t, "__name__", "<custom>"), t))
            else:
                raise TypeError(
                    f"transforms entries must be str (registered name) or callable, "
                    f"got {type(t).__name__}"
                )

        # Extract (cited, truth) pairs from FAIL rows where both values are
        # numeric. Booleans pass `isinstance(_, int)` so explicitly exclude.
        pairs: list[tuple[float, float]] = []
        sample_indices: list[int] = []
        for idx, row in enumerate(rows):
            if row.get(passed_field) is not False:
                continue
            cited = row.get(cited_field)
            truth = row.get(truth_field)
            if isinstance(cited, bool) or isinstance(truth, bool):
                continue
            if not isinstance(cited, (int, float)) or not isinstance(truth, (int, float)):
                continue
            pairs.append((float(cited), float(truth)))
            sample_indices.append(idx)

        if len(pairs) < min_pairs:
            return None

        def matches(cited: float, transformed: float) -> bool:
            # Relative tolerance with a 1e-9 absolute floor for values near zero.
            return abs(cited - transformed) <= max(abs(transformed) * epsilon, 1e-9)

        # Per-transform match counts.
        per_transform: list[tuple[str, int, list[int]]] = []
        for tname, tfn in resolved:
            count = 0
            matched_indices: list[int] = []
            for (cited, truth), idx in zip(pairs, sample_indices, strict=False):
                try:
                    transformed = tfn(truth)
                except Exception:
                    continue
                if matches(cited, transformed):
                    count += 1
                    matched_indices.append(idx)
            per_transform.append((tname, count, matched_indices))

        best_name, best_count, best_indices = max(per_transform, key=lambda r: r[1])
        fraction = best_count / len(pairs)
        if fraction < threshold:
            return None

        iter_id = ctx.results_row.get("experiment", "?")
        sample = best_indices[:8]
        all_transforms_md = "".join(
            f"* `{t}`: {c}/{len(pairs)} ({c / len(pairs):.1%})\n"
            for t, c, _ in sorted(per_transform, key=lambda r: -r[1])
        )
        return Finding(
            detector=detector_name,
            severity="warn",
            summary=(
                f"E{iter_id}: {fraction:.1%} of failures ({best_count}/{len(pairs)}) "
                f"match {best_name}({truth_field}) — rubric should accept the "
                f"{best_name} variant."
            ),
            detail=(
                f"### {detector_name} (warn)\n\n"
                f"Across {len(pairs)} fail rows where both `{cited_field}` and "
                f"`{truth_field}` are numeric:\n\n"
                f"* **{best_name} match rate: {best_count}/{len(pairs)} "
                f"({fraction:.1%})** (threshold ≥ {threshold:.0%})\n\n"
                f"All transforms tried:\n"
                f"{all_transforms_md}\n"
                f"Sample matched rows: {', '.join(f'i={i}' for i in sample)} …\n\n"
                f"The rubric is currently strict-equal on `{truth_field}` but the "
                f"model is presenting the value through a `{best_name}` transform. "
                f"Either relax the rubric to accept the transformed form, or fix "
                f"the prompt to produce raw values."
            ),
            suggested_action=(
                f"Extend rubric to accept `{best_name}({truth_field})` as a valid "
                f"match for `{cited_field}`."
            ),
        )

    fn.name = detector_name  # type: ignore[attr-defined]
    return fn


_value_transform_mismatch = _build_value_transform_detector(
    "value_transform_mismatch", default_transforms=None
)
_sign_flip_in_rubric = _build_value_transform_detector(
    "sign_flip_in_rubric", default_transforms=["abs"]
)


# Public registry — users can add their own (e.g. `obs_collision` for game-
# agent sweeps with state-jsonl obs ambiguity).
BUILTIN_DETECTORS: dict[str, FailureDetector] = {
    _silent_kill.name: _silent_kill,
    _triage_threshold_mismatch.name: _triage_threshold_mismatch,
    _eval_score_plateau.name: _eval_score_plateau,
    _bucketed_failure.name: _bucketed_failure,
    _gradient_collapse.name: _gradient_collapse,
    _value_transform_mismatch.name: _value_transform_mismatch,
    _sign_flip_in_rubric.name: _sign_flip_in_rubric,
}


# ── orchestration ─────────────────────────────────────────────────────


def audit_iter(
    *,
    results_row: dict[str, Any],
    log_path: Path | None = None,
    per_row_jsonl_path: Path | None = None,
    history: Sequence[dict[str, Any]] = (),
    detectors: Sequence[FailureDetector] | None = None,
    detector_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[Finding]:
    """Run all detectors against one iter's artefacts; return findings.

    ``detectors`` defaults to all four builtins. Pass a subset (or extend with
    custom callables) to wire a project-specific panel. ``detector_kwargs`` is
    forwarded into ``IterContext.detector_kwargs`` so per-detector tuning
    (epsilon, bucket field, etc.) lives in one place.
    """
    if detectors is None:
        detectors = list(BUILTIN_DETECTORS.values())
    ctx = IterContext(
        results_row=results_row,
        log_path=log_path,
        per_row_jsonl_path=per_row_jsonl_path,
        history=list(history),
        detector_kwargs=detector_kwargs or {},
    )
    findings: list[Finding] = []
    for det in detectors:
        finding = det(ctx)
        if finding is not None:
            findings.append(finding)
    return findings


def attach_findings_to_row(row: dict[str, Any], findings: Sequence[Finding]) -> dict[str, Any]:
    """Mutate `row` in-place to add a ``retrospective`` key with finding summaries.

    Stores only `summary` + the structured action / severity, not the full
    `detail` markdown — that lives in the sibling `retrospective_<iter>.md`
    so the JSONL row stays scannable.
    """
    row["retrospective"] = {
        "findings": [
            {
                "detector": f.detector,
                "severity": f.severity,
                "summary": f.summary,
                "suggested_action": f.suggested_action,
            }
            for f in findings
        ]
    }
    return row


def format_markdown(findings: Sequence[Finding], iter_id: int | str | None = None) -> str:
    """Render findings as the body of a `retrospective_<iter>.md` file."""
    if not findings:
        if iter_id is not None:
            return f"## E{iter_id} retrospective\n\nNo findings.\n"
        return "No findings.\n"
    if iter_id is not None:
        header = f"## E{iter_id} retrospective\n\n"
    else:
        header = "## Retrospective\n\n"
    return header + "\n\n".join(f.detail for f in findings) + "\n"


def filter_by_severity(
    findings: Sequence[Finding], min_severity: Severity = "warn"
) -> list[Finding]:
    """Subset of findings at or above ``min_severity``. Useful for the loop's
    ``on_finding`` action: ``warn`` → append to next iter notes, ``block`` →
    stop sweep."""
    return [f for f in findings if f.at_least(min_severity)]


# ── YAML config loading (per autoresearch#16 schedule integration) ────


@dataclass(frozen=True)
class RetrospectiveSpec:
    """Parsed `post_iter_retrospective:` block from a sweep's schedule YAML."""

    enabled: bool
    detectors: list[str]
    detector_kwargs: dict[str, dict[str, Any]]
    on_finding: list[dict[str, Any]]

    def selected_detectors(
        self, registry: dict[str, FailureDetector] | None = None
    ) -> list[FailureDetector]:
        """Resolve detector names against a registry (defaults to BUILTIN_DETECTORS)."""
        reg = registry if registry is not None else BUILTIN_DETECTORS
        out: list[FailureDetector] = []
        for name in self.detectors:
            if name not in reg:
                raise KeyError(
                    f"Unknown detector {name!r}. Registered: {sorted(reg)}. "
                    f"To add a custom detector, pass a registry that includes it."
                )
            out.append(reg[name])
        return out

    def action_for(self, severity: Severity) -> str | None:
        """Return the action string for the highest-severity rule matching `severity`,
        or None if no rule applies."""
        best: tuple[int, str] | None = None
        for rule in self.on_finding:
            rule_sev = rule.get("severity")
            if rule_sev not in SEVERITY_ORDER:
                continue
            if SEVERITY_ORDER[rule_sev] > SEVERITY_ORDER[severity]:
                continue
            rank = SEVERITY_ORDER[rule_sev]
            if best is None or rank > best[0]:
                best = (rank, str(rule.get("action", "")))
        return best[1] if best else None


def load_spec(path: str | Path) -> RetrospectiveSpec:
    """Parse a YAML file containing a top-level ``post_iter_retrospective`` block."""
    raw = yaml.safe_load(Path(path).read_text())
    block = raw.get("post_iter_retrospective", raw)  # accept block or root
    return RetrospectiveSpec(
        enabled=bool(block.get("enabled", True)),
        detectors=list(block.get("detectors", list(BUILTIN_DETECTORS))),
        detector_kwargs=dict(block.get("detector_kwargs", {})),
        on_finding=list(block.get("on_finding", [])),
    )


# ── CLI ───────────────────────────────────────────────────────────────


@app.command()
def audit(
    results_jsonl: Path = typer.Option(
        ...,
        "--results-jsonl",
        help="Path to the sweep's results.jsonl. The selected iter's row is inspected.",
    ),
    log_path: Path | None = typer.Option(
        None, "--log", help="Path to the per-iter log file (for silent_kill, triage detectors)."
    ),
    per_row_jsonl_path: Path | None = typer.Option(
        None,
        "--per-row-jsonl",
        help="Path to the iter's per-row rubric dump (for bucketed_failure / sign_flip detectors).",
    ),
    iter_id: str = typer.Option(
        "latest",
        "--iter",
        help="Iter index to audit, or 'latest' for the last row in the JSONL.",
    ),
    spec: Path | None = typer.Option(
        None,
        "--spec",
        help="Optional YAML spec with detector list + per-detector kwargs (autoresearch#16).",
    ),
    write_md: Path | None = typer.Option(
        None,
        "--write-md",
        help="If set, write the formatted retrospective markdown to this path.",
    ),
    write_json: bool = typer.Option(
        False,
        "--write-json",
        help="If set, append findings to the audited row in results.jsonl (in place).",
    ),
) -> None:
    """Run detectors against one iter's artefacts and print findings."""
    if not results_jsonl.exists():
        rprint(f"[red]results.jsonl not found:[/red] {results_jsonl}")
        raise typer.Exit(2)

    rows = [json.loads(ln) for ln in results_jsonl.read_text().splitlines() if ln.strip()]
    if not rows:
        rprint(f"[yellow]No rows in[/yellow] {results_jsonl}")
        raise typer.Exit(0)

    if iter_id == "latest":
        target_idx = len(rows) - 1
    else:
        try:
            target = int(iter_id)
        except ValueError:
            rprint(f"[red]--iter must be 'latest' or an integer, got[/red] {iter_id!r}")
            raise typer.Exit(2) from None
        matches = [i for i, r in enumerate(rows) if r.get("experiment") == target]
        if not matches:
            rprint(f"[red]No row with experiment={target}[/red]")
            raise typer.Exit(2)
        target_idx = matches[-1]

    target_row = rows[target_idx]
    history = rows[:target_idx]

    if spec is not None:
        spec_obj = load_spec(spec)
        detectors = spec_obj.selected_detectors()
        detector_kwargs = spec_obj.detector_kwargs
    else:
        detectors = list(BUILTIN_DETECTORS.values())
        detector_kwargs = {}

    findings = audit_iter(
        results_row=target_row,
        log_path=log_path,
        per_row_jsonl_path=per_row_jsonl_path,
        history=history,
        detectors=detectors,
        detector_kwargs=detector_kwargs,
    )

    iter_marker = target_row.get("experiment", target_idx)
    if not findings:
        rprint(f"[green]E{iter_marker}: no findings[/green]")
    else:
        rprint(f"[bold]E{iter_marker}: {len(findings)} finding(s)[/bold]")
        for f in findings:
            colour = {"info": "cyan", "warn": "yellow", "block": "red"}[f.severity]
            rprint(f"  [{colour}]{f.severity.upper()}[/{colour}] [{f.detector}] {f.summary}")
            rprint(f"    → {f.suggested_action}")

    if write_md is not None:
        write_md.parent.mkdir(parents=True, exist_ok=True)
        write_md.write_text(format_markdown(findings, iter_id=iter_marker))
        rprint(f"[dim]wrote retrospective markdown:[/dim] {write_md}")

    if write_json:
        attach_findings_to_row(target_row, findings)
        rows[target_idx] = target_row
        results_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        rprint(f"[dim]updated row E{iter_marker} in:[/dim] {results_jsonl}")


@app.command()
def list_detectors() -> None:
    """Print the names of all built-in detectors."""
    for name in BUILTIN_DETECTORS:
        rprint(name)


__all__ = [
    "BUILTIN_DETECTORS",
    "Finding",
    "FailureDetector",
    "IterContext",
    "RetrospectiveSpec",
    "Severity",
    "SEVERITY_ORDER",
    "app",
    "attach_findings_to_row",
    "audit_iter",
    "filter_by_severity",
    "format_markdown",
    "load_spec",
]
