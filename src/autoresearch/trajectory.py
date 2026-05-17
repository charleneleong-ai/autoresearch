"""Trajectory writer — ShareGPT-shaped per-episode log for agentic-RL pipelines.

Lifted from `orak-2025-starter-kit/agents/_harness/trajectory.py` (the
keystone artifact for trajectory→SFT fine-tuning workflows). Two properties
matter for downstream use:

1. **Per-episode rollups** — steps are buffered in memory, then flushed at
   ``flush_episode`` time into one of two files:

   * ``trajectory_samples.jsonl`` — episodes that completed without any
     fallback action (clean for SFT)
   * ``failed_trajectories.jsonl`` — episodes with any fallback or crash
     (kept separately so they can be inspected / curated / used as
     negatives, but not bled into the SFT split)

2. **ShareGPT-shaped steps** — each step renders as a ``conversations``
   array of ``{from, value}`` turns (system / human / gpt), so the
   ``trajectory_samples.jsonl`` is directly loadable as an SFT dataset
   without a reformatting pass. ``<REASONING_SCRATCHPAD>`` tags are
   normalised to ``<think>`` so the data plays nicely with thinking-mode
   datasets.

Usage::

    from autoresearch import StepRecord, TrajectoryWriter

    writer = TrajectoryWriter(log_dir, model="gemma-2-2b")
    for step in episode_steps:
        writer.add_step(StepRecord(
            step=step.idx,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            assistant_output=raw_output,
            action=parsed_action,
            tokens_prompt=usage.prompt_tokens,
            tokens_completion=usage.completion_tokens,
        ))
    writer.flush_episode(episode_id, completed=True, final_score=score, game_name="mario")
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Replace ``<REASONING_SCRATCHPAD>`` tags with ``<think>`` for SFT compat."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace(
        "</REASONING_SCRATCHPAD>", "</think>"
    )


def has_incomplete_scratchpad(content: str) -> bool:
    """Return True if a scratchpad opened but never closed (truncated output)."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def format_recent_history(records: list[StepRecord]) -> str:
    """Render a compact outcome-tagged history block from recent step records.

    Format::

        step 12: action=move_north  score=1 (+0)  state=changed
        step 13: action=move_north  score=1 (+0)  state=unchanged (loop?)
        step 14: action=move_east   score=1 (+0)  state=changed

    Two signals — ``score_delta`` against the previous step and
    ``state=changed/unchanged`` (via ``obs_digest`` equality) — give the
    planner enough evidence for anti-loop / continue-what's-working
    heuristics to fire.  Returns an empty string when ``records`` is empty.
    """
    if not records:
        return ""
    lines: list[str] = []
    for i, r in enumerate(records):
        score = r.info_score
        if score is None:
            score_str = "score=?"
        else:
            prev_score = records[i - 1].info_score if i > 0 else None
            if prev_score is None:
                score_str = f"score={_fmt_score(score)}"
            else:
                delta = score - prev_score
                sign = "+" if delta >= 0 else ""
                score_str = f"score={_fmt_score(score)} ({sign}{_fmt_score(delta)})"

        if r.obs_digest is None:
            state_str = "state=?"
        elif i == 0:
            state_str = "state=initial"
        elif r.obs_digest == records[i - 1].obs_digest:
            state_str = "state=unchanged (loop?)"
        else:
            state_str = "state=changed"

        action = (r.action or "").strip().splitlines()[0][:60] if r.action else "?"
        lines.append(f"step {r.step}: action={action}  {score_str}  {state_str}")
    return "\n".join(lines)


def _fmt_score(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


@dataclass
class StepRecord:
    """One agent-environment step. Renders to ShareGPT via :meth:`to_sharegpt`.

    Fields added for outcome tagging (used by :func:`format_recent_history`
    and long-horizon planners):

    * ``info_score`` — game score at this step; ``None`` if unavailable.
    * ``obs_digest`` — short hash of the observation string; used for
      loop detection (repeated ``obs_digest`` → ``state=unchanged``).
    """

    step: int
    system_prompt: str | None
    user_prompt: str
    assistant_output: str
    action: str
    reasoning: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    cached_tokens: int = 0
    is_fallback: bool = False
    fallback_reason: str | None = None
    info_score: float | None = None
    obs_digest: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_sharegpt(self) -> dict[str, Any]:
        convs: list[dict[str, str]] = []
        if self.system_prompt:
            convs.append({"from": "system", "value": self.system_prompt})
        convs.append({"from": "human", "value": self.user_prompt})
        convs.append(
            {
                "from": "gpt",
                "value": convert_scratchpad_to_think(self.assistant_output),
            }
        )
        return {
            "step": self.step,
            "action": self.action,
            "reasoning": self.reasoning,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "tokens": {
                "prompt": self.tokens_prompt,
                "completion": self.tokens_completion,
                "total": self.tokens_total,
                "cached": self.cached_tokens,
            },
            "conversations": convs,
            "timestamp": self.timestamp,
        }


class TrajectoryWriter:
    """Per-episode buffer of :class:`StepRecord` s, flushed at episode end.

    Each ``flush_episode`` writes one JSONL line containing the rolled-up
    episode (steps + token totals + fallback count + final score) to either
    the success or failed file, then clears the buffer for the next episode.

    Coexists with any per-step writer the agent already has — this only emits
    the rolled-up episode files.
    """

    def __init__(self, log_dir: str | Path, *, model: str = "unknown") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.success_path = self.log_dir / "trajectory_samples.jsonl"
        self.failed_path = self.log_dir / "failed_trajectories.jsonl"
        self._buffer: list[StepRecord] = []

    def add_step(self, record: StepRecord) -> None:
        self._buffer.append(record)

    def recent(self, k: int) -> list[StepRecord]:
        """Return the last ``k`` step records from the live in-episode buffer.

        Used by long-horizon agents (e.g. a subtask planner) to feed
        outcome-tagged history into the next plan call without waiting for
        episode flush.  Returns an empty list when ``k <= 0``.
        """
        if k <= 0:
            return []
        return list(self._buffer[-k:])

    def flush_episode(
        self,
        episode_id: int,
        *,
        completed: bool,
        final_score: float,
        game_name: str,
    ) -> Path:
        """Write the buffered episode to success/failed file. Returns the path."""
        any_fallback = any(r.is_fallback for r in self._buffer)
        is_success = completed and not any_fallback
        target = self.success_path if is_success else self.failed_path

        entry = {
            "episode_id": episode_id,
            "game_name": game_name,
            "model": self.model,
            "completed": completed,
            "final_score": final_score,
            "n_steps": len(self._buffer),
            "n_fallbacks": sum(1 for r in self._buffer if r.is_fallback),
            "total_cached_tokens": sum(r.cached_tokens for r in self._buffer),
            "total_input_tokens": sum(r.tokens_prompt for r in self._buffer),
            "total_output_tokens": sum(r.tokens_completion for r in self._buffer),
            "timestamp": datetime.now(UTC).isoformat(),
            "steps": [r.to_sharegpt() for r in self._buffer],
        }
        try:
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(
                "trajectory[%s ep=%s] flushed → %s (success=%s, steps=%d, fallbacks=%d)",
                game_name,
                episode_id,
                target.name,
                is_success,
                len(self._buffer),
                entry["n_fallbacks"],
            )
        except OSError as e:
            logger.warning("failed to write episode trajectory: %s", e)

        self._buffer.clear()
        return target


# ── Post-hoc trajectory introspection ────────────────────────────────────


@dataclass
class MilestoneSpec:
    """Predicate that fires the first time a game_states.jsonl row crosses a milestone."""

    name: str
    predicate: Callable[[dict[str, Any]], bool]


@dataclass
class DwellSpec:
    """Predicate for counting how many steps the agent spent in a named zone."""

    name: str
    predicate: Callable[[dict[str, Any]], bool]


@dataclass
class ActionSpec:
    """Extracts a hashable target from an action row for perseveration counting."""

    extract_target: Callable[[dict[str, Any]], tuple | None]


@dataclass
class IterMetrics:
    """Per-iter metrics extracted from a ``game_states.jsonl`` run directory."""

    run_id: str
    total_steps: int
    final_score: float
    score_pct: float
    first_milestone_step: dict[str, int | None]
    dwell_counts: dict[str, int]
    action_count: int
    perseveration_pct: float
    final_zone: str
    error: str | None = None


def extract_iter_metrics(
    run_dir: Path,
    *,
    milestone_specs: list[MilestoneSpec],
    dwell_specs: list[DwellSpec] | None = None,
    action_spec: ActionSpec | None = None,
    score_extractor: Callable[[dict[str, Any]], float],
    zone_extractor: Callable[[dict[str, Any]], str],
    score_max: float = 1.0,
) -> IterMetrics:
    """Extract per-iter behavioural metrics from a ``game_states.jsonl`` run dir.

    The generic framework:
    - ``milestone_specs`` — adapter-supplied predicates for first-bank detection.
    - ``dwell_specs``    — adapter-supplied zone predicates for map-dwell counts.
    - ``action_spec``   — adapter-supplied target extractor for perseveration.
    - ``score_extractor`` / ``zone_extractor`` — pull score + zone from each row.

    Pokemon-specific milestones / zones live in ``game_adapter.TRAJECTORY_MILESTONES``
    etc.; Mario / 2048 ship their own adapter blocks.
    """
    gs = run_dir / "game_states.jsonl"
    if not gs.exists():
        return IterMetrics(
            run_id=run_dir.name,
            total_steps=0,
            final_score=0.0,
            score_pct=0.0,
            first_milestone_step={s.name: None for s in milestone_specs},
            dwell_counts={s.name: 0 for s in (dwell_specs or [])},
            action_count=0,
            perseveration_pct=0.0,
            final_zone="?",
            error="no game_states.jsonl",
        )

    lines = gs.read_text().splitlines()
    first_ms: dict[str, int | None] = {s.name: None for s in milestone_specs}
    dwell: Counter[str] = Counter()
    targets: list[tuple] = []
    final_zone = "?"
    final_score = 0.0

    for i, raw in enumerate(lines):
        try:
            row = json.loads(raw)
        except Exception:
            continue

        for spec in milestone_specs:
            if first_ms[spec.name] is None and spec.predicate(row):
                first_ms[spec.name] = i

        for spec in dwell_specs or []:
            if spec.predicate(row):
                dwell[spec.name] += 1

        if action_spec is not None:
            t = action_spec.extract_target(row)
            if t is not None:
                targets.append(t)

        final_zone = zone_extractor(row)
        final_score = score_extractor(row)

    summ = run_dir / "evaluation_summary.json"
    if summ.exists():
        try:
            with summ.open() as f:
                ep = json.load(f).get("episodes", [{}])[0]
            fs = ep.get("final_score")
            if fs is not None:
                final_score = float(fs)
        except Exception:
            pass

    if len(targets) > 1:
        repeats = sum(1 for j in range(1, len(targets)) if targets[j] == targets[j - 1])
        perseveration = repeats / max(len(targets) - 1, 1) * 100
    else:
        perseveration = 0.0

    return IterMetrics(
        run_id=run_dir.name,
        total_steps=len(lines),
        final_score=final_score,
        score_pct=round(final_score / score_max * 100, 2) if score_max else 0.0,
        first_milestone_step=first_ms,
        dwell_counts={s.name: dwell[s.name] for s in (dwell_specs or [])},
        action_count=len(targets),
        perseveration_pct=round(perseveration, 1),
        final_zone=final_zone,
    )


__all__ = [
    "ActionSpec",
    "DwellSpec",
    "IterMetrics",
    "MilestoneSpec",
    "StepRecord",
    "TrajectoryWriter",
    "convert_scratchpad_to_think",
    "extract_iter_metrics",
    "format_recent_history",
    "has_incomplete_scratchpad",
]
