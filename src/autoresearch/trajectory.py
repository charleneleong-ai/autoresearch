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


@dataclass
class StepRecord:
    """One agent-environment step. Renders to ShareGPT via :meth:`to_sharegpt`."""

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


__all__ = [
    "StepRecord",
    "TrajectoryWriter",
    "convert_scratchpad_to_think",
    "has_incomplete_scratchpad",
]
