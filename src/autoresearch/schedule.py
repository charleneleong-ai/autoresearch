"""Schedule-driven planning for :class:`SweepRunner`, plus a score-json extractor.

The minimal contract a project needs to run a sweep:

* ``configs/schedules/<name>.yaml`` with ``command`` (sub-command of the train
  CLI), ``common_overrides`` (flat CLI list) and ``iters`` (``description`` +
  ``overrides`` each).
* A train CLI that accepts ``--tag`` and writes ``<results_dir>/<tag>_score.json``
  holding at least the primary score (``score_key``), and optionally ``wandb_url``,
  ``runtime_min`` and any per-term metrics.

``SchedulePlanner`` turns each iter into an :class:`IterPlan` tagged
``<schedule>_i<NN>`` and skips iters that already have a logged row, so a
sweep can be relaunched after a crash. ``ScoreJsonExtractor`` reads the json
back into a results row, classifying it with :func:`decide_status` against
the rows already logged for the config.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from autoresearch.results import STATUS_CRASH, decide_status, load_results
from autoresearch.sweep_runner import IterPlan


@dataclass
class Schedule:
    name: str
    command: str
    common_overrides: list[str]
    iters: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> Schedule:
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            name=Path(path).stem,
            command=str(data.get("command", "train")),
            common_overrides=[str(x) for x in data.get("common_overrides", [])],
            iters=list(data["iters"]),
        )

    def iter_tag(self, index: int) -> str:
        return f"{self.name}_i{index:02d}"


def tag_of(row: dict[str, Any]) -> str:
    """The iter tag is the first token of the row's description."""
    return str(row.get("description", "")).split(" ")[0]


@dataclass
class SchedulePlanner:
    """One :class:`IterPlan` per schedule iter, in order, skipping already-logged tags."""

    schedule: Schedule
    config_name: str
    train_cmd: list[str]  # e.g. ["python", "-m", "myproj.cli"]; the schedule's command is appended
    results_dir: Path
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    default_timeout_min: int = 180

    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        done = {tag_of(r) for r in history}
        for i, it in enumerate(self.schedule.iters):
            tag = self.schedule.iter_tag(i)
            if tag in done:
                continue
            overrides = [str(x) for x in it.get("overrides", [])]
            yield IterPlan(
                cmd=[
                    *self.train_cmd,
                    self.schedule.command,
                    *self.schedule.common_overrides,
                    *overrides,
                    "--tag",
                    tag,
                ],
                description=f"{tag} {it.get('description', '')}".strip(),
                config_name=self.config_name,
                notes=" ".join(overrides),
                env={**os.environ, **self.env},
                cwd=self.cwd,
                timeout_min=int(it.get("timeout_min", self.default_timeout_min)),
            )


@dataclass
class ScoreJsonExtractor:
    """``<results_dir>/<tag>_score.json`` -> results row; missing json or non-zero exit -> CRASH."""

    results_dir: Path
    config_name: str
    experiments_dir: Path
    tag: str
    score_key: str = "score"
    metric_keys: tuple[str, ...] = ()
    steps_key: str = "steps"

    def extract(self, plan: IterPlan, run_id: str | None, exit_code: int) -> list[dict[str, Any]]:
        iter_tag = tag_of({"description": plan.description})
        path = Path(self.results_dir) / f"{iter_tag}_score.json"
        base = {
            "description": plan.description,
            "config_name": self.config_name,
            "notes": plan.notes,
        }
        if exit_code != 0 or not path.exists():
            return [{**base, "score": 0.0, "status": STATUS_CRASH}]
        data = json.loads(path.read_text())
        score = float(data[self.score_key])
        history = load_results(self.experiments_dir, self.tag, self.config_name)
        return [
            {
                **base,
                "score": score,
                "status": decide_status(history, score),
                "wandb_url": str(data.get("wandb_url", "")),
                "runtime_min": float(data.get("runtime_min", 0.0)),
                "steps": int(data.get(self.steps_key, 0) or 0),
                "metrics": {k: data[k] for k in self.metric_keys if k in data},
            }
        ]
