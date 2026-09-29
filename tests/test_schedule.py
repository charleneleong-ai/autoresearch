"""Schedule loading, resumable planning, and score-json extraction with decide_status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.results import log_experiment
from autoresearch.schedule import Schedule, SchedulePlanner, ScoreJsonExtractor, tag_of
from autoresearch.sweep_runner import IterPlan


@pytest.fixture
def schedule(tmp_path: Path) -> Schedule:
    p = tmp_path / "v9.yaml"
    p.write_text(
        "common_overrides: [--epochs, 1]\niters:\n"
        "  - {description: a, overrides: [--context, 0]}\n"
        "  - {description: b, overrides: [--context, 2], timeout_min: 5}\n"
    )
    return Schedule.load(p)


def test_planner_builds_tagged_commands_and_skips_logged_iters(schedule, tmp_path):
    planner = SchedulePlanner(schedule, "mlp", ["python", "-m", "proj.cli"], tmp_path)
    plans = list(planner.plan_iters([{"description": "v9_i00 a"}]))
    assert [p.description for p in plans] == ["v9_i01 b"]
    assert plans[0].cmd == [
        "python",
        "-m",
        "proj.cli",
        "train",
        "--epochs",
        "1",
        "--context",
        "2",
        "--tag",
        "v9_i01",
    ]
    assert plans[0].timeout_min == 5 and plans[0].notes == "--context 2"
    assert tag_of({"description": plans[0].description}) == "v9_i01"


def test_extractor_labels_baseline_keep_discard_and_crash(schedule, tmp_path):
    exp = tmp_path / "experiments"
    planner = SchedulePlanner(schedule, "mlp", ["python"], tmp_path)
    p0, p1 = list(planner.plan_iters([]))
    ex = ScoreJsonExtractor(tmp_path, "mlp", exp, "t", metric_keys=("r2",))
    (tmp_path / "v9_i00_score.json").write_text(
        json.dumps({"score": 0.8, "r2": 0.9, "steps": 3, "wandb_url": "u"})
    )
    (tmp_path / "v9_i01_score.json").write_text(json.dumps({"score": 0.7}))
    row0 = ex.extract(p0, None, 0)[0]
    assert row0["status"] == "BASELINE" and row0["metrics"] == {"r2": 0.9} and row0["steps"] == 3
    log_experiment(experiments_dir=exp, tag="t", config_name="mlp", score=0.8, status="BASELINE")
    assert ex.extract(p1, None, 0)[0]["status"] == "DISCARD"
    assert ex.extract(p1, None, 1)[0]["status"] == "CRASH"
    assert (
        ex.extract(IterPlan(cmd=[], description="v9_i07 missing"), None, 0)[0]["status"] == "CRASH"
    )


def test_iter_command_runs_without_the_schedule_common_overrides(tmp_path):
    p = tmp_path / "v9.yaml"
    p.write_text(
        "common_overrides: [--epochs, 1]\niters:\n"
        "  - {description: seed, overrides: [--seed, 1]}\n"
        "  - {description: avg, command: ensemble, overrides: [--members, 'v9_i00,base']}\n"
    )
    plans = list(SchedulePlanner(Schedule.load(p), "mlp", ["proj"], tmp_path).plan_iters([]))
    assert plans[0].cmd == ["proj", "train", "--epochs", "1", "--seed", "1", "--tag", "v9_i00"]
    assert plans[1].cmd == ["proj", "ensemble", "--members", "v9_i00,base", "--tag", "v9_i01"]
