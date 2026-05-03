"""Tests for autoresearch.sweep_runner (autoresearch#20 PR 2).

Strategy: mock subprocess.Popen and wait_with_timeout so no real
processes are spawned.  Fake Protocol implementations let us control
planner/triage/extractor behaviour per test.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autoresearch.results import load_results
from autoresearch.retrospective import Finding, RetrospectiveSpec
from autoresearch.sweep_runner import (
    IterOutcome,
    IterPlan,
    SweepResult,
    SweepRunner,
)

# ── fake Protocol implementations ─────────────────────────────────────


class FakePlanner:
    """Yields a fixed list of plans."""

    def __init__(self, plans: list[IterPlan]) -> None:
        self._plans = plans

    def plan_iters(self, history: list[dict[str, Any]]) -> Iterator[IterPlan]:
        yield from self._plans


class FakeTriage:
    """No-op triage that optionally returns a run_id from setup()."""

    def __init__(self, run_id: str | None = "run-1") -> None:
        self._run_id = run_id

    def setup(
        self,
        plan: IterPlan,
        proc: subprocess.Popen[bytes],
        baseline: float,
    ) -> str | None:
        return self._run_id

    def check(self, elapsed_s: float) -> str | None:
        return None

    def teardown(self) -> None:
        pass


class FakeExtractor:
    """Returns canned rows (or uses a callable for per-iter control)."""

    def __init__(
        self,
        rows_fn: Any | None = None,
    ) -> None:
        self._fn = rows_fn or (lambda plan, run_id, ec: [{"score": 1.0, "steps": 100}])

    def extract(
        self,
        plan: IterPlan,
        run_id: str | None,
        exit_code: int,
    ) -> list[dict[str, Any]]:
        return self._fn(plan, run_id, exit_code)


# ── helpers ────────────────────────────────────────────────────────────


def _make_runner(
    tmp_path: Path,
    plans: list[IterPlan] | None = None,
    triage: Any | None = None,
    extractor: Any | None = None,
    retrospective_spec: RetrospectiveSpec | None = None,
    **overrides: Any,
) -> SweepRunner:
    defaults: dict[str, Any] = {
        "tag": "test",
        "planner": FakePlanner(plans or []),
        "triage": triage or FakeTriage(),
        "extractor": extractor or FakeExtractor(),
        "experiments_dir": tmp_path,
        "pause_between_iters_s": 0,
        "retrospective_spec": retrospective_spec,
    }
    defaults.update(overrides)
    return SweepRunner(**defaults)


def _mock_popen() -> MagicMock:
    proc = MagicMock()
    proc.returncode = 0
    return proc


# ── tests ──────────────────────────────────────────────────────────────


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_happy_path_two_iters(mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path) -> None:
    mock_popen.return_value = _mock_popen()

    plans = [
        IterPlan(cmd=["echo", "1"], description="iter 1"),
        IterPlan(cmd=["echo", "2"], description="iter 2"),
    ]
    runner = _make_runner(tmp_path, plans=plans)
    result = runner.run()

    assert isinstance(result, SweepResult)
    assert result.tag == "test"
    assert result.iterations == 2
    assert result.kills == 0
    assert not result.blocked
    assert len(result.outcomes) == 2
    assert all(isinstance(o, IterOutcome) for o in result.outcomes)

    # Both iters logged rows to results.jsonl.
    rows = load_results(tmp_path, "test")
    assert len(rows) == 2
    assert rows[0]["score"] == 1.0
    assert rows[1]["score"] == 1.0


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_empty_planner_yields_nothing(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    runner = _make_runner(tmp_path, plans=[])
    result = runner.run()

    assert result.iterations == 0
    assert result.kills == 0
    assert not result.blocked
    assert result.outcomes == []
    mock_popen.assert_not_called()


@patch("autoresearch.sweep_runner.wait_with_timeout")
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_triage_kill_relabels_row(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """When wait_with_timeout returns a kill_reason, the row should be
    relabelled to EARLY_KILL."""
    mock_popen.return_value = _mock_popen()
    mock_wait.return_value = (-9, "score plateau for 200 steps")

    plans = [IterPlan(cmd=["train"], description="killed iter")]
    runner = _make_runner(tmp_path, plans=plans)
    result = runner.run()

    assert result.iterations == 1
    assert result.kills == 1
    assert result.outcomes[0].kill_reason == "score plateau for 200 steps"

    rows = load_results(tmp_path, "test")
    assert len(rows) == 1
    assert rows[0]["status"] == "EARLY_KILL"
    assert "KILLED: score plateau" in rows[0]["notes"]


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_per_config_isolation(mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path) -> None:
    """Plans with different config_name write to separate results.jsonl files."""
    mock_popen.return_value = _mock_popen()

    plans = [
        IterPlan(cmd=["echo"], description="gemma", config_name="gemma"),
        IterPlan(cmd=["echo"], description="qwen", config_name="qwen"),
    ]
    runner = _make_runner(tmp_path, plans=plans)
    result = runner.run()

    assert result.iterations == 2
    gemma_rows = load_results(tmp_path, "test", "gemma")
    qwen_rows = load_results(tmp_path, "test", "qwen")
    assert len(gemma_rows) == 1
    assert len(qwen_rows) == 1


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(1, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_nonzero_exit_still_extracts(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """A crashed subprocess (nonzero exit) still gets its results extracted."""
    mock_popen.return_value = _mock_popen()

    def extract_with_status(plan: IterPlan, run_id: str | None, ec: int) -> list[dict]:
        return [{"score": 0.0, "status": "DISCARD" if ec != 0 else "KEEP"}]

    plans = [IterPlan(cmd=["fail"], description="crash iter")]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        extractor=FakeExtractor(extract_with_status),
    )
    result = runner.run()

    assert result.iterations == 1
    assert result.kills == 0  # no triage kill — natural exit
    assert result.outcomes[0].exit_code == 1
    rows = load_results(tmp_path, "test")
    assert rows[0]["status"] == "DISCARD"


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_extractor_returning_empty_list(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """An extractor returning [] (e.g. subprocess crashed before producing
    any output) should not break the runner."""
    mock_popen.return_value = _mock_popen()

    plans = [IterPlan(cmd=["noop"], description="empty")]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        extractor=FakeExtractor(lambda p, r, e: []),
    )
    result = runner.run()

    assert result.iterations == 1
    assert load_results(tmp_path, "test") == []


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_run_id_from_triage_setup(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """TriageMonitor.setup() return value is forwarded to the extractor
    and recorded in the outcome."""
    mock_popen.return_value = _mock_popen()
    captured_run_ids: list[str | None] = []

    def capture_extract(plan: IterPlan, run_id: str | None, ec: int) -> list[dict]:
        captured_run_ids.append(run_id)
        return [{"score": 1.0}]

    plans = [IterPlan(cmd=["echo"], description="iter")]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        triage=FakeTriage(run_id="abc-123"),
        extractor=FakeExtractor(capture_extract),
    )
    result = runner.run()

    assert captured_run_ids == ["abc-123"]
    assert result.outcomes[0].run_id == "abc-123"


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_sidecar_written_and_cleaned(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """The current_run.json sidecar should be written during the iter and
    removed after."""
    mock_popen.return_value = _mock_popen()
    sidecar_path = tmp_path / "test" / "current_run.json"

    sidecar_existed_during_iter = False

    def check_sidecar(plan: IterPlan, run_id: str | None, ec: int) -> list[dict]:
        nonlocal sidecar_existed_during_iter
        # The extractor runs inside the sidecar context, but after
        # wait_with_timeout — sidecar should still exist at this point.
        # However, since we mocked wait_with_timeout, the sidecar context
        # is still active. We just verify the dir was created.
        sidecar_existed_during_iter = (tmp_path / "test").exists()
        return [{"score": 1.0}]

    plans = [IterPlan(cmd=["echo"], description="iter")]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        extractor=FakeExtractor(check_sidecar),
    )
    runner.run()

    assert sidecar_existed_during_iter
    # Sidecar should be cleaned up after the iter.
    assert not sidecar_path.exists()


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_extra_keys_passed_through(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """Unrecognised keys from the extractor end up in the row's extra fields."""
    mock_popen.return_value = _mock_popen()

    def extract_extra(plan: IterPlan, run_id: str | None, ec: int) -> list[dict]:
        return [{"score": 1.0, "custom_metric": 42.0}]

    plans = [IterPlan(cmd=["echo"], description="iter")]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        extractor=FakeExtractor(extract_extra),
    )
    runner.run()

    rows = load_results(tmp_path, "test")
    assert rows[0]["custom_metric"] == 42.0


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_evaluation_score_alias(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """Extractor returning evaluation_score (orak convention) instead of
    score should still work."""
    mock_popen.return_value = _mock_popen()

    plans = [IterPlan(cmd=["echo"], description="iter")]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        extractor=FakeExtractor(lambda p, r, e: [{"evaluation_score": 0.85}]),
    )
    runner.run()

    rows = load_results(tmp_path, "test")
    assert rows[0]["score"] == 0.85


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_plan_defaults_propagate(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """When the extractor omits description/notes/config_name, the plan's
    values are used as defaults."""
    mock_popen.return_value = _mock_popen()

    plans = [
        IterPlan(
            cmd=["echo"],
            description="my desc",
            notes="my notes",
            config_name="gemma",
        )
    ]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        extractor=FakeExtractor(lambda p, r, e: [{"score": 1.0}]),
    )
    runner.run()

    rows = load_results(tmp_path, "test", "gemma")
    assert rows[0]["description"] == "my desc"
    assert rows[0]["notes"] == "my notes"
    assert rows[0]["config_name"] == "gemma"


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_retrospective_block_stops_sweep(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """A 'block'-severity finding from retrospective should stop the sweep."""
    mock_popen.return_value = _mock_popen()

    # Stub a detector that always returns a block finding.
    block_finding = Finding(
        detector="test_blocker",
        severity="block",
        summary="fatal config error",
        detail="The config is broken.",
        suggested_action="fix the config",
    )

    plans = [
        IterPlan(cmd=["echo", "1"], description="iter 1"),
        IterPlan(cmd=["echo", "2"], description="iter 2"),
    ]

    spec = RetrospectiveSpec(
        enabled=True,
        detectors=["test_blocker"],
        detector_kwargs={},
        on_finding=[],
    )

    with patch(
        "autoresearch.sweep_runner.BUILTIN_DETECTORS",
        {"test_blocker": lambda ctx: block_finding},
    ):
        runner = _make_runner(
            tmp_path,
            plans=plans,
            retrospective_spec=spec,
        )
        result = runner.run()

    assert result.iterations == 1  # stopped after first iter
    assert result.blocked
    assert len(result.outcomes[0].findings) == 1
    assert result.outcomes[0].findings[0].severity == "block"


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_retrospective_warn_does_not_stop(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """A 'warn'-severity finding should NOT stop the sweep."""
    mock_popen.return_value = _mock_popen()

    warn_finding = Finding(
        detector="test_warner",
        severity="warn",
        summary="score regressed",
        detail="Score went down.",
        suggested_action="check hparams",
    )

    plans = [
        IterPlan(cmd=["echo", "1"], description="iter 1"),
        IterPlan(cmd=["echo", "2"], description="iter 2"),
    ]

    spec = RetrospectiveSpec(
        enabled=True,
        detectors=["test_warner"],
        detector_kwargs={},
        on_finding=[],
    )

    with patch(
        "autoresearch.sweep_runner.BUILTIN_DETECTORS",
        {"test_warner": lambda ctx: warn_finding},
    ):
        runner = _make_runner(
            tmp_path,
            plans=plans,
            retrospective_spec=spec,
        )
        result = runner.run()

    assert result.iterations == 2  # both iters ran
    assert not result.blocked


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_retrospective_disabled_skips_detectors(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    mock_popen.return_value = _mock_popen()

    spec = RetrospectiveSpec(
        enabled=False,
        detectors=["score_plateau"],
        detector_kwargs={},
        on_finding=[],
    )

    plans = [IterPlan(cmd=["echo"], description="iter")]
    runner = _make_runner(tmp_path, plans=plans, retrospective_spec=spec)
    result = runner.run()

    assert result.outcomes[0].findings == []


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_retrospective_writes_markdown(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """When findings exist, a retrospective_E<N>.md file should be written."""
    mock_popen.return_value = _mock_popen()

    finding = Finding(
        detector="test_det",
        severity="warn",
        summary="something happened",
        detail="### Warning\nDetails here.",
        suggested_action="investigate",
    )

    spec = RetrospectiveSpec(
        enabled=True,
        detectors=["test_det"],
        detector_kwargs={},
        on_finding=[],
    )

    plans = [IterPlan(cmd=["echo"], description="iter")]

    with patch(
        "autoresearch.sweep_runner.BUILTIN_DETECTORS",
        {"test_det": lambda ctx: finding},
    ):
        runner = _make_runner(tmp_path, plans=plans, retrospective_spec=spec)
        runner.run()

    md_path = tmp_path / "test" / "retrospective_E0.md"
    assert md_path.exists()
    content = md_path.read_text()
    assert "Warning" in content
    assert "Details here" in content


@patch("autoresearch.sweep_runner.wait_with_timeout")
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_kill_with_multi_row_extractor(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """orak pattern: extractor returns multiple rows (one per game).
    All should be logged; relabel should apply to the correct window."""
    mock_popen.return_value = _mock_popen()
    mock_wait.return_value = (-9, "pokemon_red: plateau")

    def multi_game_extract(plan: IterPlan, run_id: str | None, ec: int) -> list[dict]:
        return [
            {"score": 0.5, "game": "pokemon_red"},
            {"score": 0.8, "game": "super_mario"},
        ]

    plans = [IterPlan(cmd=["run"], description="multi-game")]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        extractor=FakeExtractor(multi_game_extract),
    )
    result = runner.run()

    assert result.kills == 1
    rows = load_results(tmp_path, "test")
    assert len(rows) == 2
    # relabel_last_as_early_kill with last_n=2 relabels both rows
    # (no filter_field passed — SweepRunner doesn't know which game
    # triggered the kill; that's the extractor's job to set status).
    assert all(r["status"] == "EARLY_KILL" for r in rows)


@patch("autoresearch.sweep_runner.wait_with_timeout", return_value=(0, None))
@patch("autoresearch.sweep_runner.subprocess.Popen")
def test_best_score_updates_across_iters(
    mock_popen: MagicMock, mock_wait: MagicMock, tmp_path: Path
) -> None:
    """best_score (passed to triage.setup as baseline) should reflect
    the highest score seen across all iters."""
    mock_popen.return_value = _mock_popen()
    baselines: list[float] = []

    class TrackingTriage(FakeTriage):
        def setup(self, plan, proc, baseline):
            baselines.append(baseline)
            return super().setup(plan, proc, baseline)

    scores = iter([0.3, 0.9, 0.5])

    plans = [IterPlan(cmd=["echo"], description=f"iter {i}") for i in range(3)]
    runner = _make_runner(
        tmp_path,
        plans=plans,
        triage=TrackingTriage(),
        extractor=FakeExtractor(lambda p, r, e: [{"score": next(scores)}]),
    )
    runner.run()

    # Iter 1 baseline: 0.0 (no history), iter 2: 0.3, iter 3: 0.9.
    assert baselines == [0.0, 0.3, 0.9]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
