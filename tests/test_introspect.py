"""Tests for autoresearch.introspect — CLI --format json / text."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoresearch.introspect import app

_runner = CliRunner()


def _make_adapter_module(tmp_path: Path) -> str:
    """Write a minimal adapter .py into tmp_path and return its dotted module path."""
    pkg = tmp_path / "fake_adapter_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "adapter.py").write_text("""\
from autoresearch.trajectory import MilestoneSpec
def _gi(r): return r.get('obs', {}).get('game_info', {})
TRAJECTORY_MILESTONES = [MilestoneSpec('M1', lambda r: _gi(r).get('score', 0) >= 1)]
TRAJECTORY_SCORE_EXTRACTOR = lambda r: float(_gi(r).get('score', 0))
TRAJECTORY_ZONE_EXTRACTOR = lambda r: _gi(r).get('map_name', '?') or '?'
TRAJECTORY_SCORE_MAX = 1.0
""")
    return "fake_adapter_pkg.adapter"


def _make_iter_dir(base: Path, name: str, *, score: int, map_name: str) -> Path:
    d = base / name
    d.mkdir(parents=True)
    row = {"obs": {"game_info": {"score": score, "map_name": map_name}}, "action": "look"}
    (d / "game_states.jsonl").write_text(json.dumps(row) + "\n")
    return d


@contextmanager
def _on_path(p: Path):
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path.pop(0)


@pytest.fixture
def adapter(tmp_path: Path):
    """Yields (adapter_module_str, runs_dir) with tmp_path on sys.path."""
    mod = _make_adapter_module(tmp_path)
    runs = tmp_path / "runs"
    with _on_path(tmp_path):
        yield mod, runs


def test_format_json_emits_valid_json(adapter) -> None:
    mod, runs = adapter
    _make_iter_dir(runs, "iter_1", score=1, map_name="Route1")

    result = _runner.invoke(
        app, ["--run", f"S:{runs}:iter_*", "--adapter", mod, "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed[0]["label"] == "S"
    iters = parsed[0]["iters"]
    assert len(iters) == 1
    assert iters[0]["run_id"] == "iter_1"
    assert iters[0]["score_pct"] == 100.0
    assert iters[0]["first_milestone_step"]["M1"] == 0
    assert "mean_score_pct" in parsed[0]


def test_format_text_is_default(adapter) -> None:
    mod, runs = adapter
    _make_iter_dir(runs, "iter_1", score=0, map_name="Pallet")

    result = _runner.invoke(app, ["--run", f"T:{runs}:iter_*", "--adapter", mod])

    assert result.exit_code == 0, result.output
    assert "══════" in result.output
    assert "final=" in result.output


def test_format_text_renders_step_zero_milestones(adapter) -> None:
    """Milestone reached at step 0 should render as `M1@0`, not `M1@n/a` —
    the `or 'n/a'` idiom mistakenly treats 0 as missing."""
    mod, runs = adapter
    _make_iter_dir(runs, "iter_1", score=1, map_name="Route1")

    result = _runner.invoke(app, ["--run", f"T:{runs}:iter_*", "--adapter", mod])

    assert result.exit_code == 0, result.output
    assert "M1@0" in result.output
    assert "M1@n/a" not in result.output
