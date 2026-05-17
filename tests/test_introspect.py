"""Tests for autoresearch.introspect — CLI --format json / text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from autoresearch.introspect import app

_runner = CliRunner()


def _make_adapter_module(tmp_path: Path) -> str:
    """Write a minimal adapter .py into tmp_path and return its dotted module path."""
    adapter_dir = tmp_path / "fake_adapter_pkg"
    adapter_dir.mkdir()
    (adapter_dir / "__init__.py").write_text("")
    code_lines = [
        "from autoresearch.trajectory import MilestoneSpec",
        "def _gi(r): return r.get('obs', {}).get('game_info', {})",
        "TRAJECTORY_MILESTONES = [MilestoneSpec('M1', lambda r: _gi(r).get('score', 0) >= 1)]",
        "TRAJECTORY_SCORE_EXTRACTOR = lambda r: float(_gi(r).get('score', 0))",
        "TRAJECTORY_ZONE_EXTRACTOR = lambda r: _gi(r).get('map_name', '?') or '?'",
        "TRAJECTORY_SCORE_MAX = 1.0",
    ]
    (adapter_dir / "adapter.py").write_text("\n".join(code_lines) + "\n")
    return "fake_adapter_pkg.adapter"


def _make_iter_dir(base: Path, name: str, score: int, map_name: str) -> Path:
    d = base / name
    d.mkdir(parents=True)
    row = {"obs": {"game_info": {"score": score, "map_name": map_name}}, "action": "look"}
    (d / "game_states.jsonl").write_text(json.dumps(row) + "\n")
    return d


def test_format_json_emits_valid_json(tmp_path: Path) -> None:
    _make_iter_dir(tmp_path / "runs", "iter_1", score=1, map_name="Route1")
    adapter_mod = _make_adapter_module(tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        result = _runner.invoke(
            app,
            [
                "--run",
                f"S:{tmp_path / 'runs'}:iter_*",
                "--adapter",
                adapter_mod,
                "--format",
                "json",
            ],
        )
    finally:
        sys.path.pop(0)

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    assert parsed[0]["label"] == "S"
    iters = parsed[0]["iters"]
    assert len(iters) == 1
    assert iters[0]["run_id"] == "iter_1"
    assert iters[0]["score_pct"] == 100.0
    assert iters[0]["first_milestone_step"]["M1"] == 0
    assert "mean_score_pct" in parsed[0]


def test_format_text_is_default(tmp_path: Path) -> None:
    _make_iter_dir(tmp_path / "runs", "iter_2", score=0, map_name="Pallet")
    adapter_mod = _make_adapter_module(tmp_path)
    sys.path.insert(0, str(tmp_path))
    try:
        result = _runner.invoke(
            app,
            ["--run", f"T:{tmp_path / 'runs'}:iter_*", "--adapter", adapter_mod],
        )
    finally:
        sys.path.pop(0)

    assert result.exit_code == 0, result.output
    assert "══════" in result.output
    assert "final=" in result.output
