"""Tests for the bucket-1 sweep helpers from autoresearch#20:
* `relabel_last_as_early_kill` (in `autoresearch.results`)
* `write_sidecar` / `clear_sidecar` / `sidecar` context manager (in `autoresearch.current_run`)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.current_run import clear_sidecar, sidecar, write_sidecar
from autoresearch.results import (
    log_experiment,
    relabel_last_as_early_kill,
)

# ── relabel_last_as_early_kill ────────────────────────────────────────


def test_relabel_single_row_default_case(tmp_path: Path) -> None:
    """gemma4-rlvr's pattern: one iter writes one row; relabel that row."""
    log_experiment(
        experiments_dir=tmp_path,
        tag="t",
        score=5.0,
        steps=100,
        status="KEEP",
    )
    n = relabel_last_as_early_kill(
        experiments_dir=tmp_path, tag="t", kill_reason="iteration timeout (30min)"
    )
    assert n == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "t" / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["status"] == "EARLY_KILL"
    assert "KILLED: iteration timeout" in rows[0]["notes"]


def test_relabel_filters_by_field(tmp_path: Path) -> None:
    """orak's pattern: one iter writes multiple rows (per game); relabel
    only the offender's row, leave bystanders' rows untouched."""
    for game in ["pokemon_red", "super_mario", "twenty_fourty_eight"]:
        log_experiment(
            experiments_dir=tmp_path,
            tag="t",
            game=game,
            score=0.0,
            status="DISCARD",
        )
    n = relabel_last_as_early_kill(
        experiments_dir=tmp_path,
        tag="t",
        kill_reason="pokemon_red: score plateau (0.00%) for 200 steps",
        filter_field="game",
        filter_values=["pokemon_red"],
        last_n=3,
    )
    assert n == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "t" / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_game = {r["game"]: r["status"] for r in rows}
    assert by_game["pokemon_red"] == "EARLY_KILL"
    assert by_game["super_mario"] == "DISCARD"
    assert by_game["twenty_fourty_eight"] == "DISCARD"


def test_relabel_returns_zero_when_file_missing(tmp_path: Path) -> None:
    n = relabel_last_as_early_kill(experiments_dir=tmp_path, tag="never_existed", kill_reason="x")
    assert n == 0


def test_relabel_returns_zero_when_file_empty(tmp_path: Path) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "results.jsonl").write_text("")
    n = relabel_last_as_early_kill(experiments_dir=tmp_path, tag="t", kill_reason="x")
    assert n == 0


def test_relabel_filter_no_match_leaves_rows_untouched(tmp_path: Path) -> None:
    log_experiment(experiments_dir=tmp_path, tag="t", game="A", score=1.0, status="KEEP")
    n = relabel_last_as_early_kill(
        experiments_dir=tmp_path,
        tag="t",
        kill_reason="x",
        filter_field="game",
        filter_values=["B"],  # doesn't match
        last_n=3,
    )
    assert n == 0
    row = json.loads((tmp_path / "t" / "results.jsonl").read_text().strip().splitlines()[0])
    assert row["status"] == "KEEP"  # unchanged


def test_relabel_per_config_layout(tmp_path: Path) -> None:
    log_experiment(experiments_dir=tmp_path, tag="t", config_name="gemma", score=5.0, status="KEEP")
    log_experiment(experiments_dir=tmp_path, tag="t", config_name="qwen", score=5.0, status="KEEP")
    n = relabel_last_as_early_kill(
        experiments_dir=tmp_path,
        tag="t",
        config_name="gemma",
        kill_reason="x",
    )
    assert n == 1
    gemma_row = json.loads(
        (tmp_path / "t" / "gemma" / "results.jsonl").read_text().strip().splitlines()[-1]
    )
    qwen_row = json.loads(
        (tmp_path / "t" / "qwen" / "results.jsonl").read_text().strip().splitlines()[-1]
    )
    assert gemma_row["status"] == "EARLY_KILL"
    assert qwen_row["status"] == "KEEP"  # other config untouched


def test_relabel_only_inspects_last_n_rows(tmp_path: Path) -> None:
    """A pre-existing older row should NOT be relabelled even if it matches the filter."""
    log_experiment(experiments_dir=tmp_path, tag="t", game="A", score=1.0, status="KEEP")
    log_experiment(experiments_dir=tmp_path, tag="t", game="B", score=1.0, status="KEEP")
    n = relabel_last_as_early_kill(
        experiments_dir=tmp_path,
        tag="t",
        kill_reason="x",
        filter_field="game",
        filter_values=["A"],
        last_n=1,  # only inspect the last row (which is B, not A)
    )
    assert n == 0  # A is NOT in the last_n window
    rows = [
        json.loads(line)
        for line in (tmp_path / "t" / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(r["status"] == "KEEP" for r in rows)


def test_relabel_preserves_existing_notes(tmp_path: Path) -> None:
    log_experiment(
        experiments_dir=tmp_path, tag="t", score=5.0, status="DISCARD", notes="below best"
    )
    relabel_last_as_early_kill(experiments_dir=tmp_path, tag="t", kill_reason="reason")
    row = json.loads((tmp_path / "t" / "results.jsonl").read_text().strip().splitlines()[0])
    assert "KILLED: reason" in row["notes"]
    assert "below best" in row["notes"]  # preserved


# ── sidecar (write / clear / context manager) ──────────────────────────


def test_write_sidecar_creates_file(tmp_path: Path) -> None:
    payload = {
        "experiment": 5,
        "config_name": "gemma",
        "description": "iter 6 | θ=0.20",
        "started_at": "2026-05-03T12:34:56Z",
    }
    p = write_sidecar(payload, tag="t", config_name="gemma", experiments_dir=tmp_path)
    assert p == tmp_path / "t" / "gemma" / "current_run.json"
    assert json.loads(p.read_text()) == payload


def test_clear_sidecar_when_present(tmp_path: Path) -> None:
    write_sidecar({"x": 1}, tag="t", experiments_dir=tmp_path)
    assert clear_sidecar(tag="t", experiments_dir=tmp_path) is True
    assert not (tmp_path / "t" / "current_run.json").exists()


def test_clear_sidecar_when_absent_returns_false(tmp_path: Path) -> None:
    assert clear_sidecar(tag="t", experiments_dir=tmp_path) is False


def test_sidecar_context_manager_unlinks_on_normal_exit(tmp_path: Path) -> None:
    with sidecar({"experiment": 0}, tag="t", experiments_dir=tmp_path) as p:
        assert p.exists()
        assert json.loads(p.read_text()) == {"experiment": 0}
    assert not p.exists()


def test_sidecar_context_manager_unlinks_on_exception(tmp_path: Path) -> None:
    """Cleanup must run even if the iter blows up — otherwise the chart's
    RUNNING dot stays stale until manual intervention."""
    with pytest.raises(RuntimeError, match="iter exploded"):
        with sidecar({"experiment": 0}, tag="t", experiments_dir=tmp_path) as p:
            assert p.exists()
            raise RuntimeError("iter exploded")
    assert not p.exists()


def test_sidecar_per_config_isolation(tmp_path: Path) -> None:
    with sidecar({"x": 1}, tag="t", config_name="gemma", experiments_dir=tmp_path) as g:
        with sidecar({"x": 2}, tag="t", config_name="qwen", experiments_dir=tmp_path) as q:
            assert g != q
            assert json.loads(g.read_text())["x"] == 1
            assert json.loads(q.read_text())["x"] == 2


def test_sidecar_overwrites_stale_payload(tmp_path: Path) -> None:
    """A leftover sidecar from a previous run shouldn't bleed into the next iter."""
    write_sidecar({"experiment": 99, "stale": True}, tag="t", experiments_dir=tmp_path)
    with sidecar({"experiment": 0}, tag="t", experiments_dir=tmp_path) as p:
        assert json.loads(p.read_text()) == {"experiment": 0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
