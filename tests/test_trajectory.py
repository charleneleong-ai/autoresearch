"""Tests for autoresearch.trajectory — TrajectoryWriter, StepRecord, extract_iter_metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path

from autoresearch.trajectory import (
    ActionSpec,
    DwellSpec,
    IterMetrics,
    MilestoneSpec,
    StepRecord,
    TrajectoryWriter,
    convert_scratchpad_to_think,
    extract_iter_metrics,
    format_recent_history,
    has_incomplete_scratchpad,
)

# ── scratchpad helpers ─────────────────────────────────────────────────


def test_convert_scratchpad_to_think_replaces_both_tags() -> None:
    src = "<REASONING_SCRATCHPAD>thinking out loud</REASONING_SCRATCHPAD>then act"
    assert convert_scratchpad_to_think(src) == "<think>thinking out loud</think>then act"


def test_convert_scratchpad_to_think_passthrough_when_absent() -> None:
    assert convert_scratchpad_to_think("just an action") == "just an action"
    assert convert_scratchpad_to_think("") == ""
    assert convert_scratchpad_to_think(None) is None  # type: ignore[arg-type]


def test_has_incomplete_scratchpad_detects_open_only() -> None:
    assert has_incomplete_scratchpad("<REASONING_SCRATCHPAD>abc")
    assert not has_incomplete_scratchpad("<REASONING_SCRATCHPAD>abc</REASONING_SCRATCHPAD>")
    assert not has_incomplete_scratchpad("no tags here")
    assert not has_incomplete_scratchpad("")


# ── StepRecord.to_sharegpt ─────────────────────────────────────────────


def test_step_to_sharegpt_includes_system_when_present() -> None:
    rec = StepRecord(
        step=1,
        system_prompt="be concise",
        user_prompt="what is 2+2?",
        assistant_output="4",
        action="answer",
        tokens_prompt=10,
        tokens_completion=1,
        tokens_total=11,
        cached_tokens=0,
    )
    out = rec.to_sharegpt()
    assert out["step"] == 1
    assert out["action"] == "answer"
    convs = out["conversations"]
    assert convs[0] == {"from": "system", "value": "be concise"}
    assert convs[1] == {"from": "human", "value": "what is 2+2?"}
    assert convs[2] == {"from": "gpt", "value": "4"}
    assert out["tokens"] == {"prompt": 10, "completion": 1, "total": 11, "cached": 0}


def test_step_to_sharegpt_skips_system_when_absent() -> None:
    rec = StepRecord(
        step=2,
        system_prompt=None,
        user_prompt="hi",
        assistant_output="hello",
        action="greet",
    )
    out = rec.to_sharegpt()
    assert len(out["conversations"]) == 2
    assert out["conversations"][0]["from"] == "human"
    assert out["conversations"][1]["from"] == "gpt"


def test_step_to_sharegpt_normalises_scratchpad_in_assistant_value() -> None:
    rec = StepRecord(
        step=3,
        system_prompt=None,
        user_prompt="play",
        assistant_output="<REASONING_SCRATCHPAD>plan</REASONING_SCRATCHPAD>MOVE",
        action="MOVE",
    )
    out = rec.to_sharegpt()
    assert out["conversations"][1]["value"] == "<think>plan</think>MOVE"


def test_step_to_sharegpt_carries_fallback_metadata() -> None:
    rec = StepRecord(
        step=4,
        system_prompt=None,
        user_prompt="?",
        assistant_output="",
        action="NOOP",
        is_fallback=True,
        fallback_reason="LLM 429 timeout",
    )
    out = rec.to_sharegpt()
    assert out["is_fallback"] is True
    assert out["fallback_reason"] == "LLM 429 timeout"


# ── StepRecord outcome fields ──────────────────────────────────────────


def _make_step(step_num: int, *, fallback: bool = False) -> StepRecord:
    return StepRecord(
        step=step_num,
        system_prompt=None,
        user_prompt=f"obs at step {step_num}",
        assistant_output=f"action_{step_num}",
        action=f"a{step_num}",
        tokens_prompt=100,
        tokens_completion=10,
        cached_tokens=50 if step_num > 1 else 0,
        is_fallback=fallback,
    )


def test_step_record_info_score_defaults_to_none() -> None:
    assert _make_step(1).info_score is None


def test_step_record_obs_digest_defaults_to_none() -> None:
    assert _make_step(1).obs_digest is None


def test_step_record_accepts_outcome_fields() -> None:
    rec = StepRecord(
        step=5,
        system_prompt=None,
        user_prompt="u",
        assistant_output="a",
        action="x",
        info_score=3.5,
        obs_digest="deadbeef",
    )
    assert rec.info_score == 3.5
    assert rec.obs_digest == "deadbeef"


# ── TrajectoryWriter ───────────────────────────────────────────────────


def test_writer_creates_log_dir_and_paths(tmp_path: Path) -> None:
    sub = tmp_path / "deep" / "nested"
    writer = TrajectoryWriter(sub, model="gemma-2-2b")
    assert sub.exists()
    assert writer.success_path == sub / "trajectory_samples.jsonl"
    assert writer.failed_path == sub / "failed_trajectories.jsonl"
    assert writer.model == "gemma-2-2b"


def test_clean_episode_lands_in_success_file(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    for i in range(1, 4):
        writer.add_step(_make_step(i))
    target = writer.flush_episode(episode_id=0, completed=True, final_score=12.5, game_name="mario")

    assert target == writer.success_path
    assert writer.success_path.exists()
    assert not writer.failed_path.exists()
    entry = json.loads(writer.success_path.read_text())
    assert entry["episode_id"] == 0
    assert entry["completed"] is True
    assert entry["final_score"] == 12.5
    assert entry["game_name"] == "mario"
    assert entry["n_steps"] == 3
    assert entry["n_fallbacks"] == 0
    assert entry["total_input_tokens"] == 300
    assert entry["total_output_tokens"] == 30
    assert entry["total_cached_tokens"] == 100  # steps 2 & 3 had 50 each


def test_episode_with_any_fallback_lands_in_failed_file(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(_make_step(1))
    writer.add_step(_make_step(2, fallback=True))
    writer.add_step(_make_step(3))
    target = writer.flush_episode(episode_id=0, completed=True, final_score=5.0, game_name="2048")

    assert target == writer.failed_path
    assert not writer.success_path.exists()
    assert json.loads(writer.failed_path.read_text())["n_fallbacks"] == 1


def test_incomplete_episode_lands_in_failed_file_even_without_fallback(tmp_path: Path) -> None:
    """`completed=False` (eg. mid-episode crash) → failed file regardless of fallbacks."""
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(_make_step(1))
    target = writer.flush_episode(
        episode_id=0, completed=False, final_score=0.0, game_name="pokemon"
    )
    assert target == writer.failed_path


def test_buffer_clears_after_flush(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(_make_step(1))
    writer.add_step(_make_step(2))
    writer.flush_episode(episode_id=0, completed=True, final_score=1.0, game_name="g")
    writer.add_step(_make_step(1))
    writer.flush_episode(episode_id=1, completed=True, final_score=2.0, game_name="g")

    lines = writer.success_path.read_text().strip().split("\n")
    assert len(lines) == 2
    ep0, ep1 = json.loads(lines[0]), json.loads(lines[1])
    assert ep0["n_steps"] == 2
    assert ep1["n_steps"] == 1
    assert ep1["episode_id"] == 1


def test_steps_in_entry_are_sharegpt_shaped(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(
        StepRecord(step=0, system_prompt="sys", user_prompt="u", assistant_output="a", action="A")
    )
    writer.flush_episode(episode_id=0, completed=True, final_score=0, game_name="g")
    step = json.loads(writer.success_path.read_text())["steps"][0]
    assert {c["from"] for c in step["conversations"]} == {"system", "human", "gpt"}
    assert step["action"] == "A"


def test_writer_appends_across_calls(tmp_path: Path) -> None:
    """Two writers pointed at the same dir append; they don't truncate."""
    w1 = TrajectoryWriter(tmp_path)
    w1.add_step(_make_step(1))
    w1.flush_episode(episode_id=0, completed=True, final_score=1, game_name="g")

    w2 = TrajectoryWriter(tmp_path)
    w2.add_step(_make_step(1))
    w2.flush_episode(episode_id=1, completed=True, final_score=2, game_name="g")

    assert len(w1.success_path.read_text().strip().split("\n")) == 2


def test_unicode_serialises_without_escape(tmp_path: Path) -> None:
    """`ensure_ascii=False` keeps non-ASCII readable in the JSONL."""
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(
        StepRecord(
            step=0,
            system_prompt=None,
            user_prompt="マリオは ジャンプ する",
            assistant_output="JUMP",
            action="JUMP",
        )
    )
    writer.flush_episode(episode_id=0, completed=True, final_score=0, game_name="mario")
    assert "マリオ" in writer.success_path.read_text()


# ── TrajectoryWriter.recent ────────────────────────────────────────────


def test_recent_returns_last_k(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    for i in range(1, 6):
        writer.add_step(_make_step(i))
    assert [r.step for r in writer.recent(3)] == [3, 4, 5]


def test_recent_k_larger_than_buffer_returns_all(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    for i in range(1, 4):
        writer.add_step(_make_step(i))
    assert len(writer.recent(99)) == 3


def test_recent_k_zero_returns_empty(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(_make_step(1))
    assert writer.recent(0) == []


def test_recent_does_not_mutate_buffer(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    for i in range(1, 4):
        writer.add_step(_make_step(i))
    writer.recent(2).clear()
    assert len(writer._buffer) == 3


# ── format_recent_history ─────────────────────────────────────────────


def test_format_recent_history_empty() -> None:
    assert format_recent_history([]) == ""


def test_format_recent_history_score_delta_and_state_change() -> None:
    def _rec(step: int, score: float, digest: str) -> StepRecord:
        return StepRecord(
            step=step,
            system_prompt=None,
            user_prompt="u",
            assistant_output="a",
            action="north" if step < 3 else "east",
            info_score=score,
            obs_digest=digest,
        )

    records = [_rec(1, 0.0, "aaa"), _rec(2, 0.0, "aaa"), _rec(3, 1.0, "bbb")]
    lines = format_recent_history(records).splitlines()
    assert "state=initial" in lines[0]
    assert "state=unchanged (loop?)" in lines[1]
    assert "(+0)" in lines[1]
    assert "state=changed" in lines[2]
    assert "(+1)" in lines[2]


def test_format_recent_history_missing_score() -> None:
    records = [
        StepRecord(step=1, system_prompt=None, user_prompt="u", assistant_output="a", action="x")
    ]
    out = format_recent_history(records)
    assert "score=?" in out
    assert "state=?" in out


def test_format_recent_history_action_truncated_at_60_chars() -> None:
    records = [
        StepRecord(
            step=1,
            system_prompt=None,
            user_prompt="u",
            assistant_output="a",
            action="z" * 200,
            info_score=1.0,
            obs_digest="x",
        ),
    ]
    out = format_recent_history(records)
    assert "z" * 60 in out
    assert "z" * 61 not in out


# ── extract_iter_metrics ───────────────────────────────────────────────

_MOVE_TO_RE = re.compile(r"move_to[^()]*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def _score(row: dict) -> float:
    gi = row.get("obs", {}).get("game_info", {})
    try:
        return float(int(gi.get("score", 0)))
    except (TypeError, ValueError):
        return 0.0


def _zone(row: dict) -> str:
    return row.get("obs", {}).get("game_info", {}).get("map_name", "?") or "?"


def _move_target(row: dict) -> tuple[int, int] | None:
    s = row.get("action", "")
    m = _MOVE_TO_RE.search(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


_MILESTONES = [MilestoneSpec(f"M{i}", lambda row, i=i: _score(row) >= i) for i in range(1, 4)]
_DWELL = [
    DwellSpec("Route1", lambda row: _zone(row) == "Route1"),
    DwellSpec("Viridian", lambda row: "Viridian" in _zone(row)),
]
_ACTION_SPEC = ActionSpec(extract_target=_move_target)


def _write_gs(tmp_path: Path, rows: list[dict]) -> Path:
    (tmp_path / "game_states.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_extract_basic_milestones_and_dwell(tmp_path: Path) -> None:
    rows = [
        {"obs": {"game_info": {"score": 0, "map_name": "PalletTown"}}, "action": "look"},
        {"obs": {"game_info": {"score": 1, "map_name": "Route1"}}, "action": "move_to(5, 3)"},
        {"obs": {"game_info": {"score": 2, "map_name": "Route1"}}, "action": "move_to(6, 3)"},
        {"obs": {"game_info": {"score": 2, "map_name": "ViridianCity"}}, "action": "move_to(6, 3)"},
    ]
    m = extract_iter_metrics(
        _write_gs(tmp_path, rows),
        milestone_specs=_MILESTONES,
        dwell_specs=_DWELL,
        action_spec=_ACTION_SPEC,
        score_extractor=_score,
        zone_extractor=_zone,
        score_max=3.0,
    )
    assert m.error is None
    assert m.total_steps == 4
    assert m.first_milestone_step == {"M1": 1, "M2": 2, "M3": None}
    assert m.dwell_counts["Route1"] == 2
    assert m.dwell_counts["Viridian"] == 1
    assert m.action_count == 3
    assert m.final_zone == "ViridianCity"
    assert abs(m.score_pct - 2.0 / 3.0 * 100) < 0.1


def test_extract_perseveration(tmp_path: Path) -> None:
    rows = [
        {"obs": {"game_info": {"score": 0, "map_name": "A"}}, "action": "move_to(1, 1)"},
        {"obs": {"game_info": {"score": 0, "map_name": "A"}}, "action": "move_to(1, 1)"},
        {"obs": {"game_info": {"score": 0, "map_name": "A"}}, "action": "move_to(2, 2)"},
        {"obs": {"game_info": {"score": 0, "map_name": "A"}}, "action": "move_to(2, 2)"},
    ]
    m = extract_iter_metrics(
        _write_gs(tmp_path, rows),
        milestone_specs=[],
        action_spec=_ACTION_SPEC,
        score_extractor=_score,
        zone_extractor=_zone,
        score_max=1.0,
    )
    assert abs(m.perseveration_pct - 66.7) < 0.2  # 2 repeats out of 3 pairs


def test_extract_missing_game_states(tmp_path: Path) -> None:
    m = extract_iter_metrics(
        tmp_path, milestone_specs=[], score_extractor=_score, zone_extractor=_zone, score_max=1.0
    )
    assert m.error is not None
    assert "game_states.jsonl" in m.error


def test_extract_evaluation_summary_overrides_score(tmp_path: Path) -> None:
    _write_gs(tmp_path, [{"obs": {"game_info": {"score": 2, "map_name": "A"}}, "action": "look"}])
    (tmp_path / "evaluation_summary.json").write_text(
        json.dumps({"episodes": [{"final_score": 5.0}]})
    )
    m = extract_iter_metrics(
        tmp_path, milestone_specs=[], score_extractor=_score, zone_extractor=_zone, score_max=7.0
    )
    assert m.final_score == 5.0
    assert abs(m.score_pct - 5.0 / 7.0 * 100) < 0.1


def test_extract_no_actions_yields_zero_perseveration(tmp_path: Path) -> None:
    rows = [{"obs": {"game_info": {"score": 0, "map_name": "A"}}, "action": "look"}]
    m = extract_iter_metrics(
        _write_gs(tmp_path, rows),
        milestone_specs=[],
        score_extractor=_score,
        zone_extractor=_zone,
        score_max=1.0,
    )
    assert m.perseveration_pct == 0.0
    assert m.action_count == 0


def test_extract_returns_itmetrics_dataclass(tmp_path: Path) -> None:
    rows = [{"obs": {"game_info": {"score": 0, "map_name": "X"}}, "action": "look"}]
    m = extract_iter_metrics(
        _write_gs(tmp_path, rows),
        milestone_specs=_MILESTONES,
        score_extractor=_score,
        zone_extractor=_zone,
        score_max=3.0,
    )
    assert isinstance(m, IterMetrics)
    assert m.run_id == tmp_path.name
