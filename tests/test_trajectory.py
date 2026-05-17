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

# ── helpers ────────────────────────────────────────────────────────────

_MOVE_TO_RE = re.compile(r"move_to[^()]*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def _make_step(n: int, *, fallback: bool = False) -> StepRecord:
    return StepRecord(
        step=n,
        system_prompt=None,
        user_prompt=f"obs {n}",
        assistant_output=f"out {n}",
        action=f"a{n}",
        tokens_prompt=100,
        tokens_completion=10,
        cached_tokens=50 if n > 1 else 0,
        is_fallback=fallback,
    )


def _score(row: dict) -> float:
    try:
        return float(int(row.get("obs", {}).get("game_info", {}).get("score", 0)))
    except (TypeError, ValueError):
        return 0.0


def _zone(row: dict) -> str:
    return row.get("obs", {}).get("game_info", {}).get("map_name", "?") or "?"


def _move_target(row: dict) -> tuple[int, int] | None:
    m = _MOVE_TO_RE.search(row.get("action", ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _write_gs(tmp_path: Path, rows: list[dict]) -> Path:
    (tmp_path / "game_states.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def _eim(run_dir: Path, *, milestone_specs=(), dwell_specs=None, action_spec=None, score_max=1.0):
    """Thin wrapper — pre-fills the shared score/zone extractors."""
    return extract_iter_metrics(
        run_dir,
        milestone_specs=list(milestone_specs),
        dwell_specs=dwell_specs,
        action_spec=action_spec,
        score_extractor=_score,
        zone_extractor=_zone,
        score_max=score_max,
    )


_MILESTONES = [MilestoneSpec(f"M{i}", lambda row, i=i: _score(row) >= i) for i in range(1, 4)]
_DWELL = [
    DwellSpec("Route1", lambda row: _zone(row) == "Route1"),
    DwellSpec("Viridian", lambda row: "Viridian" in _zone(row)),
]
_ACTION_SPEC = ActionSpec(extract_target=_move_target)

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


# ── StepRecord ────────────────────────────────────────────────────────


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
    out = StepRecord(
        step=2, system_prompt=None, user_prompt="hi", assistant_output="hello", action="greet"
    ).to_sharegpt()
    assert len(out["conversations"]) == 2
    assert out["conversations"][0]["from"] == "human"


def test_step_to_sharegpt_normalises_scratchpad() -> None:
    out = StepRecord(
        step=3,
        system_prompt=None,
        user_prompt="play",
        assistant_output="<REASONING_SCRATCHPAD>plan</REASONING_SCRATCHPAD>MOVE",
        action="MOVE",
    ).to_sharegpt()
    assert out["conversations"][1]["value"] == "<think>plan</think>MOVE"


def test_step_to_sharegpt_carries_fallback_metadata() -> None:
    out = StepRecord(
        step=4,
        system_prompt=None,
        user_prompt="?",
        assistant_output="",
        action="NOOP",
        is_fallback=True,
        fallback_reason="LLM 429 timeout",
    ).to_sharegpt()
    assert out["is_fallback"] is True
    assert out["fallback_reason"] == "LLM 429 timeout"


def test_step_record_outcome_fields() -> None:
    rec = _make_step(1)
    assert rec.info_score is None
    assert rec.obs_digest is None

    rec2 = StepRecord(
        step=5,
        system_prompt=None,
        user_prompt="u",
        assistant_output="a",
        action="x",
        info_score=3.5,
        obs_digest="deadbeef",
    )
    assert rec2.info_score == 3.5
    assert rec2.obs_digest == "deadbeef"


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
    assert not writer.failed_path.exists()
    entry = json.loads(writer.success_path.read_text())
    assert entry == {
        **entry,
        "episode_id": 0,
        "completed": True,
        "final_score": 12.5,
        "game_name": "mario",
        "n_steps": 3,
        "n_fallbacks": 0,
        "total_input_tokens": 300,
        "total_output_tokens": 30,
        "total_cached_tokens": 100,
    }


def test_episode_with_fallback_lands_in_failed_file(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    for step in [_make_step(1), _make_step(2, fallback=True), _make_step(3)]:
        writer.add_step(step)
    target = writer.flush_episode(episode_id=0, completed=True, final_score=5.0, game_name="2048")
    assert target == writer.failed_path
    assert not writer.success_path.exists()
    assert json.loads(writer.failed_path.read_text())["n_fallbacks"] == 1


def test_incomplete_episode_lands_in_failed_file(tmp_path: Path) -> None:
    """`completed=False` → failed file regardless of fallbacks."""
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(_make_step(1))
    assert (
        writer.flush_episode(episode_id=0, completed=False, final_score=0.0, game_name="pokemon")
        == writer.failed_path
    )


def test_buffer_clears_after_flush(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    for step in [_make_step(1), _make_step(2)]:
        writer.add_step(step)
    writer.flush_episode(episode_id=0, completed=True, final_score=1.0, game_name="g")
    writer.add_step(_make_step(1))
    writer.flush_episode(episode_id=1, completed=True, final_score=2.0, game_name="g")

    lines = writer.success_path.read_text().strip().split("\n")
    ep0, ep1 = json.loads(lines[0]), json.loads(lines[1])
    assert ep0["n_steps"] == 2 and ep1["n_steps"] == 1 and ep1["episode_id"] == 1


def test_steps_in_entry_are_sharegpt_shaped(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    writer.add_step(
        StepRecord(step=0, system_prompt="sys", user_prompt="u", assistant_output="a", action="A")
    )
    writer.flush_episode(episode_id=0, completed=True, final_score=0, game_name="g")
    step = json.loads(writer.success_path.read_text())["steps"][0]
    assert {c["from"] for c in step["conversations"]} == {"system", "human", "gpt"}
    assert step["action"] == "A"


def test_writer_appends_across_instances(tmp_path: Path) -> None:
    for ep, score in enumerate([1, 2]):
        w = TrajectoryWriter(tmp_path)
        w.add_step(_make_step(1))
        w.flush_episode(episode_id=ep, completed=True, final_score=score, game_name="g")
    assert len(TrajectoryWriter(tmp_path).success_path.read_text().strip().split("\n")) == 2


def test_unicode_serialises_without_escape(tmp_path: Path) -> None:
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


def test_recent_edge_cases(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path)
    for i in range(1, 4):
        writer.add_step(_make_step(i))
    assert len(writer.recent(99)) == 3  # k > buffer → all
    assert writer.recent(0) == []  # k=0 → empty
    writer.recent(2).clear()
    assert len(writer._buffer) == 3  # copy, not view


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

    lines = format_recent_history(
        [_rec(1, 0.0, "aaa"), _rec(2, 0.0, "aaa"), _rec(3, 1.0, "bbb")]
    ).splitlines()
    assert "state=initial" in lines[0]
    assert "state=unchanged (loop?)" in lines[1] and "(+0)" in lines[1]
    assert "state=changed" in lines[2] and "(+1)" in lines[2]


def test_format_recent_history_missing_score() -> None:
    out = format_recent_history(
        [StepRecord(step=1, system_prompt=None, user_prompt="u", assistant_output="a", action="x")]
    )
    assert "score=?" in out and "state=?" in out


def test_format_recent_history_action_truncated_at_60_chars() -> None:
    out = format_recent_history(
        [
            StepRecord(
                step=1,
                system_prompt=None,
                user_prompt="u",
                assistant_output="a",
                action="z" * 200,
                info_score=1.0,
                obs_digest="x",
            )
        ]
    )
    assert "z" * 60 in out and "z" * 61 not in out


# ── extract_iter_metrics ───────────────────────────────────────────────


def test_extract_basic_milestones_and_dwell(tmp_path: Path) -> None:
    rows = [
        {"obs": {"game_info": {"score": 0, "map_name": "PalletTown"}}, "action": "look"},
        {"obs": {"game_info": {"score": 1, "map_name": "Route1"}}, "action": "move_to(5, 3)"},
        {"obs": {"game_info": {"score": 2, "map_name": "Route1"}}, "action": "move_to(6, 3)"},
        {"obs": {"game_info": {"score": 2, "map_name": "ViridianCity"}}, "action": "move_to(6, 3)"},
    ]
    m = _eim(
        _write_gs(tmp_path, rows),
        milestone_specs=_MILESTONES,
        dwell_specs=_DWELL,
        action_spec=_ACTION_SPEC,
        score_max=3.0,
    )
    assert m.error is None
    assert m.total_steps == 4
    assert m.first_milestone_step == {"M1": 1, "M2": 2, "M3": None}
    assert m.dwell_counts == {"Route1": 2, "Viridian": 1}
    assert m.action_count == 3
    assert m.final_zone == "ViridianCity"
    assert abs(m.score_pct - 2.0 / 3.0 * 100) < 0.1


def test_extract_perseveration(tmp_path: Path) -> None:
    rows = [
        {"obs": {"game_info": {"score": 0, "map_name": "A"}}, "action": f"move_to({x}, {y})"}
        for x, y in [(1, 1), (1, 1), (2, 2), (2, 2)]
    ]
    m = _eim(_write_gs(tmp_path, rows), action_spec=_ACTION_SPEC)
    assert abs(m.perseveration_pct - 66.7) < 0.2  # 2 repeats out of 3 pairs


def test_extract_missing_game_states(tmp_path: Path) -> None:
    m = _eim(tmp_path)
    assert m.error is not None and "game_states.jsonl" in m.error


def test_extract_evaluation_summary_overrides_score(tmp_path: Path) -> None:
    _write_gs(tmp_path, [{"obs": {"game_info": {"score": 2, "map_name": "A"}}, "action": "look"}])
    (tmp_path / "evaluation_summary.json").write_text(
        json.dumps({"episodes": [{"final_score": 5.0}]})
    )
    m = _eim(tmp_path, score_max=7.0)
    assert m.final_score == 5.0
    assert abs(m.score_pct - 5.0 / 7.0 * 100) < 0.1


def test_extract_no_actions_and_dataclass_type(tmp_path: Path) -> None:
    rows = [{"obs": {"game_info": {"score": 0, "map_name": "X"}}, "action": "look"}]
    m = _eim(_write_gs(tmp_path, rows), milestone_specs=_MILESTONES, score_max=3.0)
    assert isinstance(m, IterMetrics)
    assert m.run_id == tmp_path.name
    assert m.perseveration_pct == 0.0
    assert m.action_count == 0
