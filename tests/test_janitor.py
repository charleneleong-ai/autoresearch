"""Tests for autoresearch.janitor — passive triage for orphan processes.

Detection rules are tested with synthetic ProcInfo lists; no real `ps`.
Mirrors test_gpu_monitor's pattern (inject data, assert pure behaviour).
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from autoresearch.janitor import (
    JanitorConfig,
    KillRule,
    KillTarget,
    ProcInfo,
    _kill_targets,
    find_orphans,
)

_REMOVED = KillRule.REMOVED_WORKTREE_PYTHON
_GAME = KillRule.ORPHAN_GAME_BINARY
_WRAPPER = KillRule.ORPHAN_GAME_SERVER_WRAPPER
_STALE = KillRule.STALE_MULTIPROCESSING_POOL


def _proc(pid: int = 1, cmd: str = "", ppid: int = 1, etime_s: int = 7200) -> ProcInfo:
    return ProcInfo(pid=pid, ppid=ppid, etime_s=etime_s, cmd=cmd)


def _rules(
    cmd: str, *, worktree_gone: bool, config: JanitorConfig = JanitorConfig()
) -> list[KillRule]:
    targets = find_orphans(
        [_proc(cmd=cmd)],
        current_pid=0,
        worktree_exists=lambda p: not worktree_gone,
        config=config,
    )
    return [t.rule for t in targets]


_MP_CMD = "/x/.venv/bin/python -c from multiprocessing.spawn import spawn_main; spawn_main(...)"


# ── Rule matching: cmd × worktree-state → expected rules ────────────────────


@pytest.mark.parametrize(
    "cmd,worktree_gone,expected",
    [
        # Rule 1: removed-worktree python (path-agnostic)
        pytest.param(
            "/workspace/gone/.venv/bin/python run.py", True, [_REMOVED], id="removed_worktree"
        ),
        pytest.param("/workspace/here/.venv/bin/python run.py", False, [], id="live_worktree"),
        pytest.param(
            "/workspace/feat/adopt-keep-recent/.venv/bin/python run.py",
            True,
            [_REMOVED],
            id="nested_with_slashes",
        ),
        # Path-agnostic: regex doesn't anchor to /workspace/. One non-/workspace/
        # case is enough proof — the rule is path-content-insensitive by design.
        pytest.param("/home/runner/work/r/.venv/bin/python s.py", True, [_REMOVED], id="ci_runner"),
        # Rule 2: orphan game binary
        pytest.param("/root/StarCraftII/.../SC2_x64 -listen", False, [_GAME], id="sc2_x64"),
        pytest.param("/some/path/pyboy --foo", False, [_GAME], id="pyboy_as_binary"),
        # Precedence: live-worktree python running a game binary → Rule 2 wins
        pytest.param(
            "/workspace/live/.venv/bin/python /workspace/live/pyboy",
            False,
            [_GAME],
            id="precedence_live_worktree_game_binary",
        ),
        # Word-boundary regression: `pyboy_env` arg ≠ pyboy binary
        pytest.param(
            "/workspace/live/.venv/bin/python train.py --game pyboy_env",
            False,
            [],
            id="pyboy_substring_in_arg",
        ),
        # Rule 2b: orphan mcp game-server wrapper. Live worktree + parent died →
        # `server.py` reparented to PPID=1. Hyphenated game-dir pins `[\w-]+`
        # — covers both `\w` and `-` in one case.
        pytest.param(
            "/x/.venv/bin/python /x/evaluation_utils/mcp_game_servers/pokemon-red/server.py",
            False,
            [_WRAPPER],
            id="mcp_wrapper_hyphenated_game",
        ),
        # Precedence: removed-worktree wins over wrapper rule.
        pytest.param(
            "/gone/.venv/bin/python /gone/evaluation_utils/mcp_game_servers/pokemon_red/server.py",
            True,
            [_REMOVED],
            id="precedence_removed_worktree_over_wrapper",
        ),
        # Path component must contain `mcp_game_servers/<name>/server.py` — a
        # bare `server.py` elsewhere is not a match (avoids killing unrelated
        # internal servers reparented for unrelated reasons).
        pytest.param(
            "/x/.venv/bin/python /opt/some/other/server.py",
            False,
            [],
            id="bare_server_py_no_match",
        ),
        # Safety: KEEP_ALWAYS short-circuits even when worktree is gone
        pytest.param(
            "/workspace/v/.venv/bin/python -m vllm.entrypoints.openai.api_server",
            True,
            [],
            id="keep_vllm",
        ),
        pytest.param("python3 -m http.server 9000", True, [], id="keep_http_server_with_port"),
        pytest.param("python3 -m http.server", True, [], id="keep_http_server_default_port"),
        pytest.param(
            "/usr/bin/python3 /bin/jupyter-notebook --no-browser", True, [], id="keep_jupyter"
        ),
        pytest.param("/opt/syncthing/syncthing serve", True, [], id="keep_syncthing"),
        pytest.param("/opt/tb/tensorboard_data_server", True, [], id="keep_tensorboard"),
        # `\bvllm` boundary: `myvllm` fork is NOT keep-always → falls to Rule 1
        pytest.param(
            "/workspace/dead/.venv/bin/python -m myvllm.entrypoints.api",
            True,
            [_REMOVED],
            id="myvllm_fork_falls_to_rule_1",
        ),
    ],
)
def test_rule_matches(cmd: str, worktree_gone: bool, expected: list[KillRule]):
    assert _rules(cmd, worktree_gone=worktree_gone) == expected


# ── Guards needing non-default proc state ───────────────────────────────────


def test_ppid_not_1_is_never_killed():
    targets = find_orphans(
        [_proc(cmd="/workspace/gone/.venv/bin/python run.py", ppid=999)],
        current_pid=0,
        worktree_exists=lambda p: False,
    )
    assert targets == []


def test_self_pid_is_never_killed():
    targets = find_orphans(
        [_proc(pid=42, cmd="/workspace/gone/.venv/bin/python run.py")],
        current_pid=42,
        worktree_exists=lambda p: False,
    )
    assert targets == []


# ── Rule 3 (stale multiprocessing) — needs etime_s × threshold ──────────────


@pytest.mark.parametrize(
    "etime_s,age_threshold_s,expected",
    [
        pytest.param(3 * 86400, 24 * 3600, [_STALE], id="3d_fires_default_24h"),
        pytest.param(2 * 3600, 24 * 3600, [], id="2h_below_default_24h"),
        pytest.param(2 * 3600, 1 * 3600, [_STALE], id="2h_fires_with_1h_config"),
    ],
)
def test_stale_multiprocessing(etime_s: int, age_threshold_s: int, expected: list[KillRule]):
    targets = find_orphans(
        [_proc(cmd=_MP_CMD, etime_s=etime_s)],
        current_pid=0,
        worktree_exists=lambda p: True,
        config=JanitorConfig(stale_multiprocessing_age_s=age_threshold_s),
    )
    assert [t.rule for t in targets] == expected


# ── JanitorConfig — injection of extras ─────────────────────────────────────


def test_extra_keep_patterns_protect_custom_infra():
    cfg = JanitorConfig(extra_keep_patterns=(re.compile(r"\bdvc daemon\b"),))
    assert (
        _rules(
            "/workspace/gone/.venv/bin/python /usr/bin/dvc daemon", worktree_gone=True, config=cfg
        )
        == []
    )


def test_extra_game_binary_tokens_extend_rule():
    cfg = JanitorConfig(extra_game_binary_tokens=("my_emulator",))
    assert _rules("/some/path/my_emulator --foo", worktree_gone=False, config=cfg) == [_GAME]


def test_extra_game_server_wrapper_patterns_extend_rule():
    # Projects with a non-orak game-server layout can declare their own pattern.
    cfg = JanitorConfig(
        extra_game_server_wrapper_patterns=(re.compile(r"\bmy_envs/[\w-]+/runner\.py\b"),)
    )
    assert _rules(
        "/x/.venv/bin/python /x/my_envs/dota/runner.py --port 9000",
        worktree_gone=False,
        config=cfg,
    ) == [_WRAPPER]


def test_stale_reason_interpolates_configured_threshold():
    cfg = JanitorConfig(stale_multiprocessing_age_s=1 * 3600)
    targets = find_orphans(
        [_proc(cmd=_MP_CMD, etime_s=2 * 3600)],
        current_pid=0,
        worktree_exists=lambda p: True,
        config=cfg,
    )
    assert "> 1h" in targets[0].reason


# ── ProcInfo property + ps parsing ──────────────────────────────────────────


def test_etime_hours_divides_by_3600():
    # Pins the divisor (since Rule 3's reason string interpolates etime_hours
    # and no other test asserts on the value). One non-trivial case is enough.
    assert _proc(etime_s=7200).etime_hours == pytest.approx(2.0)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("2585923 1 41:32 /a/python run.py", (2585923, 1, 41 * 60 + 32)),
        ("2024836 1 3-12:00:00 /b/python -c x", (2024836, 1, 3 * 86400 + 12 * 3600)),
        ("1955474 1 15:23:11 /c/server.py", (1955474, 1, 15 * 3600 + 23 * 60 + 11)),
        ("999 1 00:05 /d/x", (999, 1, 5)),
    ],
)
def test_from_ps_line(line: str, expected: tuple[int, int, int]):
    p = ProcInfo.from_ps_line(line)
    assert (p.pid, p.ppid, p.etime_s) == expected


# ── Kill flow — mock os.kill ────────────────────────────────────────────────


def _target(pid: int = 1) -> KillTarget:
    return KillTarget(_proc(pid=pid), _GAME, "test")


_NO_WAIT = JanitorConfig(sigterm_wait_s=0)


def test_permission_error_recorded_as_denied():
    with patch("autoresearch.janitor.os.kill", side_effect=PermissionError("EPERM")):
        assert _kill_targets([_target()], config=_NO_WAIT) == {1: "denied"}


def test_sigkill_lookup_error_handled():
    # SIGTERM ok → liveness check ok → SIGKILL races and ProcessLookupError.
    with patch(
        "autoresearch.janitor.os.kill", side_effect=[None, None, ProcessLookupError("race")]
    ):
        assert _kill_targets([_target(999)], config=_NO_WAIT) == {999: "term"}
