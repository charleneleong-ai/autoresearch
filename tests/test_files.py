"""Tests for autoresearch.files — keep_recent rotation + warn_if_tmp_data_dir."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from autoresearch.files import keep_recent, warn_if_tmp_data_dir

# ── helpers ────────────────────────────────────────────────────────────


def _touch(path: Path, *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _step_key(p: Path) -> int:
    """Extract trailing integer from `agent_step_NNN.pkl`; used to verify custom `key=`."""
    stem = p.stem
    return int(stem.rsplit("_", 1)[-1])


@pytest.fixture(autouse=True)
def _capture_files_warnings(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="autoresearch.files")


def _warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# ── keep_recent ────────────────────────────────────────────────────────


class TestKeepRecent:
    """Rolling-window file cleanup matching a glob."""

    def test_keeps_n_most_recent_by_mtime_default(self, tmp_path: Path) -> None:
        for i, mtime in enumerate([100.0, 200.0, 300.0, 400.0, 500.0]):
            _touch(tmp_path / f"ckpt_{i}.pkl", mtime=mtime)

        deleted = keep_recent(tmp_path, "ckpt_*.pkl", n=2)

        remaining = sorted(p.name for p in tmp_path.glob("ckpt_*.pkl"))
        assert remaining == ["ckpt_3.pkl", "ckpt_4.pkl"]
        assert sorted(p.name for p in deleted) == ["ckpt_0.pkl", "ckpt_1.pkl", "ckpt_2.pkl"]

    def test_returns_deleted_oldest_first(self, tmp_path: Path) -> None:
        for i, mtime in enumerate([100.0, 200.0, 300.0, 400.0]):
            _touch(tmp_path / f"ckpt_{i}.pkl", mtime=mtime)

        deleted = keep_recent(tmp_path, "ckpt_*.pkl", n=1)

        assert [p.name for p in deleted] == ["ckpt_0.pkl", "ckpt_1.pkl", "ckpt_2.pkl"]

    def test_custom_key_sorts_by_embedded_step_number(self, tmp_path: Path) -> None:
        # Write files in *reverse* step order — mtime would keep wrong set.
        now = time.time()
        _touch(tmp_path / "agent_step_100.pkl", mtime=now - 10)
        _touch(tmp_path / "agent_step_50.pkl", mtime=now - 5)
        _touch(tmp_path / "agent_step_200.pkl", mtime=now - 20)
        _touch(tmp_path / "agent_step_10.pkl", mtime=now)

        keep_recent(tmp_path, "agent_step_*.pkl", n=2, key=_step_key)

        remaining = sorted(p.name for p in tmp_path.glob("agent_step_*.pkl"))
        assert remaining == ["agent_step_100.pkl", "agent_step_200.pkl"]

    def test_on_delete_callback_replaces_default_unlink(self, tmp_path: Path) -> None:
        for i, mtime in enumerate([100.0, 200.0, 300.0]):
            _touch(tmp_path / f"ckpt_{i}.pkl", mtime=mtime)
        seen: list[Path] = []

        keep_recent(tmp_path, "ckpt_*.pkl", n=1, on_delete=seen.append)

        # Files still exist because the callback didn't actually delete them.
        assert sorted(p.name for p in tmp_path.glob("ckpt_*.pkl")) == [
            "ckpt_0.pkl",
            "ckpt_1.pkl",
            "ckpt_2.pkl",
        ]
        assert sorted(p.name for p in seen) == ["ckpt_0.pkl", "ckpt_1.pkl"]

    @pytest.mark.parametrize(
        ("n", "expected_kept"),
        [
            (10, 3),  # n larger than match count: keep everything
            (3, 3),  # n equal to match count: keep everything
        ],
    )
    def test_keeps_all_when_n_at_or_above_match_count(
        self, tmp_path: Path, n: int, expected_kept: int
    ) -> None:
        for i in range(3):
            _touch(tmp_path / f"ckpt_{i}.pkl", mtime=100.0 + i)

        deleted = keep_recent(tmp_path, "ckpt_*.pkl", n=n)

        assert deleted == []
        assert len(list(tmp_path.glob("ckpt_*.pkl"))) == expected_kept

    def test_n_zero_deletes_everything(self, tmp_path: Path) -> None:
        for i in range(3):
            _touch(tmp_path / f"ckpt_{i}.pkl", mtime=100.0 + i)

        deleted = keep_recent(tmp_path, "ckpt_*.pkl", n=0)

        assert sorted(p.name for p in deleted) == ["ckpt_0.pkl", "ckpt_1.pkl", "ckpt_2.pkl"]
        assert list(tmp_path.glob("ckpt_*.pkl")) == []

    def test_missing_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert keep_recent(tmp_path / "does_not_exist", "*.pkl", n=2) == []

    def test_glob_matching_nothing_returns_empty_list(self, tmp_path: Path) -> None:
        _touch(tmp_path / "other.txt", mtime=100.0)

        assert keep_recent(tmp_path, "ckpt_*.pkl", n=2) == []
        # Non-matching file untouched.
        assert (tmp_path / "other.txt").exists()

    def test_negative_n_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n must be"):
            keep_recent(tmp_path, "*.pkl", n=-1)

    def test_accepts_str_directory(self, tmp_path: Path) -> None:
        for i in range(3):
            _touch(tmp_path / f"ckpt_{i}.pkl", mtime=100.0 + i)

        keep_recent(str(tmp_path), "ckpt_*.pkl", n=1)

        assert [p.name for p in tmp_path.glob("ckpt_*.pkl")] == ["ckpt_2.pkl"]


# ── warn_if_tmp_data_dir ──────────────────────────────────────────────


class TestWarnIfTmpDataDir:
    """Heads-up warning when sweep artefacts land on /tmp (tmpfs ENOSPC risk)."""

    @pytest.mark.parametrize(
        "path",
        [
            "/tmp/sweep_outputs",
            "/tmp/nested/deep/checkpoints",
            "/tmp",
        ],
    )
    def test_warns_for_paths_under_tmp(self, path: str, caplog: pytest.LogCaptureFixture) -> None:
        emitted = warn_if_tmp_data_dir(path)
        warnings = _warning_messages(caplog)

        assert emitted is True
        assert len(warnings) == 1
        assert "/tmp" in warnings[0]

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/sweep_outputs",
            "/var/data/checkpoints",
            "/workspace/experiments",
        ],
    )
    def test_silent_for_paths_outside_tmp(
        self, path: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        emitted = warn_if_tmp_data_dir(path)

        assert emitted is False
        assert _warning_messages(caplog) == []

    def test_resolves_symlinks_pointing_into_tmp(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Symlink outside /tmp that points into /tmp — warn() must follow it.
        target = Path("/tmp") / f"resolved_{os.getpid()}"
        target.mkdir(exist_ok=True)
        link = tmp_path / "link_to_tmp"
        link.symlink_to(target)

        try:
            emitted = warn_if_tmp_data_dir(link)
            assert emitted is True
            assert len(_warning_messages(caplog)) == 1
        finally:
            link.unlink(missing_ok=True)
            target.rmdir()

    def test_accepts_pathlike(self, caplog: pytest.LogCaptureFixture) -> None:
        warn_if_tmp_data_dir(Path("/tmp/foo"))
        assert len(_warning_messages(caplog)) == 1
