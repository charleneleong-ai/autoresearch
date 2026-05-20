"""Filesystem helpers for sweep-side housekeeping.

`keep_recent` is a rolling-window cleanup for any directory of glob-matching
artefacts (checkpoints, replay buffers, eval rollouts). `warn_if_tmp_data_dir`
flags long-running sweeps that are about to fill a tmpfs-backed /tmp.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TMP_ROOT = Path("/tmp").resolve()


def _mtime_key(p: Path) -> float:
    return p.stat().st_mtime


def _default_unlink(p: Path) -> None:
    p.unlink(missing_ok=True)


def keep_recent(
    directory: Path | str,
    glob: str,
    n: int,
    *,
    key: Callable[[Path], Any] | None = None,
    on_delete: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Keep the `n` most-recent files matching `directory.glob(glob)`; remove the rest.

    `key` ranks each file (higher = more recent). Defaults to `Path.stat().st_mtime`;
    pass a custom callable when filenames embed a step number, since mtime gets
    clobbered by rsync/cp -a and by filesystem remounts.

    `on_delete` is called once per file to be removed (oldest first); defaults to
    `Path.unlink(missing_ok=True)`. Pass a callback to also clean up sidecar files
    (e.g. a `.json` summary next to each `.pkl`).

    Returns the paths removed, oldest first.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    directory = Path(directory)
    if not directory.is_dir():
        return []

    matches = list(directory.glob(glob))
    if len(matches) <= n:
        return []

    matches.sort(key=key or _mtime_key)
    to_remove = matches[: len(matches) - n]

    remove = on_delete or _default_unlink
    for path in to_remove:
        remove(path)

    return to_remove


def warn_if_tmp_data_dir(path: Path | str) -> bool:
    """Log a warning if `path` resolves under /tmp; return True if emitted.

    /tmp is often tmpfs-backed (capped at a fraction of RAM). Long-running sweeps
    that write checkpoints / rollouts there hit ENOSPC mid-run, then crash with a
    pickle-write failure that masks the real cause.
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(_TMP_ROOT):
        logger.warning(
            "Data directory %s resolves under /tmp — tmpfs ENOSPC will kill long sweeps. "
            "Move it to persistent storage.",
            resolved,
        )
        return True
    return False
