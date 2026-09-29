"""A W&B run behind one switch, so callers never branch on whether tracking is available.

``WANDB_MODE=disabled`` (or a failed ``wandb.init``) turns every method into a
no-op. Loading a ``.env`` is optional; blank ``WANDB_*`` / ``HF_*`` values are
dropped because an empty ``WANDB_API_KEY=`` shadows a netrc login and makes
``wandb.init`` fail silently.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GUARDED_PREFIXES = ("WANDB_", "HF_")


def load_env(path: Path | None) -> None:
    """Load ``path`` with python-dotenv if present, then drop blank guarded variables."""
    if path is not None and Path(path).exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(path)
        except ImportError:
            logger.debug("python-dotenv not installed; skipping %s", path)
    for key in [k for k, v in os.environ.items() if k.startswith(_GUARDED_PREFIXES) and v == ""]:
        os.environ.pop(key)


class Run:
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        *,
        project: str | None = None,
        entity: str | None = None,
        job_type: str = "train",
        env_file: Path | None = None,
    ) -> None:
        load_env(env_file)
        self.run: Any = None
        if os.environ.get("WANDB_MODE", "").lower() == "disabled":
            return
        try:
            import wandb

            self.run = wandb.init(
                project=project or os.environ.get("WANDB_PROJECT"),
                entity=entity or os.environ.get("WANDB_ENTITY") or None,
                name=name,
                job_type=job_type,
                config=config,
                reinit=True,
            )
        except Exception as exc:  # offline box, missing login: keep running without tracking
            logger.warning("wandb disabled: %s", exc)

    @property
    def url(self) -> str:
        return getattr(self.run, "url", "") or ""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_images(self, paths: dict[str, Path]) -> None:
        if self.run is not None:
            import wandb

            self.run.log({k: wandb.Image(str(p)) for k, p in paths.items()})

    def summary(self, metrics: dict[str, Any]) -> None:
        if self.run is not None:
            for k, v in metrics.items():
                if isinstance(v, int | float | str | bool):
                    self.run.summary[k] = v

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
