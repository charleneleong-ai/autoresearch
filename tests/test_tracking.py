"""Run wrapper: disabled mode and failed init are no-ops; blank guarded env values are dropped."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from autoresearch.tracking import Run, load_env


def _failing_init(**_: object) -> None:
    raise RuntimeError("no login")


def test_load_env_drops_blank_guarded_values(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("WANDB_PROJECT", "keep-me")
    load_env(tmp_path / "missing.env")
    assert "WANDB_API_KEY" not in os.environ and "HF_TOKEN" not in os.environ
    assert os.environ["WANDB_PROJECT"] == "keep-me"


def test_disabled_run_is_a_noop(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    run = Run("r", {"a": 1})
    run.log({"x": 1})
    run.summary({"S": 0.5})
    run.log_images({"fig": Path("/nonexistent.png")})
    run.finish()
    assert run.run is None and run.url == ""


def test_failed_init_degrades_to_noop(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    # a stand-in module, so the test runs with or without the optional wandb extra installed
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=_failing_init))
    run = Run("r", {})
    assert run.run is None and run.url == ""
