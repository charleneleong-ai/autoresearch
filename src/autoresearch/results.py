"""Results JSONL I/O for autoresearch sweeps.

Provides a tiny stable interface for reading/writing experiment rows to
`experiments/<TAG>[/<config_name>]/results.jsonl`. Decoupled from any
specific experiment runner — works equally with gemma4-rlvr's training
loop and orak-2025-starter-kit's MACLA loop.

The `config_name` parameter enables per-config sub-results so multiple
parallel sweeps (e.g. gemma vs qwen) can write side-by-side without
trampling each other's JSONL.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def tag_dir(
    experiments_dir: str | Path,
    tag: str | None = None,
    config_name: str | None = None,
) -> Path:
    """Resolve `<experiments_dir>/<tag>[/<config_name>]/`. Creates if needed.

    Layout:
        Flat:        experiments/<TAG>/                    (single sweep)
        Per-config:  experiments/<TAG>/<config_name>/      (multi-sweep)
    """
    d = Path(experiments_dir)
    if tag:
        d = d / tag.lower().replace(" ", "_")
    if config_name:
        d = d / config_name.lower().replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_results(
    experiments_dir: str | Path = "experiments",
    tag: str | None = None,
    config_name: str | None = None,
) -> list[dict[str, Any]]:
    """Load result rows from `<experiments_dir>/<tag>[/<config_name>]/results.jsonl`.

    Returns an empty list if the file doesn't exist yet.
    """
    results_file = tag_dir(experiments_dir, tag, config_name) / "results.jsonl"
    if not results_file.exists():
        return []
    return [
        json.loads(line)
        for line in results_file.read_text().splitlines()
        if line.strip()
    ]


def log_experiment(
    *,
    experiments_dir: str | Path = "experiments",
    tag: str | None = None,
    config_name: str | None = None,
    game: str | None = None,
    score: float = 0.0,
    steps: int = 0,
    status: str = "DISCARD",
    description: str = "",
    wandb_url: str = "",
    notes: str = "",
    game_score: float = 0.0,
    runtime_min: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append a single experiment row to the per-tag (and optional per-config)
    `results.jsonl`.

    The `experiment` index is auto-assigned as `len(existing rows for game)` —
    matches the gemma4-rlvr / orak convention. Pass `extra` for any project-
    specific keys (e.g. `metrics`, `failure_zone`).

    Returns the path to the JSONL file written.
    """
    existing = load_results(experiments_dir, tag, config_name)
    if game is not None:
        per_game = [e for e in existing if e.get("game") == game]
        experiment_num = len(per_game)
    else:
        experiment_num = len(existing)

    entry: dict[str, Any] = {
        "experiment": experiment_num,
        "evaluation_score": score,
        "score": score,  # gemma4-rlvr uses "score"; orak uses "evaluation_score"
        "game_score": game_score,
        "steps": steps,
        "runtime_min": runtime_min,
        "status": status.upper(),
        "description": description,
        "notes": notes,
        "tags": [tag] if tag else [],
        "wandb_url": wandb_url,
        "timestamp": datetime.now().isoformat(),
    }
    if game is not None:
        entry["game"] = game
    if config_name:
        entry["config_name"] = config_name
    if extra:
        entry.update(extra)

    results_file = tag_dir(experiments_dir, tag, config_name) / "results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return results_file
