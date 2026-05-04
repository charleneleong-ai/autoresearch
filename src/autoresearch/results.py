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
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Final

# gemma4-rlvr writes "score"; orak writes "evaluation_score". Readers should
# accept both — this list is the canonical fallback order.
_SCORE_FIELDS: tuple[str, ...] = ("evaluation_score", "score")


def get_score(row: dict[str, Any], score_field: str | None = None) -> float:
    """Return a row's score, transparently handling the score/evaluation_score alias.

    With ``score_field=None`` (default), tries ``evaluation_score`` then ``score``.
    With ``score_field`` set, that field is tried first, then the canonical chain.
    Returns ``0.0`` if nothing matches.
    """
    if score_field is not None and score_field in row:
        return row[score_field]
    for f in _SCORE_FIELDS:
        if f in row:
            return row[f]
    return 0.0


def filter_by_game(rows: list[dict[str, Any]], game: str | None) -> list[dict[str, Any]]:
    """Return only rows whose ``game`` field matches. ``None`` returns rows unchanged."""
    if not game:
        return rows
    return [r for r in rows if r.get("game") == game]


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
    return [json.loads(line) for line in results_file.read_text().splitlines() if line.strip()]


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
        experiment_num = len(filter_by_game(existing, game))
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


def relabel_last_as_early_kill(
    *,
    experiments_dir: str | Path = "experiments",
    tag: str | None = None,
    config_name: str | None = None,
    kill_reason: str,
    filter_field: str | None = None,
    filter_values: list[str] | tuple[str, ...] | None = None,
    last_n: int = 1,
) -> int:
    """Relabel the most recently appended row(s) as ``status="EARLY_KILL"``.

    Two project-side patterns this covers, both nearly verbatim duplicates
    in orak-2025-starter-kit and gemma4-rlvr's ``experiments/autoresearch.py``:

    * **Single-row relabel** (gemma4-rlvr): one iter writes one row; on kill,
      patch that one row. ``relabel_last_as_early_kill(tag=..., kill_reason=...)``.
    * **Filtered multi-row relabel** (orak): one iter writes one row *per game*
      (multiple rows per iter); on kill, only patch the offender game's row, not
      the bystanders. ``relabel_last_as_early_kill(tag=..., kill_reason=...,
      filter_field="game", filter_values=["pokemon_red"], last_n=N_games)``.

    Parameters
    ----------
    experiments_dir, tag, config_name
        Resolves to the same `results.jsonl` `log_experiment` writes to.
    kill_reason
        Free-form string. Stored as ``"KILLED: <reason>. "`` prepended to ``notes``.
    filter_field, filter_values
        If both set, restrict patching to rows whose ``filter_field`` value is
        in ``filter_values``. Useful when an iter writes multiple rows
        (e.g. one per game) and only some should be marked killed.
    last_n
        Inspect the last ``last_n`` rows of the JSONL. Defaults to 1
        (single-row case).

    Returns
    -------
    int
        Number of rows actually relabelled (0 if file missing, nothing matched,
        or no rows in the file).

    Notes
    -----
    Idempotent — re-running with the same ``kill_reason`` will keep prepending
    "KILLED: ..." prefixes. Callers that retry should de-dupe on their side.
    """
    results_file = tag_dir(experiments_dir, tag, config_name) / "results.jsonl"
    if not results_file.exists():
        return 0
    lines = [ln for ln in results_file.read_text().splitlines() if ln.strip()]
    if not lines:
        return 0

    # Window of candidate row indices (most recent first).
    candidate_indices = list(range(max(0, len(lines) - last_n), len(lines)))
    patched = 0
    for idx in candidate_indices:
        try:
            entry = json.loads(lines[idx])
        except json.JSONDecodeError:
            continue
        if filter_field is not None and filter_values is not None:
            if entry.get(filter_field) not in filter_values:
                continue
        entry["status"] = "EARLY_KILL"
        entry["notes"] = f"KILLED: {kill_reason}. " + str(entry.get("notes", ""))
        lines[idx] = json.dumps(entry)
        patched += 1
    if patched:
        results_file.write_text("\n".join(lines) + "\n")
    return patched


# ── kill-reason categorisation ─────────────────────────────────────────

# Public category codes. Stable strings so caller code can switch on them
# without depending on an Enum identity.
KILL_POLICY_DIVERGENCE: Final = "policy_divergence"
KILL_LOSS_BLOWUP: Final = "loss_blowup"
KILL_GPU_SPIKE: Final = "gpu_spike"
KILL_GPU_SLOW: Final = "gpu_slow"
KILL_GPU_HANG: Final = "gpu_hang"
KILL_GPU_WASTED: Final = "gpu_wasted"
KILL_GPU_UNDERSIZED: Final = "gpu_undersized"
KILL_NO_LEARNING: Final = "no_learning"
KILL_UNKNOWN: Final = "unknown"

KILL_CATEGORIES: Final[tuple[str, ...]] = (
    KILL_POLICY_DIVERGENCE,
    KILL_LOSS_BLOWUP,
    KILL_GPU_SPIKE,
    KILL_GPU_SLOW,
    KILL_GPU_HANG,
    KILL_GPU_WASTED,
    KILL_GPU_UNDERSIZED,
    KILL_NO_LEARNING,
    KILL_UNKNOWN,
)

_KL_NUM_RE = re.compile(r"\|kl\|=([\d.]+)")
_LOSS_NUM_RE = re.compile(r"\|loss\|=([\d.]+)")
_SPIKE_NUM_RE = re.compile(r"spike ([\d.]+)s")
_SLOW_NUM_RE = re.compile(r"= ?([\d.]+)s")


# Type alias for project-supplied classifiers passed to
# `categorize_kill_reason(extra_classifier=...)`. Receives the lowercased
# reason; returns a (category, extras) tuple to win, or None to fall through
# to the builtin patterns.
KillClassifier = Callable[[str], tuple[str, dict[str, str]] | None]


def categorize_kill_reason(
    reason: str | None,
    *,
    extra_classifier: KillClassifier | None = None,
) -> tuple[str, dict[str, str]]:
    """Pattern-match a free-form triage ``kill_reason`` into a stable category
    plus any extracted numeric extras.

    The categoriser captures the union of patterns that sweep loops in
    gemma4-rlvr and orak-2025-starter-kit emit when killing a run for triage,
    GPU, or learning-rate reasons. Output is a tuple so each project can
    format its own short label string from the same classification.

    Parameters
    ----------
    reason
        The free-form kill-reason string. ``None`` or empty returns
        ``(KILL_UNKNOWN, {})``.
    extra_classifier
        Optional project-supplied callable that runs *before* the builtin
        patterns. Receives the **lowercased** reason; returns a ``(category,
        extras)`` tuple to win the classification, or ``None`` to fall
        through to the builtin GPU/loss/policy patterns. Lets a project
        register custom failure modes (eg. ``"tokenizer_race"``,
        ``"wandb_throttle"``) without forking the package. Empty/``None``
        ``reason`` short-circuits to ``KILL_UNKNOWN`` *before* the
        ``extra_classifier`` runs, so callbacks never see empty input.

    Returns
    -------
    tuple[str, dict[str, str]]
        ``(category, extras)``. ``category`` is one of ``KILL_CATEGORIES`` or
        any string the ``extra_classifier`` returned; ``extras`` carries any
        numeric value the pattern surfaced (eg. ``{"kl": "0.5"}`` for policy
        divergence, ``{"step_time": "210.5"}`` for a GPU spike). Empty dict
        when the pattern matched without a numeric, or for ``KILL_UNKNOWN``.

    Examples
    --------
    Builtin patterns:

    >>> categorize_kill_reason("|kl|=0.7 suggests policy divergence")
    ('policy_divergence', {'kl': '0.7'})
    >>> categorize_kill_reason("step_time spike 210.5s on step 4")
    ('gpu_spike', {'step_time': '210.5'})
    >>> categorize_kill_reason("")
    ('unknown', {})

    Project-specific extension:

    >>> import re
    >>> def my_extra(kr: str) -> tuple[str, dict[str, str]] | None:
    ...     if "tokenizer race" in kr:
    ...         return "tokenizer_race", {}
    ...     if "wandb 429" in kr:
    ...         m = re.search(r"retry-after=([\\d.]+)", kr)
    ...         return "wandb_throttle", ({"retry_after": m.group(1)} if m else {})
    ...     return None  # fall through to builtins
    >>> categorize_kill_reason("tokenizer race detected", extra_classifier=my_extra)
    ('tokenizer_race', {})
    >>> categorize_kill_reason(
    ...     "wandb 429 retry-after=30.0", extra_classifier=my_extra
    ... )
    ('wandb_throttle', {'retry_after': '30.0'})
    >>> categorize_kill_reason(
    ...     "|kl|=0.7 suggests policy divergence", extra_classifier=my_extra
    ... )
    ('policy_divergence', {'kl': '0.7'})
    """
    if not reason:
        return KILL_UNKNOWN, {}
    kr = reason.lower()

    if extra_classifier is not None:
        result = extra_classifier(kr)
        if result is not None:
            return result

    if "kl" in kr and ("policy" in kr or "divergence" in kr):
        m = _KL_NUM_RE.search(kr)
        return KILL_POLICY_DIVERGENCE, ({"kl": m.group(1)} if m else {})
    if "loss" in kr and ("divergence" in kr or "blow" in kr):
        m = _LOSS_NUM_RE.search(kr)
        return KILL_LOSS_BLOWUP, ({"loss": m.group(1)} if m else {})
    if "step_time" in kr and "spike" in kr:
        m = _SPIKE_NUM_RE.search(kr)
        return KILL_GPU_SPIKE, ({"step_time": m.group(1)} if m else {})
    if "hang" in kr:
        return KILL_GPU_HANG, {}
    if "step_time" in kr or "slow" in kr:
        m = _SLOW_NUM_RE.search(kr)
        return KILL_GPU_SLOW, ({"step_time": m.group(1)} if m else {})
    if "wasted compute" in kr or "underutil" in kr:
        return KILL_GPU_WASTED, {}
    if "undersized" in kr or ("peak" in kr and "mem" in kr):
        return KILL_GPU_UNDERSIZED, {}
    if "no reward" in kr or "baseline" in kr:
        return KILL_NO_LEARNING, {}
    return KILL_UNKNOWN, {}
