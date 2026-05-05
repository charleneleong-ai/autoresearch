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

# Status code constants. Stable strings so caller code can switch on them
# without depending on an Enum identity. Both downstream sweeps use these
# exact values in their `results.jsonl` rows.
STATUS_BASELINE: Final = "BASELINE"
STATUS_KEEP: Final = "KEEP"
STATUS_DISCARD: Final = "DISCARD"
STATUS_EARLY_KILL: Final = "EARLY_KILL"
STATUS_CRASH: Final = "CRASH"
STATUS_RUNNING: Final = "RUNNING"

KEEP_STATUSES: Final[tuple[str, ...]] = (STATUS_KEEP, STATUS_BASELINE)


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


# Type alias for project-supplied score extractors passed to ``decide_status``.
ScoreFn = Callable[[dict[str, Any]], float | None]


def decide_status(
    prior: list[dict[str, Any]],
    score: float,
    *,
    score_fn: ScoreFn | None = None,
    keep_statuses: tuple[str, ...] = KEEP_STATUSES,
) -> str:
    """Classify a new row as BASELINE / KEEP / DISCARD against its history.

    Generalises the inline classifiers in ``gemma4-rlvr`` and
    ``orak-2025-starter-kit``:

    * No prior row has a ``keep_statuses`` status (KEEP/BASELINE by default)
      → the new row is the first comparable point and wins ``BASELINE``.
    * Otherwise compare ``score`` against the best prior comparable score.
      Strictly better wins ``KEEP``; otherwise ``DISCARD``.

    Doesn't classify EARLY_KILL or CRASH — those are caller-driven overrides
    (kill happens before scoring, crash bypasses scoring entirely) and the
    ``SweepRunner`` already handles them via the ``_relabel_target`` path.

    Parameters
    ----------
    prior
        All previously-logged rows for the comparison group (eg. one config,
        one game). Status field defaults to ``""`` if absent.
    score
        The new row's score. Must already be normalised onto the same scale
        as ``score_fn`` returns.
    score_fn
        Per-row score extractor for the prior comparison set. Defaults to
        :func:`get_score` (handles the ``score`` / ``evaluation_score``
        alias). Pass a custom callable to compare on a sub-key, eg.::

            decide_status(prior, score, score_fn=lambda r:
                ((r.get("metrics") or {}).get("heldout") or {}).get("mean_total"))

        Rows where ``score_fn`` returns ``None`` are dropped from the
        comparison set — keeps stale rows that pre-date a metrics rollout
        from blocking new rows. If *all* KEEP/BASELINE rows lack a
        comparable score, returns ``BASELINE`` (treats history as empty).
    keep_statuses
        Which statuses count as "in the comparison set". Defaults to
        ``("KEEP", "BASELINE")``. Override if a project tracks an extra
        admit status (eg. ``"PROMOTED"``).

    Returns
    -------
    str
        One of :data:`STATUS_BASELINE`, :data:`STATUS_KEEP`, :data:`STATUS_DISCARD`.

    Examples
    --------
    >>> decide_status([], 5.0)
    'BASELINE'
    >>> decide_status([{"status": "BASELINE", "score": 5.0}], 7.0)
    'KEEP'
    >>> decide_status([{"status": "BASELINE", "score": 5.0}], 5.0)
    'DISCARD'
    >>> decide_status([{"status": "DISCARD", "score": 99.0}], 1.0)
    'BASELINE'

    Custom score extractor:

    >>> heldout = lambda r: ((r.get("metrics") or {}).get("heldout") or {}).get("mean_total")
    >>> prior = [{"status": "KEEP", "score": 1.0, "metrics": {"heldout": {"mean_total": 12.0}}}]
    >>> decide_status(prior, 13.0, score_fn=heldout)
    'KEEP'
    """
    fn = score_fn or get_score
    kept = [r for r in prior if r.get("status", "") in keep_statuses]
    if not kept:
        return STATUS_BASELINE
    prior_scores = [s for s in (fn(r) for r in kept) if s is not None]
    if not prior_scores:
        return STATUS_BASELINE
    return STATUS_KEEP if score > max(prior_scores) else STATUS_DISCARD


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
KILL_TIMEOUT: Final = "timeout"
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
    KILL_TIMEOUT,
    KILL_UNKNOWN,
)

_KL_NUM_RE = re.compile(r"\|kl\|=([\d.]+)")
_LOSS_NUM_RE = re.compile(r"\|loss\|=([\d.]+)")
_SPIKE_NUM_RE = re.compile(r"spike ([\d.]+)s")
_SLOW_NUM_RE = re.compile(r"= ?([\d.]+)s")
_PLATEAU_RE = re.compile(r"\(([\d.]+)%\)")


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
    if "plateau" in kr:
        m = _PLATEAU_RE.search(kr)
        return KILL_NO_LEARNING, ({"plateau_pct": m.group(1)} if m else {})
    if (
        "no improvement" in kr
        or "no_learn" in kr
        or "no learning" in kr
        or "no reward" in kr
        or "baseline" in kr
    ):
        return KILL_NO_LEARNING, {}
    if "timeout" in kr:
        return KILL_TIMEOUT, {}
    return KILL_UNKNOWN, {}
