"""autoresearch — self-driving experiment sweep loop with live PR-updating chart."""

from __future__ import annotations

__version__ = "0.5.1"

from autoresearch.compare import (
    plot_cross_game_scoreboard,
    plot_multi_tag_overlay,
)
from autoresearch.gpu_monitor import GPUMonitor, GPUSample, GPUSummary
from autoresearch.results import (
    filter_by_game,
    get_score,
    load_results,
    log_experiment,
    tag_dir,
)
from autoresearch.verdict import (
    GameSpec,
    GameVerdict,
    VerdictSpec,
    compute_verdict,
    format_markdown,
    load_spec,
)

__all__ = [
    "__version__",
    "filter_by_game",
    "get_score",
    "load_results",
    "log_experiment",
    "tag_dir",
    "GPUMonitor",
    "GPUSample",
    "GPUSummary",
    "plot_multi_tag_overlay",
    "plot_cross_game_scoreboard",
    "GameSpec",
    "GameVerdict",
    "VerdictSpec",
    "compute_verdict",
    "format_markdown",
    "load_spec",
]
