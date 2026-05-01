"""autoresearch — self-driving experiment sweep loop with live PR-updating chart."""

from __future__ import annotations

__version__ = "0.0.3"

from autoresearch.compare import (
    plot_cross_game_scoreboard,
    plot_multi_tag_overlay,
)
from autoresearch.gpu_monitor import GPUMonitor, GPUSample, GPUSummary
from autoresearch.results import (
    load_results,
    log_experiment,
    tag_dir,
)

__all__ = [
    "__version__",
    "load_results",
    "log_experiment",
    "tag_dir",
    "GPUMonitor",
    "GPUSample",
    "GPUSummary",
    "plot_multi_tag_overlay",
    "plot_cross_game_scoreboard",
]
