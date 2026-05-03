"""autoresearch — self-driving experiment sweep loop with live PR-updating chart."""

from __future__ import annotations

__version__ = "0.9.1"

from autoresearch.compare import (
    Milestone,
    append_milestone,
    extract_metrics_from_results_jsonl,
    load_milestones_yaml,
    plot_cross_game_scoreboard,
    plot_milestone_progression,
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
from autoresearch.retrospective import (
    BUILTIN_DETECTORS,
    FailureDetector,
    Finding,
    IterContext,
    RetrospectiveSpec,
    Severity,
    attach_findings_to_row,
    audit_iter,
    filter_by_severity,
)
from autoresearch.retrospective import (
    load_spec as load_retrospective_spec,
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
    "plot_milestone_progression",
    "Milestone",
    "load_milestones_yaml",
    "append_milestone",
    "extract_metrics_from_results_jsonl",
    "GameSpec",
    "GameVerdict",
    "VerdictSpec",
    "compute_verdict",
    "format_markdown",
    "load_spec",
    # retrospective (autoresearch#16)
    "BUILTIN_DETECTORS",
    "FailureDetector",
    "Finding",
    "IterContext",
    "RetrospectiveSpec",
    "Severity",
    "attach_findings_to_row",
    "audit_iter",
    "filter_by_severity",
    "load_retrospective_spec",
]
