"""autoresearch — self-driving experiment sweep loop with live PR-updating chart."""

from __future__ import annotations

__version__ = "0.12.0"

from autoresearch.compare import (
    Milestone,
    append_milestone,
    extract_metrics_from_results_jsonl,
    load_milestones_yaml,
    plot_cross_game_scoreboard,
    plot_milestone_progression,
    plot_multi_tag_overlay,
)
from autoresearch.current_run import (
    clear_sidecar,
    sidecar,
    write_sidecar,
)
from autoresearch.gpu_monitor import GPUMonitor, GPUSample, GPUSummary
from autoresearch.results import (
    filter_by_game,
    get_score,
    load_results,
    log_experiment,
    relabel_last_as_early_kill,
    tag_dir,
)
from autoresearch.retrospective import (
    BUILTIN_DETECTORS,
    BUILTIN_TRANSFORMS,
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
from autoresearch.subprocess_utils import (
    kill_gracefully,
    wait_with_timeout,
)
from autoresearch.sweep_runner import (
    IterOutcome,
    IterPlan,
    IterPlanner,
    ResultExtractor,
    SweepResult,
    SweepRunner,
    TriageMonitor,
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
    "relabel_last_as_early_kill",
    "tag_dir",
    # sweep runner (autoresearch#20 PR 2)
    "IterOutcome",
    "IterPlan",
    "IterPlanner",
    "ResultExtractor",
    "SweepResult",
    "SweepRunner",
    "TriageMonitor",
    # sweep-loop helpers (autoresearch#20 PR 1)
    "clear_sidecar",
    "kill_gracefully",
    "sidecar",
    "wait_with_timeout",
    "write_sidecar",
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
    "BUILTIN_TRANSFORMS",
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
