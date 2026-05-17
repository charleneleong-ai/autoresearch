"""autoresearch — self-driving experiment sweep loop with live PR-updating chart."""

from __future__ import annotations

__version__ = "0.23.1"

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
    LOG_FORMATS,
    LogFormat,
    clear_sidecar,
    sidecar,
    write_sidecar,
)
from autoresearch.gpu_monitor import (
    GPUMonitor,
    GPUSample,
    GPUSummary,
    GPUTriage,
    GPUTriageThresholds,
)
from autoresearch.normalization import (
    ScoreNormalizer,
    get_normalizer,
    normalize_score,
    register_normalizer,
)
from autoresearch.prompt_caching import extract_cache_stats
from autoresearch.results import (
    KEEP_STATUSES,
    STATUS_BASELINE,
    STATUS_CRASH,
    STATUS_DISCARD,
    STATUS_EARLY_KILL,
    STATUS_KEEP,
    STATUS_RUNNING,
    decide_status,
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
from autoresearch.retry_utils import (
    ClassifiedError,
    ErrorClass,
    classify,
    jittered_backoff,
    with_retries,
)
from autoresearch.subprocess_utils import (
    CrashPattern,
    crash_reason_from_stdout,
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
from autoresearch.token_confidence import (
    ConfidenceSummary,
    bucket_by_failure,
    load_per_row_logprobs,
    plot_confidence_distribution,
    render_annotated_html,
    summarize_confidence,
    write_summary_report,
)
from autoresearch.token_confidence import (
    Sample as TokenConfidenceSample,
)
from autoresearch.trajectory import (
    ActionSpec,
    DwellSpec,
    IterMetrics,
    MilestoneSpec,
    StepRecord,
    TrajectoryWriter,
    convert_scratchpad_to_think,
    extract_iter_metrics,
    format_recent_history,
    has_incomplete_scratchpad,
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
    # row-status helpers (autoresearch#26)
    "KEEP_STATUSES",
    "STATUS_BASELINE",
    "STATUS_CRASH",
    "STATUS_DISCARD",
    "STATUS_EARLY_KILL",
    "STATUS_KEEP",
    "STATUS_RUNNING",
    "decide_status",
    # sweep runner (autoresearch#20 PR 2)
    "IterOutcome",
    "IterPlan",
    "IterPlanner",
    "ResultExtractor",
    "SweepResult",
    "SweepRunner",
    "TriageMonitor",
    # sweep-loop helpers (autoresearch#20 PR 1)
    "LOG_FORMATS",
    "LogFormat",
    "clear_sidecar",
    "kill_gracefully",
    "sidecar",
    "wait_with_timeout",
    "write_sidecar",
    # subprocess crash classification (autoresearch#26)
    "CrashPattern",
    "crash_reason_from_stdout",
    # GPU monitoring + triage thresholds (autoresearch#26)
    "GPUMonitor",
    "GPUSample",
    "GPUSummary",
    "GPUTriage",
    "GPUTriageThresholds",
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
    # trajectory writer + post-hoc introspection
    "ActionSpec",
    "DwellSpec",
    "IterMetrics",
    "MilestoneSpec",
    "StepRecord",
    "TrajectoryWriter",
    "convert_scratchpad_to_think",
    "extract_iter_metrics",
    "format_recent_history",
    "has_incomplete_scratchpad",
    # llm-utils phase 1 — retry / caching / normalisation
    "ClassifiedError",
    "ErrorClass",
    "ScoreNormalizer",
    "classify",
    "extract_cache_stats",
    "get_normalizer",
    "jittered_backoff",
    "normalize_score",
    "register_normalizer",
    "with_retries",
    # token-confidence diagnostic
    "ConfidenceSummary",
    "TokenConfidenceSample",
    "bucket_by_failure",
    "load_per_row_logprobs",
    "plot_confidence_distribution",
    "render_annotated_html",
    "summarize_confidence",
    "write_summary_report",
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
