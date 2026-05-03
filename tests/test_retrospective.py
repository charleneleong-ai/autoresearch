"""Tests for autoresearch.retrospective — detector framework + built-ins + CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoresearch.retrospective import (
    BUILTIN_DETECTORS,
    SEVERITY_ORDER,
    Finding,
    IterContext,
    app,
    attach_findings_to_row,
    audit_iter,
    filter_by_severity,
    format_markdown,
    load_spec,
)

# ── Finding / IterContext basics ───────────────────────────────────────


def test_finding_at_least_severity_ladder() -> None:
    f_info = Finding("d", "info", "s", "d", "a")
    f_warn = Finding("d", "warn", "s", "d", "a")
    f_block = Finding("d", "block", "s", "d", "a")
    assert not f_info.at_least("warn")
    assert f_warn.at_least("warn")
    assert f_block.at_least("warn")
    assert f_block.at_least("info")
    assert not f_warn.at_least("block")


def test_filter_by_severity_default_warn() -> None:
    findings = [
        Finding("a", "info", "s", "d", "a"),
        Finding("b", "warn", "s", "d", "a"),
        Finding("c", "block", "s", "d", "a"),
    ]
    result = filter_by_severity(findings)
    assert [f.detector for f in result] == ["b", "c"]


def test_severity_order_is_total() -> None:
    # Sanity: the expected ordering is preserved
    assert SEVERITY_ORDER["info"] < SEVERITY_ORDER["warn"] < SEVERITY_ORDER["block"]


# ── silent_kill ────────────────────────────────────────────────────────


def test_silent_kill_fires_on_kill_status_with_no_traceback(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("Iteration 1 starting\nGame state: idle\nKilled by triage\n")
    row = {"experiment": 5, "status": "EARLY_KILL"}
    finding = BUILTIN_DETECTORS["silent_kill"](IterContext(row, log_path=log))
    assert finding is not None
    assert finding.severity == "warn"
    assert "E5" in finding.summary
    assert "EARLY_KILL" in finding.summary


def test_silent_kill_silent_when_traceback_present(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("Traceback (most recent call last):\n  File ...\nValueError: boom\n")
    row = {"experiment": 5, "status": "EARLY_KILL"}
    assert BUILTIN_DETECTORS["silent_kill"](IterContext(row, log_path=log)) is None


def test_silent_kill_silent_on_keep_status(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("(no traceback)")
    row = {"experiment": 5, "status": "KEEP"}
    assert BUILTIN_DETECTORS["silent_kill"](IterContext(row, log_path=log)) is None


def test_silent_kill_no_log_path_does_not_crash() -> None:
    row = {"experiment": 5, "status": "TIMEOUT"}
    finding = BUILTIN_DETECTORS["silent_kill"](IterContext(row, log_path=None))
    # No log → can't see traceback either way; the detector still fires
    # (warn) because the kill happened and can't be ruled out as a crash.
    assert finding is not None


# ── triage_threshold_mismatch ──────────────────────────────────────────


def test_triage_threshold_mismatch_fires_when_kill_step_below_threshold(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("autoresearch: score plateau (0.00%) for 80 steps\n")
    row = {"experiment": 3, "status": "EARLY_KILL", "steps": 80}
    finding = BUILTIN_DETECTORS["triage_threshold_mismatch"](IterContext(row, log_path=log))
    assert finding is not None
    assert "E3" in finding.summary
    assert "80" in finding.summary  # plateau steps echoed back
    # Suggested-action should advise raising threshold
    assert "TRIAGE_SCORE_PLATEAU_STEPS" in finding.detail


def test_triage_threshold_mismatch_silent_when_iter_ran_long(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("autoresearch: score plateau (0.00%) for 80 steps\n")
    # Iter ran past min_first_score_step (default 100) — kill is "fair", not premature
    row = {"experiment": 3, "status": "EARLY_KILL", "steps": 250}
    assert BUILTIN_DETECTORS["triage_threshold_mismatch"](IterContext(row, log_path=log)) is None


def test_triage_threshold_mismatch_silent_when_no_plateau_kill(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("autoresearch: iteration timeout 30min wall-clock\n")
    row = {"experiment": 3, "status": "EARLY_KILL", "steps": 80}
    # No `score plateau` line in log → different kill cause; this detector stays silent
    assert BUILTIN_DETECTORS["triage_threshold_mismatch"](IterContext(row, log_path=log)) is None


def test_triage_threshold_mismatch_respects_custom_min_first_score(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("score plateau (0.00%) for 50 steps\n")
    row = {"experiment": 3, "status": "EARLY_KILL", "steps": 60}
    # With default min_first_score_step=100, this fires (60 < 100).
    detector = BUILTIN_DETECTORS["triage_threshold_mismatch"]
    assert detector(IterContext(row, log_path=log)) is not None
    # Override to 40 — now 60 >= 40, no fire.
    ctx = IterContext(
        row,
        log_path=log,
        detector_kwargs={"triage_threshold_mismatch": {"min_first_score_step": 40}},
    )
    assert detector(ctx) is None


# ── eval_score_plateau ─────────────────────────────────────────────────


def _row(experiment: int, score: float, status: str = "KEEP") -> dict:
    return {"experiment": experiment, "score": score, "evaluation_score": score, "status": status}


def test_eval_score_plateau_fires_on_flat_history() -> None:
    history = [_row(i, 6.5) for i in range(5)]
    cur = _row(5, 6.5)
    finding = BUILTIN_DETECTORS["eval_score_plateau"](IterContext(cur, history=history))
    assert finding is not None
    assert "E5" in finding.summary
    assert "plateau" in finding.summary.lower()


def test_eval_score_plateau_silent_when_history_too_short() -> None:
    # Default window=5; only 3 prior rows → can't tell yet
    history = [_row(i, 6.5) for i in range(3)]
    cur = _row(3, 6.5)
    assert BUILTIN_DETECTORS["eval_score_plateau"](IterContext(cur, history=history)) is None


def test_eval_score_plateau_silent_when_score_moves() -> None:
    history = [_row(i, 6.5) for i in range(5)]
    cur = _row(5, 9.0)  # big jump from 6.5
    assert BUILTIN_DETECTORS["eval_score_plateau"](IterContext(cur, history=history)) is None


def test_eval_score_plateau_silent_when_history_spread_exceeds_epsilon() -> None:
    # History itself is volatile (3 → 9 → 4 → 8 → 5) — not a plateau
    history = [_row(0, 3.0), _row(1, 9.0), _row(2, 4.0), _row(3, 8.0), _row(4, 5.0)]
    cur = _row(5, 6.0)
    assert BUILTIN_DETECTORS["eval_score_plateau"](IterContext(cur, history=history)) is None


def test_eval_score_plateau_custom_epsilon() -> None:
    history = [_row(i, 6.5) for i in range(5)]
    cur = _row(5, 7.2)  # 0.7 away from plateau median
    detector = BUILTIN_DETECTORS["eval_score_plateau"]
    # Default epsilon=0.5 → does not fire
    assert detector(IterContext(cur, history=history)) is None
    # Loosen epsilon → fires
    ctx = IterContext(
        cur,
        history=history,
        detector_kwargs={"eval_score_plateau": {"epsilon": 1.0}},
    )
    assert detector(ctx) is not None


# ── bucketed_failure ───────────────────────────────────────────────────


def _write_per_row(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "per_row.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_bucketed_failure_fires_on_concentrated_bucket(tmp_path: Path) -> None:
    rows = (
        [{"passed": False, "ground_truth_bucket": "Change in usage"} for _ in range(8)]
        + [{"passed": False, "ground_truth_bucket": "Other A"}]
        + [{"passed": False, "ground_truth_bucket": "Other B"}]
        + [{"passed": True, "ground_truth_bucket": "Change in usage"} for _ in range(20)]
    )
    p = _write_per_row(tmp_path, rows)
    cur = {"experiment": 27}
    finding = BUILTIN_DETECTORS["bucketed_failure"](IterContext(cur, per_row_jsonl_path=p))
    assert finding is not None
    assert "E27" in finding.summary
    assert "Change in usage" in finding.summary
    # Sample row indices echo back into the detail
    assert "i=0" in finding.detail


def test_bucketed_failure_silent_when_failures_below_min(tmp_path: Path) -> None:
    rows = [{"passed": False, "ground_truth_bucket": "X"} for _ in range(3)]
    p = _write_per_row(tmp_path, rows)
    # Default min_failures=5 → does not fire on 3
    assert BUILTIN_DETECTORS["bucketed_failure"](IterContext({}, per_row_jsonl_path=p)) is None


def test_bucketed_failure_silent_when_distribution_is_spread(tmp_path: Path) -> None:
    rows = [
        {"passed": False, "ground_truth_bucket": f"B{i % 5}"} for i in range(20)
    ]  # 4 fails per bucket × 5 buckets — no concentration
    p = _write_per_row(tmp_path, rows)
    assert BUILTIN_DETECTORS["bucketed_failure"](IterContext({}, per_row_jsonl_path=p)) is None


def test_bucketed_failure_respects_custom_bucket_field(tmp_path: Path) -> None:
    rows = [{"passed": False, "label": "A"} for _ in range(8)] + [
        {"passed": False, "label": "B"} for _ in range(2)
    ]
    p = _write_per_row(tmp_path, rows)
    ctx = IterContext(
        {"experiment": 1},
        per_row_jsonl_path=p,
        detector_kwargs={"bucketed_failure": {"bucket_field": "label"}},
    )
    finding = BUILTIN_DETECTORS["bucketed_failure"](ctx)
    assert finding is not None
    assert "label='A'" in finding.summary or "'A'" in finding.summary


def test_bucketed_failure_silent_when_no_per_row_file() -> None:
    # No per_row file → detector cannot run; gracefully returns None
    assert (
        BUILTIN_DETECTORS["bucketed_failure"](
            IterContext({"experiment": 1}, per_row_jsonl_path=None)
        )
        is None
    )


# ── audit_iter / orchestration ─────────────────────────────────────────


def test_audit_iter_runs_all_default_detectors() -> None:
    # Empty inputs; no detector should fire, and no detector should crash
    findings = audit_iter(results_row={"experiment": 0, "status": "KEEP"})
    assert findings == []


def test_audit_iter_collects_multiple_findings(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text("score plateau (0.00%) for 80 steps\n")
    row = {"experiment": 1, "status": "EARLY_KILL", "steps": 80}
    findings = audit_iter(results_row=row, log_path=log)
    names = {f.detector for f in findings}
    # Both silent_kill (no traceback) and triage_threshold_mismatch should fire
    assert "silent_kill" in names
    assert "triage_threshold_mismatch" in names


def test_audit_iter_with_subset_of_detectors() -> None:
    findings = audit_iter(
        results_row={"experiment": 1, "status": "EARLY_KILL"},
        detectors=[BUILTIN_DETECTORS["silent_kill"]],
    )
    assert all(f.detector == "silent_kill" for f in findings)


def test_attach_findings_to_row_writes_summaries() -> None:
    row: dict = {"experiment": 1}
    findings = [Finding("d1", "warn", "summary 1", "detail 1", "fix it")]
    attach_findings_to_row(row, findings)
    assert "retrospective" in row
    assert row["retrospective"]["findings"][0]["detector"] == "d1"
    assert row["retrospective"]["findings"][0]["summary"] == "summary 1"
    # Detail is intentionally omitted from the row — it lives in the .md
    assert "detail" not in row["retrospective"]["findings"][0]


def test_format_markdown_no_findings() -> None:
    md = format_markdown([], iter_id=5)
    assert "E5" in md
    assert "No findings" in md


def test_format_markdown_with_findings() -> None:
    findings = [
        Finding("det1", "warn", "s1", "### det1 (warn)\n\nDetail one.", "act"),
        Finding("det2", "block", "s2", "### det2 (block)\n\nDetail two.", "act"),
    ]
    md = format_markdown(findings, iter_id=27)
    assert "## E27 retrospective" in md
    assert "Detail one." in md
    assert "Detail two." in md


# ── RetrospectiveSpec / YAML loading ───────────────────────────────────


def test_load_spec_minimal(tmp_path: Path) -> None:
    p = tmp_path / "spec.yaml"
    p.write_text(
        "post_iter_retrospective:\n"
        "  enabled: true\n"
        "  detectors:\n"
        "    - silent_kill\n"
        "    - eval_score_plateau\n"
    )
    spec = load_spec(p)
    assert spec.enabled
    assert spec.detectors == ["silent_kill", "eval_score_plateau"]
    assert spec.detector_kwargs == {}


def test_load_spec_accepts_root_or_block(tmp_path: Path) -> None:
    # Block form
    p1 = tmp_path / "block.yaml"
    p1.write_text("post_iter_retrospective:\n  detectors: [silent_kill]\n")
    s1 = load_spec(p1)
    # Root form (no wrapper)
    p2 = tmp_path / "root.yaml"
    p2.write_text("detectors: [silent_kill]\n")
    s2 = load_spec(p2)
    assert s1.detectors == s2.detectors == ["silent_kill"]


def test_spec_selected_detectors_resolves_names(tmp_path: Path) -> None:
    p = tmp_path / "spec.yaml"
    p.write_text("detectors: [silent_kill, eval_score_plateau]\n")
    detectors = load_spec(p).selected_detectors()
    assert [d.name for d in detectors] == ["silent_kill", "eval_score_plateau"]


def test_spec_selected_detectors_unknown_name_raises(tmp_path: Path) -> None:
    p = tmp_path / "spec.yaml"
    p.write_text("detectors: [silent_kill, made_up_detector]\n")
    with pytest.raises(KeyError, match="made_up_detector"):
        load_spec(p).selected_detectors()


def test_spec_action_for_picks_highest_severity_match(tmp_path: Path) -> None:
    p = tmp_path / "spec.yaml"
    p.write_text(
        "detectors: [silent_kill]\n"
        "on_finding:\n"
        "  - severity: warn\n"
        "    action: append_to_next_iter_notes\n"
        "  - severity: block\n"
        "    action: stop_sweep\n"
    )
    spec = load_spec(p)
    assert spec.action_for("info") is None  # no info-level rule
    assert spec.action_for("warn") == "append_to_next_iter_notes"
    assert spec.action_for("block") == "stop_sweep"  # block rule wins for block findings


# ── CLI smoke tests ────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_list_detectors(runner: CliRunner) -> None:
    result = runner.invoke(app, ["list-detectors"])
    assert result.exit_code == 0
    for name in BUILTIN_DETECTORS:
        assert name in result.stdout


def test_cli_audit_latest_iter_no_findings(runner: CliRunner, tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps({"experiment": 0, "status": "KEEP", "score": 5.0}) + "\n")
    result = runner.invoke(app, ["audit", "--results-jsonl", str(results)])
    assert result.exit_code == 0
    assert "no findings" in result.stdout.lower()


def test_cli_audit_writes_md_and_updates_jsonl(runner: CliRunner, tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"
    log.write_text("score plateau (0.00%) for 80 steps\n")
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps({"experiment": 0, "status": "EARLY_KILL", "steps": 80}) + "\n")
    md_out = tmp_path / "out" / "retrospective.md"

    result = runner.invoke(
        app,
        [
            "audit",
            "--results-jsonl",
            str(results),
            "--log",
            str(log),
            "--write-md",
            str(md_out),
            "--write-json",
        ],
    )
    assert result.exit_code == 0
    # Markdown file written + non-trivial
    assert md_out.exists()
    md_text = md_out.read_text()
    assert "## E0 retrospective" in md_text
    # JSONL updated in place to include findings under `retrospective`
    updated = json.loads(results.read_text().strip().splitlines()[0])
    assert "retrospective" in updated
    assert len(updated["retrospective"]["findings"]) >= 1


# ── gradient_collapse ──────────────────────────────────────────────────
#
# fetch_history is monkeypatched in every test so no real wandb calls happen.
# That keeps CI fast + offline + works without the [wandb] extra installed.


def _stub_history(monkeypatch: pytest.MonkeyPatch, series: dict[str, list[float]]) -> None:
    """Replace autoresearch.wandb_history.fetch_history with a fixed return."""
    import autoresearch.wandb_history as mod

    def fake_fetch_history(*, run_url: str, keys: list[str], samples: int = 500) -> dict:
        return {k: series.get(k, []) for k in keys}

    monkeypatch.setattr(mod, "fetch_history", fake_fetch_history)


def test_gradient_collapse_silent_without_wandb_url() -> None:
    row = {"experiment": 1, "status": "KEEP"}
    assert BUILTIN_DETECTORS["gradient_collapse"](IterContext(row)) is None


def test_gradient_collapse_fires_on_loss_zero_and_flat_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_history(
        monkeypatch,
        {
            "train/loss": [0.001] * 60,  # collapsed near zero
            "train/reward": [0.42] * 60,  # perfectly flat → CV = 0
        },
    )
    row = {"experiment": 1, "status": "KEEP", "wandb_url": "charlene/orak/abc"}
    finding = BUILTIN_DETECTORS["gradient_collapse"](IterContext(row))
    assert finding is not None
    assert finding.severity == "block"
    assert "E1" in finding.summary
    assert "train/loss" in finding.summary
    assert "train/reward" in finding.summary


def test_gradient_collapse_silent_when_loss_still_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_history(
        monkeypatch,
        {"train/loss": [0.5] * 60, "train/reward": [0.42] * 60},
    )
    row = {"experiment": 1, "wandb_url": "charlene/orak/abc"}
    assert BUILTIN_DETECTORS["gradient_collapse"](IterContext(row)) is None


def test_gradient_collapse_silent_when_reward_still_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Loss collapsed but reward is volatile → not a collapse signal
    _stub_history(
        monkeypatch,
        {
            "train/loss": [0.001] * 60,
            "train/reward": [0.1, 0.4, 0.7, 0.2, 0.9] * 12,
        },
    )
    row = {"experiment": 1, "wandb_url": "charlene/orak/abc"}
    assert BUILTIN_DETECTORS["gradient_collapse"](IterContext(row)) is None


def test_gradient_collapse_silent_when_history_too_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_history(
        monkeypatch,
        {"train/loss": [0.001] * 10, "train/reward": [0.42] * 10},
    )
    row = {"experiment": 1, "wandb_url": "charlene/orak/abc"}
    # Default window=50; only 10 samples → can't decide
    assert BUILTIN_DETECTORS["gradient_collapse"](IterContext(row)) is None


def test_gradient_collapse_handles_zero_mean_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    """When mean(reward)≈0, CV is undefined; the detector should still fire on
    the collapse pattern (loss ≈ 0 + reward stuck at 0 = no learning)."""
    _stub_history(
        monkeypatch,
        {"train/loss": [0.001] * 60, "train/reward": [0.0] * 60},
    )
    row = {"experiment": 1, "wandb_url": "charlene/orak/abc"}
    finding = BUILTIN_DETECTORS["gradient_collapse"](IterContext(row))
    assert finding is not None
    assert "block" == finding.severity


def test_gradient_collapse_respects_custom_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_history(
        monkeypatch,
        {"train/loss": [0.08] * 60, "train/reward": [0.42] * 60},
    )
    row = {"experiment": 1, "wandb_url": "charlene/orak/abc"}
    detector = BUILTIN_DETECTORS["gradient_collapse"]
    # Default loss_near_zero=0.05 — 0.08 doesn't qualify
    assert detector(IterContext(row)) is None
    # Loosen threshold → fires
    ctx = IterContext(
        row,
        detector_kwargs={"gradient_collapse": {"loss_near_zero_threshold": 0.1}},
    )
    assert detector(ctx) is not None


def test_gradient_collapse_silent_on_wandb_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If fetch_history raises ValueError (bad URL) or RuntimeError (API failure),
    the detector should silently skip — sweep keeps going, no spurious finding."""
    import autoresearch.wandb_history as mod

    def boom(*, run_url: str, keys: list[str], samples: int = 500) -> dict:
        raise RuntimeError("wandb api refused: 401 unauthorized")

    monkeypatch.setattr(mod, "fetch_history", boom)
    row = {"experiment": 1, "wandb_url": "charlene/orak/abc"}
    assert BUILTIN_DETECTORS["gradient_collapse"](IterContext(row)) is None


def test_gradient_collapse_silent_when_wandb_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `from autoresearch.wandb_history import fetch_history` itself raises
    ImportError (e.g. wandb_history transitively depends on wandb in some
    future version), the detector returns None instead of crashing."""
    import builtins
    import sys

    # Simulate an ImportError on the import line in _gradient_collapse by
    # nuking the cached module + replacing __import__ to raise for it.
    sys.modules.pop("autoresearch.wandb_history", None)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "autoresearch.wandb_history":
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    row = {"experiment": 1, "wandb_url": "charlene/orak/abc"}
    assert BUILTIN_DETECTORS["gradient_collapse"](IterContext(row)) is None


# ── registry sanity ────────────────────────────────────────────────────


def test_gradient_collapse_in_registry() -> None:
    assert "gradient_collapse" in BUILTIN_DETECTORS
    assert BUILTIN_DETECTORS["gradient_collapse"].name == "gradient_collapse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
