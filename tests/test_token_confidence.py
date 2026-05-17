"""Tests for autoresearch.token_confidence — per-row logprob diagnostic."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoresearch.token_confidence import (
    Sample,
    app,
    bucket_by_failure,
    load_per_row_logprobs,
    plot_confidence_distribution,
    render_annotated_html,
    summarize_confidence,
    write_summary_report,
)


def _make_logprobs(token_ids: list[int], top1_lps: list[float], distractor_lp: float = -3.0):
    """Build a logprobs[step] = [(id, str, lp)] list with the actual token at top-1."""
    steps = []
    for tid, lp in zip(token_ids, top1_lps, strict=True):
        steps.append(
            [
                (tid, f"tok{tid}", lp),
                (tid + 100, f"tok{tid + 100}", distractor_lp),
            ]
        )
    return steps


@pytest.fixture
def per_row_path(tmp_path: Path) -> Path:
    """Synthetic per-row JSONL with mixed gate failures + logprobs."""
    rows = [
        # Row 0: passes all gates, high mean confidence
        {
            "i": 0,
            "ground_truth_triggers": ["A"],
            "completions": {"two_stage": "tok10tok11tok12"},
            "scores": {"two_stage": {"well_formed": 1.0, "no_halluc": 1.0}},
            "logprobs": {"two_stage": _make_logprobs([10, 11, 12], [-0.05, -0.05, -0.05])},
        },
        # Row 1: fails well_formed only, high confidence (confidently wrong)
        {
            "i": 1,
            "completions": {"two_stage": "tok20tok21tok22"},
            "scores": {"two_stage": {"well_formed": 0.2, "no_halluc": 1.0}},
            "logprobs": {"two_stage": _make_logprobs([20, 21, 22], [-0.05, -0.05, -0.1])},
        },
        # Row 2: fails no_halluc only, low confidence (gambling)
        {
            "i": 2,
            "completions": {"two_stage": "tok30tok31tok32"},
            "scores": {"two_stage": {"well_formed": 1.0, "no_halluc": 0.0}},
            "logprobs": {"two_stage": _make_logprobs([30, 31, 32], [-2.0, -2.5, -3.0])},
        },
        # Row 3: fails both
        {
            "i": 3,
            "completions": {"two_stage": "tok40"},
            "scores": {"two_stage": {"well_formed": 0.0, "no_halluc": 0.0}},
            "logprobs": {"two_stage": _make_logprobs([40], [-1.0])},
        },
        # Row 4: missing logprobs — should be skipped silently
        {
            "i": 4,
            "completions": {"two_stage": "no logprobs"},
            "scores": {"two_stage": {"well_formed": 1.0, "no_halluc": 1.0}},
        },
    ]
    p = tmp_path / "in.per_row.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_load_per_row_logprobs_arm_keyed_schema(per_row_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    # Row 4 missing logprobs is silently skipped
    assert len(samples) == 4
    assert all(isinstance(s, Sample) for s in samples)
    assert samples[0].scores == {"well_formed": 1.0, "no_halluc": 1.0}
    assert samples[0].sampled_token_ids == [10, 11, 12]


def test_load_per_row_logprobs_flat_schema(tmp_path: Path) -> None:
    """Project that emits one arm per file uses flat `completion`/`logprobs` keys."""
    p = tmp_path / "flat.per_row.jsonl"
    p.write_text(
        json.dumps(
            {
                "i": 0,
                "completion": "x",
                "two_stage": {"well_formed": 1.0},  # legacy top-level scores
                "logprobs": _make_logprobs([1, 2], [-0.1, -0.1]),
            }
        )
        + "\n"
    )
    samples = load_per_row_logprobs(p, arm="two_stage")
    assert len(samples) == 1
    assert samples[0].scores == {"well_formed": 1.0}


def test_chosen_logprobs_recovers_sampled_token(per_row_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    s0 = samples[0]
    chosen = s0.chosen_logprobs()
    assert chosen == [-0.05, -0.05, -0.05]
    # exp(-0.05) ≈ 0.95 — passing-row mean prob is high
    assert math.exp(sum(chosen) / len(chosen)) > 0.9


def test_per_step_entropy_bounds(per_row_path: Path) -> None:
    """Entropy is non-negative and bounded by log(K) — sanity check."""
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    for s in samples:
        for h in s.per_step_entropy():
            assert 0.0 <= h <= math.log(2) + 1e-6  # K=2 in fixture


def test_bucket_by_failure_groups_correctly(per_row_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    gates = {"well_formed": 0.5, "no_halluc": 1.0}
    buckets = bucket_by_failure(samples, gates)

    bucket_keys = {tuple(sorted(k)) for k in buckets}
    assert () in bucket_keys  # row 0: passes all
    assert ("well_formed",) in bucket_keys  # row 1
    assert ("no_halluc",) in bucket_keys  # row 2
    assert ("no_halluc", "well_formed") in bucket_keys  # row 3

    assert len(buckets[frozenset()]) == 1
    assert len(buckets[frozenset({"well_formed"})]) == 1
    assert len(buckets[frozenset({"no_halluc"})]) == 1
    assert len(buckets[frozenset({"no_halluc", "well_formed"})]) == 1


def test_summarize_confidence_distinguishes_confident_from_gambling(per_row_path: Path) -> None:
    """The headline diagnostic: confident-wrong rows have high mean prob,
    gambling rows have low mean prob. Verifies the metric separates them."""
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    # Row 1 = confidently wrong (high prob, fails well_formed)
    # Row 2 = gambling (low prob, fails no_halluc)
    confident_wrong = summarize_confidence(samples[1])
    gambling = summarize_confidence(samples[2])
    assert confident_wrong.mean_prob > 0.85
    assert gambling.mean_prob < 0.5
    assert confident_wrong.pct_low_prob == 0.0
    assert gambling.pct_low_prob == 1.0


def test_summarize_confidence_lowest_positions_sorted(per_row_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    sm = summarize_confidence(samples[2], n_lowest=3)
    probs = [p for _pos, _tok, p, _h in sm.lowest_positions]
    assert probs == sorted(probs)  # ascending p


def test_render_annotated_html_marks_low_prob_spans(per_row_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    html = render_annotated_html(samples[2], low_thresh=0.5)  # gambling row
    assert '<span class="tok-low"' in html
    assert "<table" in html
    # All three tokens have p<0.5 → all three spans should be wrapped
    assert html.count('<span class="tok-low"') == 3


def test_render_annotated_html_no_marks_for_confident_row(per_row_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    html = render_annotated_html(samples[0], low_thresh=0.5, mid_thresh=0.7)  # passes_all row
    # exp(-0.05) ≈ 0.95 > 0.7 → no spans should be marked (CSS class defs are fine)
    assert '<span class="tok-low"' not in html
    assert '<span class="tok-mid"' not in html


def test_plot_confidence_distribution_writes_png(per_row_path: Path, tmp_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    buckets = bucket_by_failure(samples, {"well_formed": 0.5, "no_halluc": 1.0})
    out = tmp_path / "dist.png"
    result = plot_confidence_distribution(buckets, out_path=out)
    assert result == out
    assert out.exists() and out.stat().st_size > 1000  # > 1KB rendered PNG


def test_write_summary_report_full_artifact_set(per_row_path: Path, tmp_path: Path) -> None:
    samples = load_per_row_logprobs(per_row_path, arm="two_stage")
    out_dir = tmp_path / "report"
    artifacts = write_summary_report(
        samples,
        {"well_formed": 0.5, "no_halluc": 1.0},
        out_dir=out_dir,
        samples_per_bucket=1,
    )
    assert artifacts["summary"].exists()
    assert artifacts["plot"].exists()
    summary_md = artifacts["summary"].read_text()
    assert "Bucket sizes" in summary_md
    # 4 buckets × 1 sample each = 4 HTML files
    assert len(list((out_dir / "samples").glob("*.html"))) == 4


def test_load_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert load_per_row_logprobs(p, arm="two_stage") == []


def test_cli_summary_smoke(per_row_path: Path, tmp_path: Path) -> None:
    """End-to-end CLI smoke via typer's CliRunner."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "summary",
            "--per-row",
            str(per_row_path),
            "--arm",
            "two_stage",
            "--gate",
            "well_formed=0.5",
            "--gate",
            "no_halluc=1.0",
            "--out",
            str(tmp_path / "rep"),
            "--samples-per-bucket",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "rep" / "summary.md").exists()
    assert (tmp_path / "rep" / "confidence_distribution.png").exists()
