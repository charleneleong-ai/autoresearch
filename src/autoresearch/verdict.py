"""Cross-tag ablation verdict — computes per-game deltas, posts to PR, optional poll-and-wait.

When you run two or more autoresearch sweeps to test whether a feature helps
(e.g. baseline vs feature-on vs feature-on+extra), you want a tight verdict:
"Δ vs comparison: +14% — HELPS." This module produces that table from the
existing `results.jsonl` files, optionally posts it as a PR comment via `gh`,
and optionally polls until all treatment sweeps have reached a target iter
count before computing.

Spec format (yaml):

    # threshold for HELPS / NEUTRAL / REGRESSES classification on (treatment - comparison)
    threshold_pct: 10
    # optional; passed through to results.load_results for per-config sub-layout
    config_name: gemma
    # display labels for the three roles; comparison is optional
    labels:
      baseline: "Stage A"
      comparison: "Stage C (vmem)"
      treatment: "Stage D (vmem+planner)"
    # one entry per game/panel
    games:
      - name: twenty_fourty_eight    # filter rows where row['game'] == name
        display: "2048"               # row label in the markdown table
        baseline: harness_check
        comparison: cognitive_check_v2
        treatment: stage_d_ablation_2048
      - name: super_mario
        display: mario
        baseline: harness_check
        comparison: mario_check
        treatment: stage_d_ablation_mario

CLI:

    autoresearch-verdict --spec verdict_spec.yaml --experiments-dir experiments

With wait + post:

    autoresearch-verdict \\
        --spec verdict_spec.yaml \\
        --wait-iters 2 \\
        --poll-s 300 \\
        --max-wait-s 14400 \\
        --post-pr 28 \\
        --repo charleneleong-ai/orak-2025-starter-kit

Verdict classification (on treatment vs comparison; falls back to vs baseline if
comparison is absent):
* ``HELPS``     — delta >= +threshold_pct
* ``REGRESSES`` — delta <= -threshold_pct
* ``NEUTRAL``   — |delta| < threshold_pct
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml
from rich import print as rprint

from autoresearch.results import filter_by_game, get_score, load_results

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ── data shapes ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GameSpec:
    name: str  # game-field value to filter on
    display: str  # display label for the row
    baseline: str  # tag for baseline sweep
    treatment: str  # tag for treatment sweep
    comparison: str | None  # optional tag for intermediate (3-way comparison)


@dataclass(frozen=True)
class VerdictSpec:
    threshold_pct: float
    labels: dict[str, str]  # {"baseline": "Stage A", "comparison": "...", "treatment": "..."}
    games: list[GameSpec]
    config_name: str | None = None


@dataclass
class GameVerdict:
    spec: GameSpec
    baseline_best: float | None
    comparison_best: float | None
    treatment_best: float | None
    baseline_iters: int
    comparison_iters: int
    treatment_iters: int

    @property
    def delta_vs_comparison_pct(self) -> float | None:
        if self.treatment_best is None or self.comparison_best is None:
            return None
        if self.comparison_best <= 0:
            return None
        return (self.treatment_best - self.comparison_best) / self.comparison_best * 100

    @property
    def delta_vs_baseline_pct(self) -> float | None:
        if self.treatment_best is None or self.baseline_best is None:
            return None
        if self.baseline_best <= 0:
            return None
        return (self.treatment_best - self.baseline_best) / self.baseline_best * 100

    def classify(self, threshold_pct: float) -> str:
        # Prefer comparison delta; fall back to baseline delta if no comparison.
        delta = self.delta_vs_comparison_pct
        if delta is None:
            delta = self.delta_vs_baseline_pct
        if delta is None:
            return "?"
        if delta >= threshold_pct:
            return "HELPS"
        if delta <= -threshold_pct:
            return "REGRESSES"
        return "NEUTRAL"


# ── spec loading ───────────────────────────────────────────────────────


def load_spec(spec_path: str | Path) -> VerdictSpec:
    """Parse a verdict spec yaml — fails loud on missing required fields."""
    raw = yaml.safe_load(Path(spec_path).read_text())
    games = [
        GameSpec(
            name=g["name"],
            display=g.get("display", g["name"]),
            baseline=g["baseline"],
            treatment=g["treatment"],
            comparison=g.get("comparison"),
        )
        for g in raw["games"]
    ]
    return VerdictSpec(
        threshold_pct=float(raw.get("threshold_pct", 10)),
        labels=raw.get("labels", {}),
        games=games,
        config_name=raw.get("config_name"),
    )


# ── computation ────────────────────────────────────────────────────────


def _best_for_tag(
    tag: str,
    game_name: str,
    experiments_dir: str | Path,
    config_name: str | None,
) -> tuple[float | None, int]:
    """Return (best_score, iter_count) for `tag` filtered to `game_name`. (None, 0) if nothing."""
    rows = filter_by_game(
        load_results(experiments_dir=experiments_dir, tag=tag, config_name=config_name),
        game_name,
    )
    if not rows:
        return None, 0
    return max(get_score(r) for r in rows), len(rows)


def compute_verdict(
    spec: VerdictSpec,
    experiments_dir: str | Path = "experiments",
) -> list[GameVerdict]:
    """Build a `GameVerdict` per game in the spec — pure function, reads JSONL only."""
    out: list[GameVerdict] = []
    for g in spec.games:
        b_best, b_iters = _best_for_tag(g.baseline, g.name, experiments_dir, spec.config_name)
        t_best, t_iters = _best_for_tag(g.treatment, g.name, experiments_dir, spec.config_name)
        if g.comparison:
            c_best, c_iters = _best_for_tag(g.comparison, g.name, experiments_dir, spec.config_name)
        else:
            c_best, c_iters = None, 0
        out.append(
            GameVerdict(
                spec=g,
                baseline_best=b_best,
                comparison_best=c_best,
                treatment_best=t_best,
                baseline_iters=b_iters,
                comparison_iters=c_iters,
                treatment_iters=t_iters,
            )
        )
    return out


# ── markdown rendering ─────────────────────────────────────────────────


def _fmt_score(best: float | None, iters: int) -> str:
    if best is None:
        return "_(missing)_"
    return f"{best:.2f} _({iters} iters)_"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "?"
    return f"{pct:+.0f}%"


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")


def format_markdown(
    verdicts: list[GameVerdict],
    spec: VerdictSpec,
    *,
    title: str = "Ablation verdict",
    timed_out: bool = False,
    note: str | None = None,
) -> str:
    """Render the verdicts as a single markdown comment body."""
    labels = spec.labels
    label_b = labels.get("baseline", "Baseline")
    label_c = labels.get("comparison", "Comparison")
    label_t = labels.get("treatment", "Treatment")

    has_comparison = any(v.spec.comparison for v in verdicts)

    header_cols = ["Game", label_b]
    if has_comparison:
        header_cols.append(label_c)
    header_cols += [label_t, f"Δ vs {label_c}" if has_comparison else f"Δ vs {label_b}"]
    if has_comparison:
        header_cols.append(f"Δ vs {label_b}")
    header_cols.append("Verdict")

    lines = [
        f"## {title}{' (timed out)' if timed_out else ''}",
        "",
        f"_(automated by `autoresearch-verdict` at {_ts()})_",
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join(["---"] * len(header_cols)) + "|",
    ]

    classifications: set[str] = set()
    for v in verdicts:
        cells = [v.spec.display, _fmt_score(v.baseline_best, v.baseline_iters)]
        if has_comparison:
            cells.append(_fmt_score(v.comparison_best, v.comparison_iters))
        cells.append(_fmt_score(v.treatment_best, v.treatment_iters))
        primary_delta = v.delta_vs_comparison_pct if has_comparison else v.delta_vs_baseline_pct
        cells.append(_fmt_pct(primary_delta))
        if has_comparison:
            cells.append(_fmt_pct(v.delta_vs_baseline_pct))
        verdict = v.classify(spec.threshold_pct)
        classifications.add(verdict)
        cells.append(f"**{verdict}**" if verdict != "?" else "_(no data)_")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(_summary_line(classifications, spec.threshold_pct))
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines) + "\n"


def _summary_line(classifications: set[str], threshold_pct: float) -> str:
    if "HELPS" in classifications and "REGRESSES" in classifications:
        return (
            f"**Mixed**: at least one game HELPS, at least one REGRESSES "
            f"(threshold ±{threshold_pct:.0f}%)."
        )
    if "HELPS" in classifications:
        return f"**Treatment helps** on at least one game (Δ ≥ +{threshold_pct:.0f}%)."
    if "REGRESSES" in classifications:
        return f"**Treatment regresses** on at least one game (Δ ≤ -{threshold_pct:.0f}%)."
    if classifications == {"NEUTRAL"}:
        return f"**Neutral** across all games (|Δ| < {threshold_pct:.0f}%)."
    return "**Insufficient data** for a verdict."


# ── polling ────────────────────────────────────────────────────────────


def _treatments_ready(
    spec: VerdictSpec, experiments_dir: str | Path, target_iters: int
) -> tuple[bool, list[str]]:
    """All treatment tags must have >= target_iters rows for their game.

    Returns (ready, list_of_pending_descriptions_for_logging).
    """
    pending: list[str] = []
    for g in spec.games:
        _, iters = _best_for_tag(g.treatment, g.name, experiments_dir, spec.config_name)
        if iters < target_iters:
            pending.append(f"{g.treatment}/{g.name} ({iters}/{target_iters})")
    return not pending, pending


def wait_for_treatments(
    spec: VerdictSpec,
    experiments_dir: str | Path,
    target_iters: int,
    poll_s: int,
    max_wait_s: int,
) -> bool:
    """Poll until all treatments reach target_iters or max_wait_s elapses.

    Returns True if all ready, False on timeout.
    """
    started = time.time()
    rprint(
        f"[cyan]\\[verdict][/cyan] waiting for {len(spec.games)} treatment sweep(s) "
        f"to reach {target_iters} iters; poll={poll_s}s, max_wait={max_wait_s}s"
    )
    while True:
        ready, pending = _treatments_ready(spec, experiments_dir, target_iters)
        if ready:
            rprint(f"[green]\\[verdict][/green] all treatments reached {target_iters} iters")
            return True
        elapsed = time.time() - started
        if elapsed > max_wait_s:
            rprint(f"[yellow]\\[verdict][/yellow] timeout after {elapsed:.0f}s; pending: {pending}")
            return False
        rprint(f"[cyan]\\[verdict][/cyan] {_ts()} pending: {pending}; sleeping {poll_s}s")
        time.sleep(poll_s)


# ── PR comment posting ─────────────────────────────────────────────────


def post_pr_comment(repo: str, pr: int, body: str) -> bool:
    """Post body as a comment on PR #pr in repo via `gh pr comment`. Returns True on success."""
    proc = subprocess.run(
        ["gh", "pr", "comment", str(pr), "--repo", repo, "--body", body],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        rprint(f"[red]\\[verdict][/red] gh pr comment failed: {proc.stderr.strip()[:300]}")
        return False
    rprint(f"[green]\\[verdict][/green] posted to {repo}#{pr}: {proc.stdout.strip()}")
    return True


# ── CLI ────────────────────────────────────────────────────────────────


@app.command()
def main(
    spec: Path = typer.Option(..., "--spec", help="Path to verdict spec yaml"),
    experiments_dir: Path = typer.Option(Path("experiments"), help="Root experiments directory"),
    title: str = typer.Option("Ablation verdict", help="Title shown above the markdown table"),
    out: Path | None = typer.Option(
        None, "--out", help="Write markdown to this file (in addition to stdout)"
    ),
    wait_iters: int = typer.Option(
        0,
        "--wait-iters",
        help="If > 0, poll until each treatment tag has >= this many iters before computing.",
    ),
    poll_s: int = typer.Option(
        300, "--poll-s", help="Poll interval in seconds (only used if --wait-iters > 0)"
    ),
    max_wait_s: int = typer.Option(
        4 * 60 * 60,
        "--max-wait-s",
        help="Give up + emit timed-out verdict after this many seconds",
    ),
    post_pr: int | None = typer.Option(
        None,
        "--post-pr",
        help="If set, post the verdict as a comment on this PR number via gh",
    ),
    repo: str | None = typer.Option(
        None, "--repo", help="GitHub repo as owner/name (required with --post-pr)"
    ),
) -> None:
    """Compute and emit an ablation verdict from a spec yaml."""
    spec_obj = load_spec(spec)

    timed_out = False
    if wait_iters > 0:
        ready = wait_for_treatments(spec_obj, experiments_dir, wait_iters, poll_s, max_wait_s)
        timed_out = not ready

    verdicts = compute_verdict(spec_obj, experiments_dir)
    body = format_markdown(verdicts, spec_obj, title=title, timed_out=timed_out)

    print(body)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        rprint(f"[green]\\[verdict][/green] wrote {out}")

    if post_pr is not None:
        if repo is None:
            rprint("[red]\\[verdict][/red] --post-pr requires --repo owner/name")
            sys.exit(2)
        ok = post_pr_comment(repo, post_pr, body)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    app()
