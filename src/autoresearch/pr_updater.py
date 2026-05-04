"""Periodic PR refresher daemon for an autoresearch sweep.

Polls every `--poll-s` seconds and on each tick:

1. Re-renders `experiments/<TAG>[/<config>]/progress.png` from the current
   `results.jsonl` via `autoresearch.render`.
2. If the PNG changed: `git add` + `git commit` + `git push` so the
   embedded image in the PR body refreshes (GitHub serves it via
   `?raw=true`).
3. Re-builds a sweep-narrative table from `results.jsonl` and PATCHes the
   PR body between the two marker comments:

       <!-- SWEEP_NARRATIVE_START -->
       (table goes here)
       <!-- SWEEP_NARRATIVE_END -->

   Both markers must already exist in the PR body (one-time setup).

The narrative refresh is independent of the PNG push — even if no chart
data changed, the table is regenerated to reflect the latest timestamps.

Design notes:
- Uses `subprocess` for `git` / `gh` rather than a Python git lib so the
  daemon stays dependency-light (only `requests` optional for direct API).
- Designed to run detached (`setsid + nohup + disown`) so it survives
  SSH / coding-agent session death — verify `PPID=1` after launch.
- Use `python -u` (or `flush=True` on prints) so the live log isn't
  silently buffered under nohup.

Usage:
    autoresearch-pr-updater \\
      --tag <task> [--config <name>] \\
      --pr <num> --repo <owner/name> --branch <branch>

All git operations run in the project's working tree (`--cwd`, default `.`).
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from autoresearch.render import render
from autoresearch.results import (
    KILL_GPU_HANG,
    KILL_GPU_SLOW,
    KILL_GPU_SPIKE,
    KILL_GPU_UNDERSIZED,
    KILL_GPU_WASTED,
    KILL_LOSS_BLOWUP,
    KILL_NO_LEARNING,
    KILL_POLICY_DIVERGENCE,
    categorize_kill_reason,
    get_score,
    load_results,
    tag_dir,
)

MARKER_START = "<!-- SWEEP_NARRATIVE_START -->"
MARKER_END = "<!-- SWEEP_NARRATIVE_END -->"


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")


_PR_NARRATIVE_LABELS: dict[str, str] = {
    KILL_GPU_SPIKE: "GPU spike",
    KILL_GPU_SLOW: "GPU slow",
    KILL_GPU_HANG: "GPU hang",
    KILL_GPU_WASTED: "GPU wasted",
    KILL_GPU_UNDERSIZED: "GPU undersized",
    KILL_LOSS_BLOWUP: "loss blow-up",
    KILL_NO_LEARNING: "no learning",
}


def _kill_short(kill_reason: str) -> str:
    """Map a long triage reason to a short narrative-table tag.

    Thin formatter over :func:`autoresearch.results.categorize_kill_reason`.
    The PR narrative table favours markdown-formatted values for the policy-
    divergence case (e.g. `` `kl=0.7` (policy) ``); other categories use
    plain-text labels keyed off the categoriser's stable category codes.
    """
    category, extras = categorize_kill_reason(kill_reason)
    if category == KILL_POLICY_DIVERGENCE:
        return f"`kl={extras['kl']}` (policy)" if extras else "policy divergence"
    if category in _PR_NARRATIVE_LABELS:
        return _PR_NARRATIVE_LABELS[category]
    # KILL_UNKNOWN — fall back to a truncated raw reason so the narrative
    # table still surfaces a hint instead of a generic "killed".
    return kill_reason[:40] if kill_reason else "killed"


def _build_narrative(rows: list[dict[str, Any]], score_field: str = "score") -> str:
    """Build the markdown table that lives between the marker comments."""
    if not rows:
        return "_(no results yet)_"

    def score(r):
        return get_score(r, score_field)

    n_kept = sum(1 for r in rows if r["status"] in ("KEEP", "BASELINE"))
    n_killed = sum(1 for r in rows if r["status"] == "EARLY_KILL")
    n_crash = sum(1 for r in rows if r["status"] == "CRASH")
    n_run = sum(1 for r in rows if r["status"] == "RUNNING")
    runtime = sum(r.get("runtime_min", 0) for r in rows)
    best = max(rows, key=score)

    summary = (
        f"_Last refresh: {_ts()}._ "
        f"**{len(rows)}** experiments · {n_kept} kept · {n_killed} killed · {n_crash} crashed"
        + (f" · {n_run} running" if n_run else "")
        + f" · {runtime:.0f}min total · "
        + f"best so far: **{score(best):.2f}** (E{best['experiment']})\n"
    )
    lines = [
        summary,
        "| E | status | score | runtime | notes |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        m = r.get("metrics") or {}
        if r["status"] == "EARLY_KILL":
            tag = f"killed: {_kill_short(m.get('kill_reason', ''))}"
        elif r["status"] == "CRASH":
            cr = m.get("crash_reason") or ""
            tag = f"crashed: {cr[:40]}" if cr else "crashed"
        elif r["status"] == "RUNNING":
            tag = "running"
        else:
            tag = r["status"].lower()
        notes = (r.get("notes") or "").replace("|", "\\|")[:80]
        rt = f"{r.get('runtime_min', 0):.0f}min"
        sc = f"{score(r):.2f}"
        run_id = r.get("wandb_run_id") or ""
        link = f" [↗]({r['wandb_url']})" if r.get("wandb_url") and run_id else ""
        lines.append(f"| E{r['experiment']} | {tag} | {sc} | {rt} | {notes}{link} |")
    return "\n".join(lines)


def _refresh_png(
    experiments_dir: Path, tag: str, config_name: str | None, png_path: Path, score_field: str
) -> bool:
    """Regenerate the PNG via `autoresearch.render`. Returns True if rewritten."""
    before_mtime = png_path.stat().st_mtime if png_path.exists() else -1
    try:
        render(
            experiments_dir=experiments_dir,
            tag=tag,
            config_name=config_name,
            out=png_path,
            score_field=score_field,
        )
    except SystemExit:
        return False  # no results yet
    return png_path.exists() and png_path.stat().st_mtime > before_mtime


def _git_push_png_if_changed(png_path: Path, branch: str, cwd: Path) -> bool:
    """Stage + commit + push the PNG. Returns True if pushed (False if no diff)."""
    subprocess.run(["git", "add", str(png_path)], cwd=str(cwd), check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(cwd))
    if diff.returncode == 0:
        return False
    subprocess.run(
        ["git", "commit", "-m", f"docs: refresh autoresearch screenshot ({_ts()})"],
        cwd=str(cwd),
        check=True,
    )
    push = subprocess.run(
        ["git", "push", "origin", branch],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if push.returncode != 0:
        rprint(f"[red]\\[pr_updater][/red] push failed: {push.stderr.strip()[:200]}")
    return push.returncode == 0


def _patch_pr_body(repo: str, pr: int, narrative: str, cwd: Path) -> bool:
    """Fetch PR body via `gh api`, splice narrative between markers, PATCH back."""
    body_proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}", "--jq", ".body"],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    if body_proc.returncode != 0:
        rprint(f"[red]\\[pr_updater][/red] gh api failed: {body_proc.stderr.strip()[:200]}")
        return False
    body = body_proc.stdout
    if MARKER_START not in body or MARKER_END not in body:
        rprint(
            f"[yellow]\\[pr_updater][/yellow] markers missing in PR #{pr} body — add "
            "<!-- SWEEP_NARRATIVE_START --> and <!-- SWEEP_NARRATIVE_END --> "
            "to the body before launching"
        )
        return False
    pre, _, rest = body.partition(MARKER_START)
    _, _, post = rest.partition(MARKER_END)
    new = pre + MARKER_START + "\n" + narrative + "\n" + MARKER_END + post
    if new == body:
        return False
    payload = json.dumps({"body": new})
    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}", "--method", "PATCH", "--input", "-"],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(cwd),
    )
    if proc.returncode != 0:
        rprint(f"[red]\\[pr_updater][/red] PATCH failed: {proc.stderr.strip()[:200]}")
        return False
    return True


def main(
    tag: str = typer.Option(..., "--tag", help="Top-level task/sweep tag (e.g. 'dd_explainer')"),
    config_name: str | None = typer.Option(
        None, "--config", help="Per-config sub-dir for multi-sweep isolation"
    ),
    pr: int = typer.Option(..., "--pr", help="PR number to PATCH"),
    repo: str = typer.Option(..., "--repo", help='owner/name (e.g. "you/repo")'),
    branch: str = typer.Option(..., "--branch", help="Branch to push the PNG commit to"),
    experiments_dir: Path = typer.Option(
        Path("experiments"),
        "--experiments-dir",
        help="Root dir holding tag/<config>/results.jsonl + progress.png",
    ),
    poll_s: int = typer.Option(
        600,
        "--poll-s",
        envvar="AUTORESEARCH_PR_UPDATER_POLL_S",
        help="Seconds between ticks (default 600 = 10 min)",
    ),
    score_field: str = typer.Option(
        "score",
        "--score-field",
        help="JSONL field to use as the headline score",
    ),
    png_path: Path | None = typer.Option(
        None,
        "--png-path",
        help="Override PNG output path (default: <tag-dir>/progress.png)",
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Working dir for git/gh subprocess calls (project root)",
    ),
) -> None:
    cwd = cwd.resolve()
    if png_path is None:
        # tag_dir handles the optional config_name + creates parent dirs.
        png_path = cwd / tag_dir(experiments_dir, tag, config_name) / "progress.png"
    png_path = png_path.resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)

    rprint(
        f"[bold cyan]\\[pr_updater][/bold cyan] starting — poll every {poll_s}s, "
        f"PR #{pr} on [bold]{repo}[/bold]\n"
        f"  experiments_dir={experiments_dir}  tag={tag}  config={config_name}\n"
        f"  png_path={png_path}"
    )

    while True:
        try:
            png_changed = _refresh_png(experiments_dir, tag, config_name, png_path, score_field)
            pushed = _git_push_png_if_changed(png_path, branch, cwd) if png_changed else False
            rows = load_results(experiments_dir, tag, config_name)
            narrative = _build_narrative(rows, score_field=score_field)
            patched = _patch_pr_body(repo, pr, narrative, cwd)
            png_tag = "[green]✓[/green]" if png_changed else "·"
            push_tag = "[green]✓[/green]" if pushed else "·"
            patch_tag = "[green]✓[/green]" if patched else "·"
            rprint(
                f"\\[pr_updater] {_ts()} — png {png_tag} push {push_tag} "
                f"pr {patch_tag} rows={len(rows)}"
            )
        except Exception as e:
            rprint(f"[red]\\[pr_updater][/red] tick error: {e}")
        time.sleep(poll_s)


def cli() -> None:
    """Entry-point wrapper so the console script (`autoresearch-pr-updater`)
    runs `main` through typer's argument parser."""
    typer.run(main)


if __name__ == "__main__":
    cli()
