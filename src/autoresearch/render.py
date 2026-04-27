"""Standalone matplotlib renderer for autoresearch progress PNGs.

Reads `experiments/<TAG>[/<config_name>]/results.jsonl` and produces a
static PNG mirroring the Plotly chart's visual encoding (status colour +
best-run halo + kill_reason inline). Runs without Plotly + kaleido + Chrome
— a clean fallback for headless CI / minimal environments where the
browser-based Plotly export is too heavy.

Single-game and multi-game (per-row `game` field) layouts both supported:
- Single-game: one axis spanning the full figure
- Multi-game: vertically stacked subplots, one per `game`

Adapt `_STATUS_STYLE` / `_kill_tag` if your project uses a different palette
or triage vocabulary.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import typer
from rich import print as rprint

from autoresearch.results import load_results, tag_dir

_STATUS_STYLE = {
    "DISCARD":    {"color": "#cccccc", "line_color": "#999",    "text_color": "#777"},
    "KEEP":       {"color": "#2ecc71", "line_color": "black",   "text_color": "#1a7a3a"},
    "BASELINE":   {"color": "#2ecc71", "line_color": "black",   "text_color": "#1a7a3a"},
    "RUNNING":    {"color": "#f1c40f", "line_color": "#9a7d0a", "text_color": "#7d6608"},
    "EARLY_KILL": {"color": "#7f8c8d", "line_color": "#34495e", "text_color": "#34495e"},
    "CRASH":      {"color": "#e74c3c", "line_color": "#922b21", "text_color": "#922b21"},
}


def _kill_tag(kill_reason: str) -> str:
    """Map a long triage reason to a short category for the inline label.

    Recognises the patterns used by gemma4-rlvr (KL/loss divergence, GPU
    spike) and orak (score plateau, baseline gate, iter timeout). Override
    by passing your own `kill_tag_fn` to `render`.
    """
    kr = (kill_reason or "").lower()
    if "kl" in kr and ("divergence" in kr or "policy" in kr):
        m = re.search(r"\|kl\|=([\d.]+)", kr)
        return f"killed: kl={m.group(1)} (policy)" if m else "killed: policy divergence"
    if "loss" in kr and ("divergence" in kr or "blow" in kr):
        m = re.search(r"\|loss\|=([\d.]+)", kr)
        return f"killed: |loss|={m.group(1)}" if m else "killed: loss blow-up"
    if "step_time" in kr or "spike" in kr:
        m = re.search(r"([\d.]+)s", kr)
        return f"killed: {m.group(1)}s/step" if m else "killed: GPU slow"
    if "plateau" in kr:
        m = re.search(r"\(([\d.]+)%\)", kr)
        return f"killed: plateau {m.group(1)}%" if m else "killed: plateau"
    if "no improvement" in kr or "no_learn" in kr or "no learning" in kr or "no reward" in kr:
        return "killed: no learning"
    if "below baseline" in kr or "baseline gate" in kr or "baseline" in kr:
        return "killed: below baseline"
    if "timeout" in kr:
        return "killed: iter timeout"
    return f"killed: {kill_reason[:30]}" if kill_reason else "killed early"


def _draw_axis(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    score_label: str,
    title: str,
) -> None:
    """Draw one experiment timeline onto the given matplotlib axis."""
    if not rows:
        ax.set_title(f"{title} — no results yet", fontsize=11, color="#999")
        ax.set_axis_off()
        return

    def score(r):
        return r.get(score_field, r.get("score", r.get("evaluation_score", 0)))
    best_exp = max(rows, key=score).get("experiment", 0)

    for r in rows:
        cfg = _STATUS_STYLE.get(r.get("status", "DISCARD"), _STATUS_STYLE["DISCARD"])
        is_best = r.get("experiment") == best_exp
        ax.scatter(
            r.get("experiment", 0), score(r),
            s=400 if is_best else 220,
            c=cfg["color"],
            edgecolors="#27ae60" if is_best else cfg["line_color"],
            linewidths=3 if is_best else 1.2,
            zorder=3,
        )

    kept = [r for r in rows if r.get("status") in ("KEEP", "BASELINE")]
    if kept:
        xs = [r["experiment"] for r in kept]
        ys, best = [], float("-inf")
        for r in kept:
            best = max(best, score(r))
            ys.append(best)
        ax.step(xs, ys, where="post", color="#27ae60", lw=2, alpha=0.6, zorder=2)

    for j, r in enumerate(rows):
        cfg = _STATUS_STYLE.get(r.get("status", "DISCARD"), _STATUS_STYLE["DISCARD"])
        is_best = r.get("experiment") == best_exp
        m = r.get("metrics") or {}
        status = r.get("status", "DISCARD")
        if status == "EARLY_KILL":
            tag = _kill_tag(m.get("kill_reason", "") or r.get("notes", ""))
        elif status == "CRASH" and m.get("crash_reason"):
            cr = m["crash_reason"]
            tag = f"crashed: {cr[:30]}{'…' if len(cr) > 30 else ''}"
        else:
            tag = status.lower()
        runtime = f"{int(r.get('runtime_min', 0))}min" if r.get("runtime_min") else ""
        head = f"E{r.get('experiment', '?')} · {runtime} · {tag}".strip(" ·").replace("·  ·", "·")
        bits = [f"{score_field}={score(r):.2f}"]
        if isinstance(m.get("final_kl"), (int, float)):
            bits.append(f"kl={m['final_kl']:.2f}")
        if r.get("steps"):
            bits.append(f"{r['steps']}st")
        text = f"{head}\n{' · '.join(bits)}"

        y_off = 1.6 if j % 2 == 0 else -1.8
        ax.annotate(
            text, xy=(r.get("experiment", 0), score(r)),
            xytext=(0, y_off * 18), textcoords="offset points",
            ha="center", va="center",
            fontsize=9 if not is_best else 10,
            fontweight="bold" if is_best else "normal",
            color=("#1a7a3a" if is_best else cfg["text_color"]),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=("#f0fff0" if is_best else "white"),
                edgecolor=("#27ae60" if is_best else cfg["color"]),
                linewidth=2 if is_best else 1,
            ),
        )

    n = len(rows)
    n_kept = sum(1 for r in rows if r.get("status") in ("KEEP", "BASELINE"))
    n_killed = sum(1 for r in rows if r.get("status") == "EARLY_KILL")
    n_crash = sum(1 for r in rows if r.get("status") == "CRASH")
    runtime_total = sum(r.get("runtime_min", 0) for r in rows)
    ax.set_title(
        f"{title} — {n} exp · {n_kept} kept · {n_killed} killed · "
        f"{n_crash} crashed · {runtime_total:.0f}min",
        fontsize=12, color="#222",
    )
    ax.set_xlabel("Experiment #", fontsize=10)
    ax.set_ylabel(score_label, fontsize=10)
    ax.grid(True, color="#eee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, n))
    ax.set_xlim(-0.5, n - 0.5)
    ymin = min(score(r) for r in rows) - 2
    ymax = max(score(r) for r in rows) + 3
    ax.set_ylim(ymin, ymax)


def render(
    *,
    experiments_dir: str | Path = "experiments",
    tag: str | None = None,
    config_name: str | None = None,
    out: str | Path | None = None,
    title: str = "Autoresearch progress",
    score_field: str = "score",
    score_label: str = "Reward (higher is better)",
) -> Path:
    """Render `progress.png` from `<experiments_dir>/<tag>[/<config_name>]/results.jsonl`.

    Auto-detects multi-game layout: if every row has a `game` field and there
    are multiple distinct games, renders vertically stacked subplots. Otherwise
    a single axis.

    Returns the output path.
    """
    rows = load_results(experiments_dir, tag, config_name)
    if not rows:
        raise SystemExit(
            f"no results to plot at {tag_dir(experiments_dir, tag, config_name)}/results.jsonl"
        )

    if out is None:
        out = tag_dir(experiments_dir, tag, config_name) / "progress.png"
    out = Path(out)

    games = sorted({r.get("game") for r in rows if r.get("game")})
    multi_game = len(games) > 1

    title_suffix = f" — {config_name}" if config_name else ""
    full_title = f"{title}{title_suffix}"

    if multi_game:
        fig, axes = plt.subplots(
            len(games), 1, figsize=(14, 5 * len(games)), dpi=140, squeeze=False
        )
        fig.patch.set_facecolor("white")
        for idx, game in enumerate(games):
            game_rows = sorted(
                (r for r in rows if r.get("game") == game),
                key=lambda r: r.get("experiment", 0),
            )
            _draw_axis(
                axes[idx][0],
                game_rows,
                score_field=score_field,
                score_label=score_label,
                title=game.replace("_", " ").title(),
            )
        fig.suptitle(full_title, fontsize=15, color="#222", y=0.995)
    else:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=140)
        fig.patch.set_facecolor("white")
        rows = sorted(rows, key=lambda r: r.get("experiment", 0))
        _draw_axis(
            ax, rows,
            score_field=score_field,
            score_label=score_label,
            title=full_title,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main(
    experiments_dir: Path = typer.Option(Path("experiments"), "--experiments-dir"),
    tag: str | None = typer.Option(None, "--tag"),
    config_name: str | None = typer.Option(
        None, "--config", help="Per-config sub-dir for multi-sweep isolation"
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output PNG path. Defaults to <tag-dir>/progress.png"
    ),
    title: str = typer.Option("Autoresearch progress", "--title"),
    score_field: str = typer.Option(
        "score", "--score-field",
        help="JSONL field to plot on y-axis (e.g. 'score' or 'evaluation_score')",
    ),
    score_label: str = typer.Option("Reward (higher is better)", "--score-label"),
) -> None:
    out_path = render(
        experiments_dir=experiments_dir,
        tag=tag,
        config_name=config_name,
        out=out,
        title=title,
        score_field=score_field,
        score_label=score_label,
    )
    rprint(f"[green]wrote[/green] {out_path}  ([dim]{out_path.stat().st_size // 1024} KB[/dim])")


def cli() -> None:
    """Entry-point wrapper so the console script (`autoresearch-render`)
    runs `main` through typer's argument parser."""
    typer.run(main)


if __name__ == "__main__":
    cli()
