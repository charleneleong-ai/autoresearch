"""Token-confidence diagnostic for per-row eval dumps with logprobs.

Consumes a per-row JSONL whose rows carry a ``logprobs`` field of shape
``logprobs[arm][step] = [(token_id, decoded_str, logprob), ...]`` (the
schema produced by gemma4-rlvr's ``two_stage_eval.py --save-logprobs N``,
but generic — any project that follows the same shape works).

What it produces, given a per-row dump and a gates dict (e.g.
``{"well_formed": 0.5, "no_hallucinated_facts": 1.0}``):

* :func:`bucket_by_failure` — groups rows by which gates fail. The same
  conjunction-collapse decomposition used in PR-F's pass_all analysis,
  but generic over any rubric.
* :func:`summarize_confidence` — per-row aggregate stats: mean prob,
  mean entropy, % of low-prob tokens, the lowest-prob positions.
* :func:`render_annotated_html` — one HTML page per sample, low-prob
  tokens wrapped in inline ``<span>`` markers. Open in a browser to read
  the completion with confidence shading.
* :func:`plot_confidence_distribution` — kernel-density of mean prob per
  failure bucket, mirrors the visual style of
  :func:`autoresearch.compare.plot_milestone_progression`.

The headline diagnostic question this answers: when a row fails, does
the model commit to the failing tokens with high confidence (broken
training signal) or sample them at high entropy (broken decoding)? Two
buckets, two different fix paths.

CLI::

    python -m autoresearch.token_confidence summary \
        --per-row data/eval.per_row.jsonl \
        --arm two_stage \
        --gate well_formed=0.5 \
        --gate no_hallucinated_facts=1.0 \
        --out reports/token_confidence/

Writes a markdown summary plus per-bucket HTML samples and a PNG plot.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(help="Token-confidence diagnostic for per-row eval dumps with logprobs.")


@dataclass
class Sample:
    """One per-row dump entry with its logprobs decoded."""

    i: int
    completion: str
    scores: dict[str, float]
    logprobs: list[list[tuple[int, str, float]]]  # [step][rank] -> (id, str, logp)
    sampled_token_ids: list[int]  # the actually-chosen id at each step
    extra: dict[str, Any] = field(default_factory=dict)

    def chosen_logprobs(self) -> list[float]:
        """Per-step logprob of the token actually sampled (top-1 unless absent)."""
        out: list[float] = []
        for step, sid in zip(self.logprobs, self.sampled_token_ids, strict=True):
            top1_id, _, top1_lp = step[0]
            if top1_id == sid:
                out.append(top1_lp)
                continue
            match = next((lp for tid, _, lp in step if tid == sid), None)
            # If the sampled id isn't in top-K, conservatively treat as the
            # min returned logprob — this only happens when LMFE constrained
            # to a token that wasn't the model's natural top-K, which is
            # exactly the diagnostic signal we want surfaced.
            out.append(match if match is not None else step[-1][2])
        return out

    def per_step_entropy(self) -> list[float]:
        """Entropy of the truncated top-K distribution at each step (nats).

        Underestimates true entropy (we only see top-K) but the ranking
        across positions/samples remains meaningful.
        """
        out: list[float] = []
        for step in self.logprobs:
            lps = [lp for _, _, lp in step]
            ps = [math.exp(lp) for lp in lps]
            total = sum(ps)
            if total <= 0:
                out.append(0.0)
                continue
            ps = [p / total for p in ps]
            out.append(-sum(p * math.log(p) for p in ps if p > 0))
        return out


@dataclass
class ConfidenceSummary:
    """Aggregate confidence stats over a single sample."""

    n_tokens: int
    mean_prob: float
    mean_entropy: float
    pct_low_prob: float  # fraction of tokens with chosen_p < low_thresh
    lowest_positions: list[tuple[int, str, float, float]]  # (pos, tok, p, entropy)


def load_per_row_logprobs(
    path: str | Path,
    *,
    arm: str,
) -> list[Sample]:
    """Load a per-row JSONL into :class:`Sample` objects for the named arm.

    Schema expected per row:

    * ``i`` — row index (passed through)
    * ``completions[arm]`` *or* ``completion`` — the generated text
    * ``logprobs[arm]`` *or* ``logprobs`` — list[step] of list[K]
      ``(token_id, decoded_str, logprob)`` tuples
    * ``scores[arm]`` *or* legacy top-level ``arm`` dict — gate scores

    The ``arm``-keyed forms match gemma4-rlvr's per_row.jsonl shape; the
    flat forms work for projects that emit one arm per file.
    """
    out: list[Sample] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        comp = (
            row.get("completions", {}).get(arm) if "completions" in row else row.get("completion")
        )
        scores = row.get("scores", {}).get(arm) if "scores" in row else row.get(arm) or {}
        lp_field = row.get("logprobs")
        lp = lp_field.get(arm) if isinstance(lp_field, dict) else lp_field
        if comp is None or lp is None:
            continue
        steps = [[(int(tid), str(tstr), float(logp)) for tid, tstr, logp in step] for step in lp]
        sampled_ids = [step[0][0] for step in steps]  # default to top-1
        extra = {
            k: v
            for k, v in row.items()
            if k not in {"completions", "completion", "logprobs", "scores"}
        }
        out.append(
            Sample(
                i=int(row.get("i", len(out))),
                completion=str(comp),
                scores={k: float(v) for k, v in scores.items() if isinstance(v, (int, float))},
                logprobs=steps,
                sampled_token_ids=sampled_ids,
                extra=extra,
            )
        )
    return out


def bucket_by_failure(
    samples: Sequence[Sample],
    gates: dict[str, float],
) -> dict[frozenset[str], list[Sample]]:
    """Group samples by which gates they fail.

    A gate ``"k": v`` fails when ``sample.scores[k] < v``. The bucket key
    is the frozenset of failing gate names — empty frozenset = passes
    all gates. Mirrors PR-F's pass_all conjunction-collapse decomposition,
    but takes the gates dict as input rather than hard-coding it.
    """
    buckets: dict[frozenset[str], list[Sample]] = {}
    for s in samples:
        fails = frozenset(k for k, thr in gates.items() if s.scores.get(k, 0.0) < thr - 1e-6)
        buckets.setdefault(fails, []).append(s)
    return buckets


def summarize_confidence(
    sample: Sample,
    *,
    low_thresh: float = 0.3,
    n_lowest: int = 15,
) -> ConfidenceSummary:
    """Per-sample confidence aggregate. ``low_thresh`` = chosen_p threshold below
    which we count a token as 'low confidence'."""
    chosen_lps = sample.chosen_logprobs()
    chosen_ps = [math.exp(lp) for lp in chosen_lps]
    entropies = sample.per_step_entropy()
    if not chosen_ps:
        return ConfidenceSummary(0, 0.0, 0.0, 0.0, [])

    n = len(chosen_ps)
    mean_prob = sum(chosen_ps) / n
    mean_entropy = sum(entropies) / n
    n_low = sum(1 for p in chosen_ps if p < low_thresh)
    pct_low = n_low / n

    indexed = list(enumerate(zip(chosen_ps, entropies, strict=True)))
    indexed.sort(key=lambda x: x[1][0])
    lowest = []
    for pos, (p, h) in indexed[:n_lowest]:
        # Recover the actually-sampled token string via top-K lookup
        sid = sample.sampled_token_ids[pos]
        tok = next((s for tid, s, _ in sample.logprobs[pos] if tid == sid), "?")
        lowest.append((pos, tok, p, h))
    return ConfidenceSummary(n, mean_prob, mean_entropy, pct_low, lowest)


_HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 920px; margin: 24px auto; padding: 0 16px; color: #222; }
h1, h2 { font-weight: 600; }
.completion { white-space: pre-wrap; font-family: "SF Mono", Menlo, Consolas, monospace;
              font-size: 13.5px; line-height: 1.55; background: #fafafa;
              border: 1px solid #ddd; border-radius: 6px; padding: 14px; }
.tok-low { background: rgba(220, 38, 38, 0.18); border-radius: 2px;
           padding: 0 1px; cursor: help; }
.tok-mid { background: rgba(245, 158, 11, 0.16); border-radius: 2px;
           padding: 0 1px; cursor: help; }
table { border-collapse: collapse; margin-top: 12px; }
th, td { padding: 4px 12px; border-bottom: 1px solid #eee; text-align: left; font-size: 13px; }
th { background: #f5f5f5; }
"""


def render_annotated_html(
    sample: Sample,
    *,
    low_thresh: float = 0.3,
    mid_thresh: float = 0.6,
    title: str | None = None,
) -> str:
    """Build a self-contained HTML page with low-confidence tokens highlighted.

    Two-tier shading: ``p < low_thresh`` = red, ``p < mid_thresh`` = amber.
    Each shaded span gets a ``title`` attr so hover reveals the exact prob.
    """
    chosen_lps = sample.chosen_logprobs()
    chosen_ps = [math.exp(lp) for lp in chosen_lps]
    entropies = sample.per_step_entropy()

    # Reconstruct text token-by-token so spans align with displayed glyphs.
    parts: list[str] = []
    for pos, (p, h) in enumerate(zip(chosen_ps, entropies, strict=True)):
        sid = sample.sampled_token_ids[pos]
        tok = next((s for tid, s, _ in sample.logprobs[pos] if tid == sid), "")
        esc = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if p < low_thresh:
            parts.append(f'<span class="tok-low" title="p={p:.3f} H={h:.2f}">{esc}</span>')
        elif p < mid_thresh:
            parts.append(f'<span class="tok-mid" title="p={p:.3f} H={h:.2f}">{esc}</span>')
        else:
            parts.append(esc)

    summary = summarize_confidence(sample, low_thresh=low_thresh)
    title = title or f"sample i={sample.i}"
    rows_html = "\n".join(
        f"<tr><td>{pos}</td><td><code>{tok!r}</code></td><td>{p:.3f}</td><td>{h:.2f}</td></tr>"
        for pos, tok, p, h in summary.lowest_positions
    )
    scores_html = ", ".join(f"<code>{k}</code>={v:.2f}" for k, v in sorted(sample.scores.items()))
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title><style>{_HTML_CSS}</style></head>
<body>
<h1>{title}</h1>
<p><b>scores</b>: {scores_html}<br>
<b>tokens</b>: {summary.n_tokens} | <b>mean p</b>: {summary.mean_prob:.3f} |
<b>mean entropy</b>: {summary.mean_entropy:.2f} nats |
<b>low-prob (<{low_thresh}):</b> {summary.pct_low_prob * 100:.1f}%</p>

<h2>Annotated completion</h2>
<div class="completion">{"".join(parts)}</div>

<h2>{len(summary.lowest_positions)} lowest-prob positions</h2>
<table><tr><th>pos</th><th>token</th><th>p</th><th>entropy</th></tr>{rows_html}</table>
</body></html>"""


def plot_confidence_distribution(
    buckets: dict[frozenset[str], list[Sample]],
    *,
    out_path: str | Path,
    title: str = "Per-row mean confidence by failure bucket",
    bins: int = 30,
) -> Path:
    """Histogram of per-sample mean prob, one stack per failure bucket.

    Reveals whether failing rows are uniformly low-confidence (fixable
    by sampling tweaks) or piled up at high mean prob (model is
    confidently wrong, fix needs retraining).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_items = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    fig, ax = plt.subplots(figsize=(11.0, 5.5), dpi=120)
    palette = plt.get_cmap("tab10").colors
    for idx, (fails, samples) in enumerate(bucket_items):
        if not samples:
            continue
        means = [
            sum(math.exp(lp) for lp in s.chosen_logprobs()) / max(len(s.logprobs), 1)
            for s in samples
        ]
        label = "passes_all" if not fails else "+".join(sorted(fails))
        label = f"{label} (n={len(samples)})"
        ax.hist(
            means,
            bins=np.linspace(0, 1, bins + 1),
            alpha=0.55,
            label=label,
            color=palette[idx % len(palette)],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xlabel("per-row mean prob of chosen tokens")
    ax.set_ylabel("number of rows")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _parse_gate_pairs(gate_args: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for arg in gate_args:
        if "=" not in arg:
            raise typer.BadParameter(f"--gate must be 'name=threshold', got {arg!r}")
        name, thr = arg.split("=", 1)
        out[name.strip()] = float(thr)
    return out


def write_summary_report(
    samples: Sequence[Sample],
    gates: dict[str, float],
    *,
    out_dir: str | Path,
    samples_per_bucket: int = 2,
    low_thresh: float = 0.3,
    title: str = "Token-confidence summary",
) -> dict[str, Path]:
    """Render the full report set: markdown summary + per-bucket HTML samples + distribution PNG.

    Returns a dict of artifact paths so the caller (CLI or another tool)
    can wire them into a PR comment / writeup.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    buckets = bucket_by_failure(samples, gates)

    artifacts: dict[str, Path] = {}
    plot_path = out_dir / "confidence_distribution.png"
    plot_confidence_distribution(buckets, out_path=plot_path, title=title)
    artifacts["plot"] = plot_path

    md = [f"# {title}", ""]
    md.append(f"Source: {len(samples)} rows, {len(buckets)} failure buckets")
    md.append(f"Gates: {', '.join(f'`{k}>={v}`' for k, v in gates.items())}")
    md.append("")
    md.append("## Bucket sizes")
    md.append("")
    md.append("| bucket | n | mean prob | mean entropy |")
    md.append("|---|---:|---:|---:|")
    bucket_items = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    for fails, bsamples in bucket_items:
        if not bsamples:
            continue
        all_ps: list[float] = []
        all_hs: list[float] = []
        for s in bsamples:
            sm = summarize_confidence(s, low_thresh=low_thresh)
            all_ps.append(sm.mean_prob)
            all_hs.append(sm.mean_entropy)
        bucket_label = "passes_all" if not fails else "+".join(sorted(fails))
        md.append(
            f"| `{bucket_label}` | {len(bsamples)} | "
            f"{sum(all_ps) / len(all_ps):.3f} | {sum(all_hs) / len(all_hs):.2f} |"
        )
    md.append("")

    md.append("## Sampled HTML reports")
    md.append("")
    for fails, bsamples in bucket_items:
        if not bsamples:
            continue
        bucket_label = "passes_all" if not fails else "+".join(sorted(fails))
        for s in bsamples[:samples_per_bucket]:
            html_path = samples_dir / f"{bucket_label}__i{s.i}.html"
            html_path.write_text(
                render_annotated_html(
                    s,
                    low_thresh=low_thresh,
                    title=f"{bucket_label} — sample i={s.i}",
                )
            )
            md.append(f"- [`{bucket_label}` i={s.i}]({html_path.relative_to(out_dir)})")
    md.append("")

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md))
    artifacts["summary"] = md_path
    return artifacts


@app.command()
def summary(
    per_row: Path = typer.Option(
        ..., "--per-row", help="Path to per-row JSONL with logprobs field."
    ),
    arm: str = typer.Option(
        "two_stage", "--arm", help="Arm name to load (key into row['logprobs']/['scores'])."
    ),
    gate: list[str] = typer.Option(
        [],
        "--gate",
        help="Repeat per gate, format `name=threshold`. A row 'fails' a gate when "
        "scores[name] < threshold. Determines the failure-bucket grouping.",
    ),
    out: Path = typer.Option(Path("reports/token_confidence"), "--out", help="Output directory."),
    samples_per_bucket: int = typer.Option(
        2, "--samples-per-bucket", help="HTML samples to render per bucket."
    ),
    low_thresh: float = typer.Option(
        0.3, "--low-thresh", help="Threshold below which a token counts as low-confidence."
    ),
    title: str = typer.Option("Token-confidence summary", "--title"),
) -> None:
    """Build the full report set: bucket markdown + per-bucket HTML samples + distribution PNG."""
    if not gate:
        raise typer.BadParameter("at least one --gate name=threshold is required")
    gates = _parse_gate_pairs(gate)
    samples = load_per_row_logprobs(per_row, arm=arm)
    if not samples:
        raise typer.BadParameter(f"no usable rows in {per_row} for arm={arm!r}")
    artifacts = write_summary_report(
        samples,
        gates,
        out_dir=out,
        samples_per_bucket=samples_per_bucket,
        low_thresh=low_thresh,
        title=title,
    )
    typer.echo(f"wrote {artifacts['summary']}")
    typer.echo(f"wrote {artifacts['plot']}")
    typer.echo(f"+ {sum(1 for _ in (out / 'samples').iterdir())} HTML samples in {out / 'samples'}")


def _bucket_size_summary(buckets: dict[frozenset[str], list[Sample]]) -> Counter:
    """Used by tests + CLI alternatives that want just the bucket counts."""
    return Counter({"+".join(sorted(k)) or "passes_all": len(v) for k, v in buckets.items()})


@app.command()
def buckets(
    per_row: Path = typer.Option(..., "--per-row"),
    arm: str = typer.Option("two_stage", "--arm"),
    gate: list[str] = typer.Option([], "--gate", help="Repeat per gate, format `name=threshold`."),
) -> None:
    """Print failure-bucket counts only (cheap, no plots / HTML / IO)."""
    if not gate:
        raise typer.BadParameter("at least one --gate name=threshold is required")
    samples = load_per_row_logprobs(per_row, arm=arm)
    counts = _bucket_size_summary(bucket_by_failure(samples, _parse_gate_pairs(gate)))
    for name, n in counts.most_common():
        typer.echo(f"  {n:>5d}  {name}")


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()


__all__ = [
    "ConfidenceSummary",
    "Sample",
    "bucket_by_failure",
    "load_per_row_logprobs",
    "plot_confidence_distribution",
    "render_annotated_html",
    "summarize_confidence",
    "write_summary_report",
]
