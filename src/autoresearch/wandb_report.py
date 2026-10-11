"""Edit one section of an existing W&B report in place (needs the ``wandb-workspaces`` extra).

A report grows with a sweep: keep the narrative blocks, and replace the
section under one heading with fresh markdown and panels each time results
land. Long tables (e.g. every run) go under collapsed headings placed
right after the section, rewritten on each update. Section starts are
detected as either an ``H2`` block or a ``MarkdownBlock`` whose first line is
``## ...`` (the updater's own previous edit leaves the latter).
"""

from __future__ import annotations

from typing import Any


def _heading_text(block: Any, wr: Any) -> str:
    if isinstance(block, wr.H2):
        return "".join(map(str, block.text))
    return str(getattr(block, "text", "")).split("\n", 1)[0]


def _is_heading(block: Any, wr: Any) -> bool:
    if isinstance(block, wr.H2):
        return True
    return isinstance(block, wr.MarkdownBlock) and block.text.lstrip().startswith("## ")


def section_bounds(blocks: list[Any], heading: str, wr: Any) -> tuple[int, int]:
    """Indices ``[start, end)`` of the section whose heading contains ``heading``."""
    heads = [i for i, b in enumerate(blocks) if _is_heading(b, wr)]
    start = next(i for i in heads if heading in _heading_text(blocks[i], wr))
    end = next((i for i in heads if i > start), len(blocks))
    return start, end


def splice_section(
    blocks: list[Any], heading: str, new_blocks: list[Any], wr: Any, drop: tuple[str, ...] = ()
) -> list[Any]:
    """``blocks`` with the section under ``heading`` replaced by ``new_blocks``; blocks after it
    whose heading is exactly one of ``drop`` (stale collapsed sections) are removed."""
    start, end = section_bounds(blocks, heading, wr)
    rest = [b for b in blocks[end:] if not (_is_heading(b, wr) and _heading_text(b, wr) in drop)]
    return blocks[:start] + new_blocks + rest


def replace_section(
    report_url: str,
    heading: str,
    markdown: str,
    panels: list[Any] | None = None,
    collapsed: dict[str, str] | None = None,
) -> str:
    """Replace the section under ``heading`` with ``markdown`` (restating it) and panels, followed
    by one collapsed heading per ``collapsed`` entry (heading -> markdown)."""
    import wandb_workspaces.reports.v2 as wr

    report = wr.Report.from_url(report_url)
    new_blocks: list[Any] = [wr.MarkdownBlock(text=markdown)]
    if panels:
        new_blocks.append(
            wr.PanelGrid(
                runsets=[wr.Runset(entity=report.entity, project=report.project)], panels=panels
            )
        )
    collapsed = collapsed or {}
    new_blocks += [
        wr.H2(text=h, collapsed_blocks=[wr.MarkdownBlock(text=md)]) for h, md in collapsed.items()
    ]
    report.blocks = splice_section(report.blocks, heading, new_blocks, wr, drop=tuple(collapsed))
    report.save()
    return report.url


def results_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    label: str = "Run",
    link_key: str = "wandb_url",
    fmt: str = "{:.4f}",
) -> str:
    """Markdown table of ``rows`` (``tag`` + metrics); first cell links to the run if possible."""
    head = f"| {label} | " + " | ".join(columns) + " |\n|---|" + "---|" * len(columns) + "\n"
    lines = []
    for r in rows:
        cell = f"[{r['tag']}]({r[link_key]})" if r.get(link_key) else str(r["tag"])
        vals = [
            fmt.format(r[c]) if isinstance(r.get(c), int | float) else str(r.get(c, ""))
            for c in columns
        ]
        lines.append(f"| {cell} | " + " | ".join(vals) + " |")
    return head + "\n".join(lines) + "\n"


VERDICT_COLUMNS = ("Experiment", "Baseline", "Promotion gate", "Result", "Verdict", "PR")


def verdict_table(entries: list[dict[str, Any]], pr_url: str = "") -> str:
    """Markdown table with one row per experiment: ``label`` (+ ``sweep``), ``baseline``, ``gate``,
    ``result``, ``verdict`` (KEEP / DISCARD / PROMOTE) and ``pr`` (a number linked under ``pr_url``,
    e.g. ``https://github.com/<owner>/<repo>/pull``, or a full URL)."""
    head = "| " + " | ".join(VERDICT_COLUMNS) + " |\n|" + "---|" * len(VERDICT_COLUMNS) + "\n"
    lines = []
    for e in entries:
        pr = e.get("pr", "")
        link = f"[#{pr}]({pr_url}/{pr})" if pr_url and str(pr).isdigit() else str(pr)
        name = f"**{e['label']}** `{e['sweep']}`" if e.get("sweep") else f"**{e['label']}**"
        cells = [
            name,
            e.get("baseline", ""),
            e.get("gate", ""),
            e.get("result", ""),
            e.get("verdict", ""),
            link,
        ]
        lines.append(
            "| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in cells) + " |"
        )
    return head + "\n".join(lines) + "\n"
