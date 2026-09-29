"""Edit one section of an existing W&B report in place (needs the ``wandb-workspaces`` extra).

A report grows with a sweep: keep the narrative blocks, and replace the
section under one heading with fresh markdown and panels each time results
land. Section starts are detected as either an ``H2`` block or a
``MarkdownBlock`` whose first line is ``## ...`` (the updater's own previous
edit leaves the latter).
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


def replace_section(
    report_url: str,
    heading: str,
    markdown: str,
    panels: list[Any] | None = None,
) -> str:
    """Replace the section under ``heading`` with ``markdown`` (restating it) and panels."""
    import wandb_workspaces.reports.v2 as wr

    report = wr.Report.from_url(report_url)
    start, end = section_bounds(report.blocks, heading, wr)
    new_blocks: list[Any] = [wr.MarkdownBlock(text=markdown)]
    if panels:
        new_blocks.append(
            wr.PanelGrid(
                runsets=[wr.Runset(entity=report.entity, project=report.project)], panels=panels
            )
        )
    report.blocks = report.blocks[:start] + new_blocks + report.blocks[end:]
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
