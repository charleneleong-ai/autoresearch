"""Section detection over report blocks and the linked results table."""

from __future__ import annotations

from types import SimpleNamespace

from autoresearch.wandb_report import results_table, section_bounds, splice_section, verdict_table


class H2:
    def __init__(self, text: str) -> None:
        self.text = [text]


class MarkdownBlock:
    def __init__(self, text: str) -> None:
        self.text = text


WR = SimpleNamespace(H2=H2, MarkdownBlock=MarkdownBlock)


def test_section_bounds_accept_h2_or_markdown_headings():
    blocks = [
        H2("Task"),
        MarkdownBlock("intro"),
        MarkdownBlock("## Results\n| a |"),
        "panel",
        H2("Queue"),
        "p",
    ]
    assert section_bounds(blocks, "Results", WR) == (2, 4)
    assert section_bounds(blocks, "Queue", WR) == (4, 6)


def test_results_table_links_runs_and_formats_numbers():
    rows = [
        {"tag": "a", "S": 0.91234, "wandb_url": "http://w/a"},
        {"tag": "b", "S": 0.5, "note": "x"},
    ]
    table = results_table(rows, ["S", "note"])
    assert "| [a](http://w/a) | 0.9123 |  |" in table
    assert "| b | 0.5000 | x |" in table


def test_splice_section_replaces_it_and_drops_stale_collapsed_headings():
    blocks = [
        H2("Task"),
        MarkdownBlock("## Results\nold"),
        "panel",
        H2("Full run log"),
        H2("Next"),
        "n",
    ]
    out = splice_section(blocks, "Results", ["new", "log"], WR, drop=("Full run log",))
    assert out == [blocks[0], "new", "log", blocks[4], "n"]


def test_verdict_table_links_pr_numbers_and_keeps_urls():
    entries = [
        {
            "sweep": "v1",
            "label": "Wide",
            "baseline": "0.97",
            "gate": "> noise",
            "result": "a | b",
            "verdict": "KEEP",
            "pr": 12,
        },
        {"label": "Grid", "verdict": "PROMOTE", "pr": "https://x/pr/3"},
    ]
    table = verdict_table(entries, pr_url="https://gh/o/r/pull")
    assert (
        "| **Wide** `v1` | 0.97 | > noise | a / b | KEEP | [#12](https://gh/o/r/pull/12) |" in table
    )
    assert "| **Grid** |  |  |  | PROMOTE | https://x/pr/3 |" in table
