"""Section detection over report blocks and the linked results table."""

from __future__ import annotations

from types import SimpleNamespace

from autoresearch.wandb_report import results_table, section_bounds


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
