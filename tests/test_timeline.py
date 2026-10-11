"""Metric timelines: which records become steps, KEEP marks, flipped terms, panels per term."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from autoresearch.timeline import Term, Timeline  # noqa: E402

RECORDS = [
    {"name": "a", "S": 0.0, "cons": 0.5, "dist": 0.3},  # unscored: dropped
    {"name": "b", "S": 0.90, "cons": 0.6, "dist": 0.2},
    {"name": "c", "S": 0.90, "cons": 0.9, "dist": 0.1},  # key unchanged: dropped
    {"name": "d", "S": 0.95, "cons": 0.55, "dist": 0.1},
]


def test_steps_keep_changes_of_the_key_and_mark_new_bests():
    rows = Timeline(
        "S", "S", (Term("cons", "consistency"), Term("dist", "boundary", flip=True))
    ).rows(RECORDS)
    assert [(r["game"], r["name"], r["status"], round(r["value"], 2)) for r in rows] == [
        ("consistency", "b", "KEEP", 0.6),
        ("consistency", "d", "DISCARD", 0.55),
        ("boundary", "b", "KEEP", 0.8),
        ("boundary", "d", "KEEP", 0.9),
    ]


def test_single_term_rows_have_no_panel_and_plot_writes_the_png(tmp_path):
    timeline = Timeline("S", "S", (Term("S"),))
    assert all("game" not in r for r in timeline.rows(RECORDS))
    out = timeline.plot(RECORDS, tmp_path / "s.png")
    assert out.exists() and out.stat().st_size > 1000
