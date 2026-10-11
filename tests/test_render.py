"""The progress renderer on small-scale scores: relative y-margin, score format and row labels."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from autoresearch.render import _draw_axis, render  # noqa: E402

ROWS = [
    {"experiment": 0, "status": "KEEP", "S": 0.99050, "name": "first"},
    {"experiment": 1, "status": "DISCARD", "S": 0.99010},
    {"experiment": 2, "status": "KEEP", "S": 0.99393, "name": "best"},
]


def draw(**kw):
    fig, ax = plt.subplots()
    _draw_axis(ax, ROWS, score_field="S", score_label="S", title="t", **kw)
    texts = [t.get_text() for t in ax.texts]
    ylim = ax.get_ylim()
    plt.close(fig)
    return texts, ylim


def test_relative_margin_keeps_small_scores_readable():
    texts, (lo, hi) = draw(score_fmt="{:.4f}", label_field="name", y_margin=0.5)
    assert 0.98 < lo < 0.9901 and 0.9939 < hi < 1.0
    assert texts[0].startswith("first") and "S=0.9939" in texts[2]
    assert texts[1].startswith("E1")  # rows without the label field keep E<n>


def test_default_margin_is_the_legacy_absolute_padding():
    _, (lo, hi) = draw()
    assert (lo, hi) == (0.9901 - 2, 0.99393 + 3)


def test_render_writes_the_png(tmp_path):
    d = tmp_path / "official"
    d.mkdir()
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    out = render(experiments_dir=tmp_path, tag="official", score_field="S", y_margin=0.5)
    assert out.exists() and out.stat().st_size > 1000
