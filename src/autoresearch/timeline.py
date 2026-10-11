"""Metric timelines through `autoresearch.render`: how metrics moved across a sequence of
records (submissions, releases, evaluations) that are not sweep rows.

A record is a flat dict of metric values plus a label. A `Timeline` keeps the records where its
`key` metric changed (and is non-zero) and draws one panel per `Term`, marking each new best of
that term as KEEP.

    terms = (Term("consistency"), Term("dlcfs", flip=True))
    Timeline("official S", key="S", terms=terms).plot(records, out)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.render import render


@dataclass(frozen=True)
class Term:
    """One plotted metric; `flip` shows 1 - x, for distances and errors where lower is better."""

    field: str
    title: str = ""
    flip: bool = False

    def value(self, record: dict[str, Any]) -> float:
        return 1 - record[self.field] if self.flip else record[self.field]


@dataclass(frozen=True)
class Timeline:
    label: str
    key: str
    terms: tuple[Term, ...]

    def steps(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The records where `key` changed to a non-zero value, in order."""
        steps, last = [], None
        for r in records:
            if r[self.key] and r[self.key] != last:
                steps.append(r)
            last = r[self.key]
        return steps

    def rows(
        self, records: list[dict[str, Any]], label_field: str = "name"
    ) -> list[dict[str, Any]]:
        """One row per (term, step); with several terms each row's `game` is its term, so each
        term gets its own panel."""
        rows = []
        for term in self.terms:
            best = float("-inf")
            for i, r in enumerate(self.steps(records)):
                v = term.value(r)
                row = {"experiment": i, "status": "KEEP" if v > best else "DISCARD", "value": v}
                row[label_field] = r.get(label_field)
                if len(self.terms) > 1:
                    row["game"] = term.title or term.field
                rows.append(row)
                best = max(best, v)
        return rows

    def plot(
        self,
        records: list[dict[str, Any]],
        out: Path,
        label_field: str = "name",
        score_fmt: str = "{:.4f}",
    ) -> Path:
        return render(
            rows=self.rows(records, label_field),
            out=out,
            title=self.label,
            score_field="value",
            score_label=self.label if len(self.terms) == 1 else "value (1 - x where flipped)",
            score_fmt=score_fmt,
            label_field=label_field,
            y_margin=0.15,
        )
