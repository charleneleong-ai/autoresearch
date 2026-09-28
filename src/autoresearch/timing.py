"""Phase timers: know where an iter's wall clock went before rightsizing anything."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Phases:
    """``with phases("fit"): ...`` accumulates minutes per phase; ``as_metrics()`` flattens them."""

    minutes: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.minutes[name] = self.minutes.get(name, 0.0) + (time.monotonic() - t0) / 60

    def as_metrics(self, prefix: str = "phase_min/") -> dict[str, float]:
        return {f"{prefix}{k}": round(v, 4) for k, v in self.minutes.items()}

    def dominant(self) -> str | None:
        return max(self.minutes, key=self.minutes.get) if self.minutes else None
