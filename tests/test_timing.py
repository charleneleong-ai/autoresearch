"""Phase timer accumulation and flattening."""

from __future__ import annotations

from autoresearch.timing import Phases


def test_phases_accumulate_and_flatten():
    phases = Phases()
    with phases("fit"):
        pass
    with phases("fit"):
        pass
    with phases("score"):
        pass
    metrics = phases.as_metrics()
    assert set(metrics) == {"phase_min/fit", "phase_min/score"}
    assert all(v >= 0 for v in metrics.values())
    phases.minutes = {"fit": 2.0, "score": 0.5}
    assert phases.dominant() == "fit"
    assert Phases().dominant() is None
