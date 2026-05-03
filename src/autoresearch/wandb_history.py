"""Thin wandb-history adapter for the retrospective module.

The `gradient_collapse` detector (and any future detectors that need to look
at training-time series) reads `train/loss`, `train/reward`, etc. from a
wandb run via this module. Wandb is an *optional* dependency — install with
``pip install autoresearch[wandb]`` to enable it. Without the extra, the
detector silently skips instead of crashing.

API: one function, ``fetch_history``, returning a plain ``dict[str, list[float]]``
keyed by series name (no pandas in the surface).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Accept either a full wandb URL or the short `entity/project/run_id` form.
_FULL_URL_RE = re.compile(
    r"^https?://(?:www\.)?wandb\.ai/"
    r"(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)"
)
_SHORT_PATH_RE = re.compile(r"^(?P<entity>[^/]+)/(?P<project>[^/]+)/(?P<run_id>[^/]+)$")


@dataclass(frozen=True)
class WandbRunRef:
    """Parsed `entity/project/run_id` triple — what `wandb.Api().run(path)` needs."""

    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


def parse_run_url(run_url: str) -> WandbRunRef:
    """Parse a full URL or `entity/project/run_id` string into a `WandbRunRef`.

    Raises ``ValueError`` for unrecognised forms.
    """
    s = run_url.strip().rstrip("/")
    m = _FULL_URL_RE.match(s) or _SHORT_PATH_RE.match(s)
    if not m:
        raise ValueError(
            f"Unrecognised wandb run reference {run_url!r}. "
            f"Expected `https://wandb.ai/<entity>/<project>/runs/<run_id>` "
            f"or `<entity>/<project>/<run_id>`."
        )
    return WandbRunRef(
        entity=m.group("entity"),
        project=m.group("project"),
        run_id=m.group("run_id"),
    )


def fetch_history(
    *,
    run_url: str,
    keys: list[str],
    samples: int = 500,
) -> dict[str, list[float]]:
    """Pull a sampled history from wandb for ``keys`` (e.g. ``train/loss``).

    Returns ``{key: [values…]}`` with non-numeric / missing entries dropped.
    Per-key list length is at most ``samples`` (wandb sub-samples for large
    runs to keep the response small — see wandb's ``Run.history`` docs).

    Raises:
      ImportError: if the ``[wandb]`` extra isn't installed. Callers (e.g. the
        gradient_collapse detector) should catch this and silently skip.
      ValueError: if ``run_url`` can't be parsed.
      RuntimeError: if wandb refuses the request (bad credentials, run not
        found, etc.) — wraps the underlying exception with context.
    """
    try:
        import wandb  # noqa: F401  (presence check)
        from wandb.apis.public import Api
    except ImportError as e:
        raise ImportError(
            "wandb-history detectors need the [wandb] extra. "
            "Install with: pip install 'autoresearch[wandb]'"
        ) from e

    ref = parse_run_url(run_url)
    api = Api()
    try:
        run = api.run(ref.path)
    except Exception as e:  # wandb raises a CommError / ValueError stack
        raise RuntimeError(f"wandb api.run({ref.path!r}) failed: {e}") from e

    # `samples=N` returns a sampled DataFrame — much faster than `scan_history`.
    df = run.history(samples=samples, keys=list(keys), pandas=True)
    out: dict[str, list[float]] = {}
    for k in keys:
        if k not in df.columns:
            out[k] = []
            continue
        col = df[k].dropna().tolist()
        out[k] = [float(v) for v in col if isinstance(v, (int, float)) and v == v]  # NaN-safe
    return out


__all__ = ["WandbRunRef", "fetch_history", "parse_run_url"]
