"""Periodic PR refresher for an autoresearch sweep.

Polls every POLL_S seconds and:
1. Re-renders `experiments/<TAG>[/<config_name>]/progress.png` from `results.jsonl`.
2. If the PNG changed: git add + commit + push (GitHub serves the embedded
   image via `?raw=true`).
3. PATCHes the active PR's body between
   `<!-- SWEEP_NARRATIVE_START -->` … `<!-- SWEEP_NARRATIVE_END -->` markers
   with a sweep-summary table built from `results.jsonl`.

TODO (v0.0.1 stub — see issue): port the full daemon from
`charleneleong-ai/dotfiles` autoresearch-loop skill template
(`templates/_pr_updater.py`). Until then, copy that template into your
project directly.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--config", dest="config_name", default=None)
    p.add_argument("--pr", type=int, required=True, help="PR number to PATCH")
    p.add_argument("--repo", required=True, help='owner/name (e.g. "you/repo")')
    p.add_argument("--branch", required=True)
    p.add_argument("--poll-s", type=int, default=600)
    p.parse_args()

    print(
        "autoresearch-pr-updater is a v0.0.1 stub.\n"
        "Until ported, copy templates/_pr_updater.py from\n"
        "  https://github.com/charleneleong-ai/dotfiles/tree/master/"
        "claude/plugins/research/skills/autoresearch-loop/templates\n"
        "into your project's experiments/ dir and run it with setsid+nohup.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
