"""In-flight RUNNING-dot daemon for the autoresearch chart.

Watches the latest `logs/autoresearch_*.log` and keeps
`experiments/<TAG>[/<config_name>]/current_run.json` in sync with whichever
iteration is currently in flight, so plot_progress can render an extra
RUNNING marker between commits without waiting for the iter to finish.

TODO (v0.1 stub — see issue): port from
`charleneleong-ai/dotfiles` autoresearch-loop skill template
(`templates/current_run_updater.py`). Until then, copy that template into
your project directly.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--config", dest="config_name", default=None)
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--poll-s", type=int, default=15)
    p.parse_args()

    print(
        "autoresearch-current-run is a v0.1 stub.\n"
        "Until ported, copy templates/current_run_updater.py from\n"
        "  https://github.com/charleneleong-ai/dotfiles/tree/master/"
        "claude/plugins/research/skills/autoresearch-loop/templates\n"
        "into your project's experiments/ dir and run it with setsid+nohup.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
