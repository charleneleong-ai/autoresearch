# autoresearch

Self-driving experiment sweep loop — daemon-detached `autoresearch.py` + live PR-updating progress chart. Extracted from a coding-agent research-loop skill and stabilised across multiple ML training projects.

## What it does

| Module | Role |
|---|---|
| `autoresearch.results` | Read/write `experiments/<TAG>[/<config_name>]/results.jsonl` — `load_results`, `log_experiment`, `_tag_dir` with optional per-config sub-results |
| `autoresearch.charts` | Plotly post-script widgets: `plotly_label_toggle` for interactive HTML charts |
| `autoresearch.render` | Standalone matplotlib renderer for `progress.png` — no Plotly/kaleido/Chrome dep |
| `autoresearch.pr_updater` | Periodic daemon: refreshes chart + regenerates `progress.html` + PATCHes the PR body between `<!-- SWEEP_NARRATIVE_START/END -->` markers (10-min poll cadence) |
| `autoresearch.current_run` | Detached daemon: watches `logs/autoresearch_*.log` and writes `current_run.json` for the in-flight RUNNING dot |

## Install

```bash
uv add autoresearch                   # once published; until then:
uv pip install git+https://github.com/charleneleong-ai/autoresearch.git
```

## Usage (v0.0.2)

```python
from autoresearch.results import load_results, log_experiment

# Load all rows for a tag (flat layout)
rows = load_results(experiments_dir="experiments", tag="my_sweep")

# Per-config layout — multiple parallel sweeps
rows_a = load_results(experiments_dir="experiments", tag="my_sweep", config_name="gemma")
rows_b = load_results(experiments_dir="experiments", tag="my_sweep", config_name="qwen")
```

```python
from autoresearch.charts import plotly_label_toggle

post_script = plotly_label_toggle(
    label_indices=label_annotation_indices,
    n_traces=len(fig.data),
    position="top-right",
)
fig.write_html(path, post_script=post_script)
```

```bash
autoresearch-render --tag my_sweep                  # flat
autoresearch-render --tag my_sweep --config gemma   # per-config
```

### Daemons (PR refresher + in-flight chart row)

Both daemons are intended to run detached so they survive SSH or coding-agent session death (verify `PPID=1` after launch):

```bash
# PR refresher: re-renders progress.png + regenerates progress.html + PATCHes PR body
setsid nohup autoresearch-pr-updater \
  --tag my_sweep --config gemma \
  --pr 42 --repo me/myproj --branch feat/sweep \
  </dev/null >>logs/pr_updater_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown

# Current-run sidecar: drives the in-flight RUNNING dot
setsid nohup autoresearch-current-run \
  --tag my_sweep --config gemma \
  </dev/null >>logs/current_run_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
```

The PR body must contain marker comments `<!-- SWEEP_NARRATIVE_START -->` and `<!-- SWEEP_NARRATIVE_END -->` somewhere in its body — the updater patches the table between them.

## Status

**v0.0.2 — Alpha, personal use.** All five modules (`results`, `charts`, `render`, `pr_updater`, `current_run`) are functional. Ported and validated against live multi-month sweeps.

## License

MIT — see `LICENSE`.
