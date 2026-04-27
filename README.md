# autoresearch

Self-driving experiment sweep loop — daemon-detached `autoresearch.py` + live PR-updating progress chart. Extracted from the [`autoresearch-loop` Claude skill](https://github.com/charleneleong-ai/dotfiles/tree/master/claude/plugins/research/skills/autoresearch-loop) and the reference implementations in [`gemma4-rlvr#4`](https://github.com/charleneleong-ai/gemma4-rlvr/pull/4) and [`orak-2025-starter-kit#20`](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/20).

## What it does

| Module | Role |
|---|---|
| `autoresearch.results` | Read/write `experiments/<TAG>[/<config_name>]/results.jsonl` — `load_results`, `log_experiment`, `_tag_dir` with optional per-config sub-results |
| `autoresearch.charts` | Plotly post-script widgets: `plotly_label_toggle` for interactive HTML charts |
| `autoresearch.render` | Standalone matplotlib renderer for `progress.png` — no Plotly/kaleido/Chrome dep |
| `autoresearch.pr_updater` | Periodic daemon: refreshes chart + PATCHes the PR body between `<!-- SWEEP_NARRATIVE_START/END -->` markers (stub — see TODO) |
| `autoresearch.current_run` | Detached daemon: watches `logs/autoresearch_*.log` and writes `current_run.json` for the in-flight RUNNING dot (stub — see TODO) |

## Install

```bash
uv add autoresearch                   # once published; until then:
uv pip install git+https://github.com/charleneleong-ai/autoresearch.git
```

## Usage (v0.1)

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

## Status

**v0.1 — Alpha, personal use.** The data layer (`results.py`) and chart widget (`charts.py`) are stable and ported verbatim from the live implementations. The renderer is functional. The `pr_updater` and `current_run` daemons are CLI-stubs — see issues for TODOs.

## License

MIT — see `LICENSE`.
