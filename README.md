# autoresearch

[![Lint](https://github.com/charleneleong-ai/autoresearch/actions/workflows/lint.yml/badge.svg)](https://github.com/charleneleong-ai/autoresearch/actions/workflows/lint.yml)
[![Test](https://github.com/charleneleong-ai/autoresearch/actions/workflows/test.yml/badge.svg)](https://github.com/charleneleong-ai/autoresearch/actions/workflows/test.yml)
[![Release](https://github.com/charleneleong-ai/autoresearch/actions/workflows/release.yml/badge.svg)](https://github.com/charleneleong-ai/autoresearch/actions/workflows/release.yml)
[![Version](https://img.shields.io/badge/version-v0.6.0-blue)](https://github.com/charleneleong-ai/autoresearch/releases/latest)

Self-driving experiment sweep loop — daemon-detached `autoresearch.py` + live PR-updating progress chart. Extracted from a coding-agent research-loop skill and stabilised across multiple ML training projects.

## What it does

| Module | Role |
|---|---|
| `autoresearch.results` | Read/write `experiments/<TAG>[/<config_name>]/results.jsonl` — `load_results`, `log_experiment`, `_tag_dir` with optional per-config sub-results |
| `autoresearch.charts` | Plotly post-script widgets: `plotly_label_toggle` for interactive HTML charts |
| `autoresearch.render` | Standalone matplotlib renderer for `progress.png` — no Plotly/kaleido/Chrome dep |
| `autoresearch.pr_updater` | Periodic daemon: refreshes chart + regenerates `progress.html` + PATCHes the PR body between `<!-- SWEEP_NARRATIVE_START/END -->` markers (10-min poll cadence) |
| `autoresearch.current_run` | Detached daemon: watches `logs/autoresearch_*.log` and writes `current_run.json` for the in-flight RUNNING dot |
| `autoresearch.gpu_monitor` | `GPUMonitor` context manager — samples `nvidia-smi` while a workload runs, emits a summary with mean util, peak memory, and rightsizing hints. Drop-in for training, sweeps, and eval scripts. |

## Install

For consumers (downstream projects depending on this package):

```bash
uv add autoresearch                   # once published; until then:
uv pip install git+https://github.com/charleneleong-ai/autoresearch.git
```

For development on this package itself, use the `mise` task runner:

```bash
mise run init        # creates .venv + installs the package + dev deps
mise run test        # runs pytest
mise run bump-dry    # previews the next release bump
```

(Setup uses `python3.11 -m venv .venv && uv pip install -e '.[dev,pr]'` under the hood — see `mise.toml`.)

## Usage

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

```bash
# Append a new milestone after each sweep verdict — milestones.yaml is the
# canonical chronological log of cross-experiment progress. File is created
# on first append; seed top-level title / primary_metric / threshold by hand
# once (see examples/milestones.example.yaml for the full schema).
autoresearch-compare append-milestone \
  --milestones-yaml docs/experiments/<task>/milestones.yaml \
  --label v3_slot_grounded \
  --description "Slot-grounded JSON output" \
  --metric mean_total=10.96 \
  --metric no_halluc=-0.48

# Or pull metrics directly from a sweep's results.jsonl (no hand-typing)
autoresearch-compare append-milestone \
  --milestones-yaml docs/experiments/<task>/milestones.yaml \
  --label e25_run \
  --from-results-jsonl experiments/<task>/<config>/results.jsonl \
  --row best \
  --extract mean_total=metrics.heldout.mean_total \
  --extract no_halluc=metrics.heldout.no_hallucinated_facts_mean

# Render the trajectory chart from the same YAML.
# `primary_metric` / `secondary_metric` accept either a scalar or a list
# (stacks multiple lines on the same axis — see examples/milestones.stacked.example.yaml).
autoresearch-compare progression \
  --milestones-yaml docs/experiments/<task>/milestones.yaml \
  --out milestones.png
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

Alpha, personal use. Validated against live multi-month sweeps. Current modules — see [`CHANGELOG.md`](CHANGELOG.md) for what landed in each release:

| Module | Purpose |
|---|---|
| `results` | JSONL I/O for `experiments/<tag>[/<config>]/results.jsonl`; `get_score` / `filter_by_game` helpers |
| `charts` | Plotly label-toggle widget for live HTML charts |
| `render` | Static matplotlib PNG render — Plotly fallback for headless / minimal envs |
| `compare` | Cross-sweep comparison plots — multi-tag overlay, cross-game scoreboard, milestone progression |
| `pr_updater` | Daemon: re-renders progress.png + commits + patches PR body between markers |
| `current_run` | Daemon: in-flight RUNNING dot driver |
| `report` | Per-sweep markdown writeup scaffolder |
| `verdict` | Cross-tag ablation verdict — HELPS / NEUTRAL / REGRESSES + optional PR comment |
| `gpu_monitor` | GPU util/memory tracker context manager |

## Releasing — automatic on merge to main

Releases are driven by [commitizen](https://commitizen-tools.github.io/commitizen/) and fire automatically on every merge to `main` whose commits warrant a bump. Three workflows split the work:

- **`lint.yml`** runs pre-commit (ruff check + ruff format + hygiene hooks) on every PR and push to main
- **`test.yml`** runs pytest on every PR and push to main (gates merge alongside lint)
- **`release.yml`** runs on push to main and:
  1. Inspects commits since the last tag — if there's a `feat:` / `fix:` / `BREAKING CHANGE`, decides the semver increment; otherwise exits as a no-op
  2. `cz bump --yes` — bumps `pyproject.toml:version` + `src/autoresearch/__init__.py:__version__`, prepends a new section to `CHANGELOG.md`, commits as `bump: version X.Y.Z → A.B.C`, creates annotated tag `vA.B.C`
  3. Pushes the bump commit + tag back to main (loop-protected via `if: !startsWith(commit_message, 'bump:')` so the bump itself doesn't re-trigger)
  4. Builds wheel + sdist
  5. Runs `pytest` against the built wheel (catches packaging bugs)
  6. Creates a GitHub Release with auto-generated notes from PR titles since the previous tag, attaches wheel + sdist

### How conventional commits map to semver

| commit prefix | bump | example |
|---|---|---|
| `feat:` | MINOR | `feat(daemons): add --poll-s envvar` → 0.0.2 → 0.1.0 |
| `fix:` | PATCH | `fix(render): handle empty results.jsonl` → 0.0.2 → 0.0.3 |
| `BREAKING CHANGE:` (or `feat!:`) | MAJOR | (see note below for major_version_zero) |
| `chore:`, `docs:`, `refactor:`, `test:`, `style:` | no bump | release.yml exits cleanly |

While the package is alpha, `[tool.commitizen] major_version_zero = true` keeps breaking changes at MINOR (so 0.x.y stays under 1.0.0 until we explicitly cut 1.0.0). Drop the flag to enable real MAJOR bumps.

### Local commands (mise tasks)

```bash
mise run init        # bootstrap .venv + dev deps + commitizen
mise run test        # pytest
mise run lint        # ruff
mise run bump-dry    # preview what cz bump would do (no writes)
mise run bump        # local bump without push (rarely needed — CI does this)
mise run release     # bump + push --follow-tags (also rarely — CI does this)
```

### CHANGELOG.md vs auto-release-notes

Both exist by design:
- **`CHANGELOG.md`** — committed alongside each bump, grouped by Feat / Fix / Refactor / BREAKING CHANGE. The canonical human-readable history; the source of truth.
- **GitHub release notes** (auto-generated) — link-rich (PR numbers, author handles) and visible on the Releases page. Complements rather than duplicates the CHANGELOG.

## License

MIT — see `LICENSE`.
