# autoresearch

[![Lint](https://github.com/charleneleong-ai/autoresearch/actions/workflows/lint.yml/badge.svg)](https://github.com/charleneleong-ai/autoresearch/actions/workflows/lint.yml)
[![Test](https://github.com/charleneleong-ai/autoresearch/actions/workflows/test.yml/badge.svg)](https://github.com/charleneleong-ai/autoresearch/actions/workflows/test.yml)
[![Release](https://github.com/charleneleong-ai/autoresearch/actions/workflows/release.yml/badge.svg)](https://github.com/charleneleong-ai/autoresearch/actions/workflows/release.yml)
[![Version](https://img.shields.io/badge/version-v0.19.4-blue)](https://github.com/charleneleong-ai/autoresearch/releases/latest)

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
uv add autoresearch                            # core only
uv add 'autoresearch[wandb]'                   # + gradient_collapse detector (wandb history)
uv pip install git+https://github.com/charleneleong-ai/autoresearch.git           # latest main
uv pip install 'autoresearch[wandb] @ git+https://github.com/charleneleong-ai/autoresearch.git'
```

Optional extras:
- `[pr]` — adds `requests` for the `pr_updater` daemon's PATCH path
- `[wandb]` — enables `wandb_history.fetch_history` and the `gradient_collapse` retrospective detector

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

### Post-iter retrospective ([#16](https://github.com/charleneleong-ai/autoresearch/issues/16))

After each iter, run a panel of `FailureDetector`s against the iter's
artefacts (`results.jsonl` row + per-iter log + optional `*.per_row.jsonl`
rubric dump). Each detector returns 0 or 1 `Finding`; findings get attached
to the row + rendered to a sibling markdown file.

```bash
# After-the-fact audit of a sweep that already finished
autoresearch-retrospective audit \
  --results-jsonl experiments/my_sweep/gemma/results.jsonl \
  --log logs/sweep_20260503.log \
  --per-row-jsonl experiments/my_sweep/gemma/per_row_E27.jsonl \
  --iter latest \
  --write-md experiments/my_sweep/gemma/retrospective_E27.md \
  --write-json   # appends `retrospective: {findings: [...]}` to the row
```

```python
# In-loop integration — call after _finalize_iter() in your project's autoresearch.py
from autoresearch import (
    BUILTIN_DETECTORS, audit_iter, attach_findings_to_row, format_markdown,
)

findings = audit_iter(
    results_row=row,
    log_path=Path("logs/sweep.log"),
    per_row_jsonl_path=Path(f"experiments/{tag}/per_row_E{i}.jsonl"),
    history=load_results(experiments_dir, tag, config_name),
    detectors=[BUILTIN_DETECTORS["silent_kill"], BUILTIN_DETECTORS["eval_score_plateau"]],
)
attach_findings_to_row(row, findings)
(out_dir / f"retrospective_E{i}.md").write_text(format_markdown(findings, iter_id=i))

# Use warn-level findings to drive the next iter's notes (self-correcting loop)
from autoresearch import filter_by_severity
for f in filter_by_severity(findings, "warn"):
    next_iter_notes.append(f"⚠ {f.detector}: {f.suggested_action}")
```

Built-in detectors and their YAML wiring (see `--spec` flag):

```yaml
# configs/schedules/<sweep>.yaml
post_iter_retrospective:
  enabled: true
  detectors:
    - silent_kill
    - triage_threshold_mismatch
    - eval_score_plateau
    - bucketed_failure
    - gradient_collapse                                          # needs [wandb] extra
    - sign_flip_in_rubric                                        # alias of value_transform_mismatch
  detector_kwargs:
    triage_threshold_mismatch: { min_first_score_step: 150 }    # task-specific
    bucketed_failure: { bucket_field: ground_truth_trigger }    # rubric field name
    gradient_collapse: { loss_key: train/policy_loss, window: 100 }
    sign_flip_in_rubric:
      cited_value_field: cited
      ground_truth_value_field: truth
      mismatch_threshold: 0.6
  on_finding:
    - { severity: warn,  action: append_to_next_iter_notes }
    - { severity: block, action: stop_sweep }
```

`gradient_collapse` reads `train/loss` and `train/reward` (or whatever you
configure under `loss_key`/`reward_key`) from the iter's wandb run via
`wandb_url` in the row, and fires `severity=block` when loss has collapsed
near zero AND reward has gone flat over the recent window — the joint
pattern that means the optimizer is stuck. It silently skips when:
`wandb_url` is absent (project doesn't use wandb), the `[wandb]` extra
isn't installed, history is shorter than `window`, or the wandb API call
fails.

`value_transform_mismatch` is the generic form that catches the family of
"rubric strict-equals on a transformed value" bugs (sign flips, unit
scaling, sign negation). Pass `transforms: [...]` to opt in:

```yaml
detectors:
  - value_transform_mismatch
detector_kwargs:
  value_transform_mismatch:
    transforms: [abs, scale_100, scale_0.01, negate]   # try all four
    cited_value_field: model_answer
    ground_truth_value_field: expected_answer
```

Built-in transforms: `abs` (sign flip), `negate` (sign reversal),
`scale_100` (decimal → percent: 0.12 → 12), `scale_0.01` (percent →
decimal: 12 → 0.12). Custom callables are accepted in-process via the
Python API — pass them directly in the `transforms` list. The
`sign_flip_in_rubric` alias is `value_transform_mismatch` pre-set with
`transforms=["abs"]`, matching #19's original spec verbatim — uses its
own kwargs key (`sign_flip_in_rubric: {...}`) so the alias and generic
form can be tuned independently in the same YAML.

Custom detectors (project-specific — e.g. `obs_collision` for game-agent
sweeps with state-jsonl obs ambiguity) are added by passing your own
`FailureDetector` callables to `audit_iter` or by registering them into a
custom dict and forwarding to `RetrospectiveSpec.selected_detectors`.

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
| `retrospective` | Post-iter failure-mode audit — pluggable detectors (silent_kill / triage_threshold_mismatch / eval_score_plateau / bucketed_failure / gradient_collapse) write findings into `results.jsonl` + sibling `retrospective_<iter>.md`. Self-correcting loop: warn-level findings are designed to feed the next iter's notes; block-level should stop the sweep. ([#16](https://github.com/charleneleong-ai/autoresearch/issues/16), [#18](https://github.com/charleneleong-ai/autoresearch/issues/18)) |
| `wandb_history` | Thin adapter: `fetch_history(run_url, keys, samples)` → `dict[str, list[float]]`. Lazy wandb import behind the `[wandb]` extra; powers `gradient_collapse` and any future detectors that read training-time series |
| `subprocess_utils` | `kill_gracefully(proc)` (SIGINT → SIGTERM → SIGKILL escalation ladder with grace windows) + `wait_with_timeout(proc, timeout_s, should_kill=...)` (poll-and-kill helper for iter loops). Extracted from duplicated boilerplate in orak / gemma4-rlvr's `experiments/autoresearch.py`. ([#20](https://github.com/charleneleong-ai/autoresearch/issues/20)) |
| `current_run` | Daemon (`autoresearch-current-run`) that tails sweep logs to maintain `current_run.json` for the chart's RUNNING dot. Also exposes `write_sidecar` / `clear_sidecar` / `sidecar` context manager for in-loop callers that already know the iter state and don't need a separate daemon process. |
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
