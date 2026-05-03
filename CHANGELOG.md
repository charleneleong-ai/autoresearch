## v0.9.0 (2026-05-03)

### Feat

- **retrospective**: wandb-history adapter + gradient_collapse detector (#21)

## v0.8.0 (2026-05-03)

### Feat

- **retrospective**: post-iter failure-mode detector framework (#17)

## v0.7.0 (2026-05-02)

### Feat

- **compare**: multi-metric axes + --from-results-jsonl extract (#15)

## v0.6.0 (2026-05-02)

### Feat

- **compare**: plot_milestone_progression — cross-experiment trajectory chart (#14)

## v0.5.1 (2026-05-02)

### Fix

- **release**: trust cz exit code; drop fragile output-text matching (#13)
- **release**: treat cz exit-21 (NO_COMMITS_TO_BUMP) as clean skip (#12)

## v0.5.0 (2026-05-02)

### Feat

- **verdict**: cross-tag ablation verdict module + CLI (#10)

## v0.4.1 (2026-05-01)

### Refactor

- **results**: dedup score/game-filter helpers across modules (#8)

## v0.4.0 (2026-05-01)

### Feat

- **compare**: cross-sweep comparison plots (#6)

## v0.3.0 (2026-04-28)

### Feat

- **report**: dynamic relpath for schedule link, support any --out depth (#5)

## v0.2.0 (2026-04-28)

### Feat

- **report**: autoresearch-report CLI for per-sweep writeups (#4)

## v0.1.0 (2026-04-27)

### Feat

- **gpu_monitor**: GPUMonitor context manager for util + memory tracking

## v0.0.3 (2026-04-27)

### Fix

- **devx**: pre-commit hooks (ruff + commitizen) + fix remaining E501
- **lint**: satisfy ruff — Optional → X | None, disable B008 for typer
- **release**: commitizen + GHA semver-driven auto-release on merge

## v0.0.2 (2026-04-27)

### Feat

- **render**: swap last print → rich.print for consistency
- **daemons**: switch print → rich.print with colour markup
- **daemons**: bind --poll-s to envvar in pr_updater + current_run
- **daemons**: port pr_updater + current_run from gemma4-rl, switch to typer

### Refactor

- **pr_updater**: use tag_dir for png_path default

## v0.0.1 (2026-04-27)

### Feat

- v0.1 skeleton — results, charts, render + daemon stubs
