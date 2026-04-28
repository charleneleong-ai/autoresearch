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
