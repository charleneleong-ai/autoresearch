## Unreleased

### Feat

- **release**: tag-triggered GHA release + scripts/release.py helper

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
