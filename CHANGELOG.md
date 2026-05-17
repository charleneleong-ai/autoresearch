## v0.24.2 (2026-05-17)

### Fix

- **introspect**: render milestone step 0 as @0, not @n/a (#67)

## v0.24.1 (2026-05-17)

### Refactor

- **package**: housekeeping — Popen leak fix, narrow excepts, hoist imports (#66)

## v0.24.0 (2026-05-17)

### Feat

- **trajectory**: introspect framework — MilestoneSpec / DwellSpec / ActionSpec + CLI (#65)

## v0.23.1 (2026-05-16)

### Fix

- **release**: use annotated tags so --follow-tags pushes them (#63)

## v0.23.0 (2026-05-16)

### Feat

- **compare**: error bars on milestone progression (#62)

### Refactor

- **scoreboard**: dedupe chart + CLI between scoreboard variants (#61)

## v0.22.0 (2026-05-11)

### Feat

- **consolidate**: single-index scoreboard via consolidate() + scoreboard-from-index (#57)

## v0.21.0 (2026-05-08)

### Feat

- **current_run**: pluggable LogFormat presets (default + untimed) (#54)

### Refactor

- lint cleanup + extract C901 helpers (#53)

## v0.20.0 (2026-05-05)

### Feat

- **trajectory**: outcome-tagging fields, TrajectoryWriter.recent(), format_recent_history (#50)

## v0.19.6 (2026-05-05)

### Fix

- **render**: delegate _kill_tag to categorize_kill_reason

## v0.19.5 (2026-05-05)

### Fix

- **ci**: use RELEASE_PAT when available + correct empty-commit comment (#46)

## v0.19.4 (2026-05-05)

### Fix

- **ci**: empty-commit kick to trigger workflows on bot-opened release PR (#44)

## v0.19.3 (2026-05-05)

### Fix

- **ci**: add workflow_dispatch trigger to release.yml (#42)

## v0.19.2 (2026-05-05)

### Fix

- **ci**: treat cz exit 3 (NO_COMMITS_FOUND) as clean skip (#40)

## v0.19.1 (2026-05-05)

### Fix

- **ci**: re-anchor release tag to main HEAD in catch-up step (#38)

## v0.19.0 (2026-05-05)

### Feat

- **token-confidence**: per-row logprob diagnostic for eval failures (#33)

## v0.18.0 (2026-05-05)

### Feat

- **llm-utils**: retry_utils + prompt_caching + normalization — phase 1 from orak (#34)

## v0.17.0 (2026-05-05)

### Feat

- **trajectory**: TrajectoryWriter + StepRecord for agentic-RL pipelines (#32)

## v0.16.0 (2026-05-05)

### Feat

- **triage**: crash_reason_from_stdout + GPUTriage + decide_status (#30)

## v0.15.0 (2026-05-05)

### Feat

- **results**: extra_classifier hook for project-specific kill categories (v0.14.0) (#27)

## v0.14.0 (2026-05-04)

### Feat

- **results**: extra_classifier hook for project-specific kill categories
- **results**: public categorize_kill_reason classifier (v0.13.0) (#26)

## v0.13.0 (2026-05-04)

### Feat

- **results**: public categorize_kill_reason classifier

## v0.12.0 (2026-05-03)

### Feat

- **loop**: in-package SweepRunner with three Protocols (#25)

## v0.11.0 (2026-05-03)

### Feat

- **sweep-helpers**: bucket-1 — relabel + kill ladder + sidecar (#24)

## v0.10.0 (2026-05-03)

### Feat

- **retrospective**: value_transform_mismatch + sign_flip_in_rubric alias (#23)

## v0.9.1 (2026-05-03)

### Fix

- **retrospective**: broaden gradient_collapse exception handler (#22)

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
