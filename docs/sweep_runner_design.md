# In-package sweep runner — design doc (issue #20)

Status: **draft / pre-implementation**. Posted for discussion before any code lands. The eventual implementation PR follows once we settle the shape.

## Goal

Promote the iter-loop pattern duplicated across [`charleneleong-ai/orak-2025-starter-kit/experiments/autoresearch.py`](https://github.com/charleneleong-ai/orak-2025-starter-kit/blob/master/experiments/autoresearch.py) and [`charleneleong-ai/gemma4-rlvr/experiments/autoresearch.py`](https://github.com/charleneleong-ai/gemma4-rlvr/blob/main/experiments/autoresearch.py) into the package, behind a clean enough abstraction that adding a third project doesn't require copy-pasting ~600 lines of orchestration code.

## Method

Side-by-side read of both projects' `experiments/autoresearch.py`, classified into three buckets:

- **Identical** — extract verbatim into the package (zero project-side change required).
- **Same shape, different implementation** — extract behind a `Protocol` so each project plugs in its own.
- **Project-specific** — leaves nothing to extract; stays per-project.

Both files are similar size: orak ~1305 LOC, gemma4-rlvr ~640 LOC. Most of the orak length is project-specific param-proposal / trajectory-analysis logic that shouldn't be promoted.

## Bucket 1 — identical (extract into package)

| Pattern | orak | gemma4-rlvr | Notes |
|---|---|---|---|
| `_relabel_last_as_early_kill(tag/config, kill_reason)` mutates the last `results.jsonl` row to `status="EARLY_KILL"` and prepends `KILLED: <reason>` to `notes` | `experiments/autoresearch.py:695` | `experiments/autoresearch.py:223` | Implementations are nearly verbatim — orak has a per-game targeting refinement (only relabel the offender game); g4-rlvr has none because there's no game concept. Single function with optional `target_filter` covers both. |
| Subprocess lifecycle: `Popen` → graceful `SIGINT` → escalate to `terminate()` → escalate to `kill()` with grace windows | `run_experiment` lines 776-902 | `_run_with_triage` lines 366-540 | Both have the same three-step kill ladder with similar timeouts. Differs only in *what triggers* the kill, which is the Bucket 2 concern. |
| Iter wall-clock timeout (`ITER_TIMEOUT_MIN`, default 30) | constant @ line 574 | constant @ top | Same default value in both. Used identically. |
| `current_run.json` sidecar writing (the "RUNNING dot" feed for `plot_progress`) | `_write_sidecar` line 723 | inline in `main` lines 593-602 | Same JSON shape (`{experiment, config_name, description, started_at, log_path, iter_marker}`); g4-rlvr writes inline, orak helpered. Helper version belongs in package. |
| Post-iter pause to let GPU mem fully release before next subprocess (g4-rlvr `pause_s=15`; orak doesn't have this — but should) | — | line 575 + line 631 | Recommend extracting + setting orak default to non-zero. |

**Estimated extracted code**: ~120 LOC.

## Bucket 2 — same shape, different impl (Protocol candidates)

These are the loop's three real moving parts. Each project does the same thing in a different way; a `Protocol` per slot lets each plug in its own.

### 2a. `IterPlanner` — what's the next iter?

```python
class IterPlanner(Protocol):
    """Yields one IterPlan per iteration. Generator stops when the sweep is done."""
    def plan_iters(self, history: list[dict]) -> Iterator[IterPlan]: ...

@dataclass
class IterPlan:
    cmd: list[str]              # subprocess command to launch
    description: str            # for results.jsonl + chart label
    config_name: str | None     # which config slot (drives results.jsonl path)
    notes: str = ""             # extra context (autoresearch param diff, etc)
```

| Project | Current implementation |
|---|---|
| **orak** | Per-game `propose_next_params` reads `experiments/<tag>/results.jsonl` history, computes new YAML deltas via heuristic rules (theta curve, warmup steps, ...), writes back to `configs/<game>/agent/<config_type>.yaml`, builds cmd `[python, run.py, --config-name=<config>, --local, --games <g1> <g2>]`. |
| **gemma4-rlvr** | YAML schedule (`configs/schedules/<name>.yaml`) declares a flat list of `(config, extras, notes)` triples. Each iter pops one, builds cmd `[python, train.py, "train", "-c", config, "-d", desc, *extras]`. |

The two are radically different (one is a control-flow loop with feedback, the other is a static iteration over a list) but they meet the same contract: *given the history so far, yield the next iter to run, or stop*. Generator-based interface lets both work — schedule-driven is `for entry in schedule: yield IterPlan(...)`; feedback-driven is `while convergence_check(): propose → yield IterPlan(...)`.

### 2b. `TriageMonitor` — when to kill an in-flight iter

```python
class TriageMonitor(Protocol):
    """Polled while subprocess runs. Returns kill_reason on trigger, None to continue."""
    def setup(self, plan: IterPlan, baseline: float) -> None: ...
    def check(self, elapsed_s: float) -> str | None: ...
    def teardown(self) -> None: ...
```

| Project | What it monitors |
|---|---|
| **orak** | Polls `game_logs/<game>/<run_id>/game_states.jsonl` every 5s for plateau (max_eval unchanged for `TRIAGE_SCORE_PLATEAU_STEPS_PER_GAME` steps), no_learn (no episode improvement for `TRIAGE_NO_LEARN_EPISODES`), baseline_gate (max_eval < baseline × `TRIAGE_BASELINE_FACTOR` after 100 steps). Game-aware. |
| **gemma4-rlvr** | Reads subprocess stdout line-by-line, greps for step-marker lines, extracts `step_time` / `reward` / `kl` / `loss` / `grad_norm`. Triggers on slow steps, no-learn windows, KL spikes, loss explosion. Optional GPU watcher thread (deep-RL specific). |

Different signals, same interface. The `setup()` hook lets each implementation latch onto whatever channel it needs (file path for orak; subprocess.stdout pipe for g4-rlvr).

### 2c. `ResultExtractor` — turn a finished iter into `results.jsonl` row(s)

```python
class ResultExtractor(Protocol):
    """Called once a subprocess exits. Returns one or more rows to log."""
    def extract(self, plan: IterPlan, run_id: str, exit_code: int) -> list[dict]: ...
```

| Project | What it extracts |
|---|---|
| **orak** | `extract_run_results(run_id, games)` reads each game's `game_states.jsonl`, computes `max_eval` / `episodes` / `steps`, normalises evaluation_score, formats notes string, calls `log_experiment` with one row per game. |
| **gemma4-rlvr** | Reads training metrics from heldout eval JSONL (which the subprocess writes), extracts `score` / `mean_total` / `no_halluc` etc, formats notes, calls `log_experiment` with one row per iter. |

Both call into `autoresearch.results.log_experiment` which is already package-owned. The pre-call extraction is what's project-specific.

## Bucket 3 — project-specific (don't extract)

| Pattern | Why it stays per-project |
|---|---|
| orak's `propose_next_params` (game-aware heuristics over MACLA θ-curve params) | The tuning logic is genuinely specific to MACLA's learning dynamics. |
| orak's `analyze_trajectory` (per-game death-cluster, action-repetition, map-stuck detection) | Reads game-server-specific signals; no equivalent exists for non-game projects. |
| orak's `_apply_prompt_change` / `_apply_param_change` | Modifies game-specific YAML configs. |
| g4-rlvr's `_load_schedule` (parses the schedule YAML) | The YAML schema is g4-rlvr-specific (lists training-config names, not iter parameter deltas). |
| g4-rlvr's GPU watcher (`_gpu_watcher` thread) | Specific to deep-RL training where GPU underutil is the dominant failure signature. Could be a separately-importable helper, but doesn't belong in `SweepRunner`'s default. |
| g4-rlvr's `_crash_reason_from_lines` (greps stdout for traceback patterns) | Same shape across projects (pattern lookup) but the patterns themselves are RL-specific. Generalise via `TriageMonitor.check_for_crash_on_exit()` if there's appetite. |

## Proposed v1 API (minimal extraction)

```python
# autoresearch/sweep_runner.py — new module

@dataclass
class IterPlan:
    cmd: list[str]
    description: str
    config_name: str | None = None
    notes: str = ""
    timeout_min: int | None = None      # override SweepRunner.iter_timeout_min

class IterPlanner(Protocol):
    def plan_iters(self, history: list[dict]) -> Iterator[IterPlan]: ...

class TriageMonitor(Protocol):
    def setup(self, plan: IterPlan, baseline: float) -> None: ...
    def check(self, elapsed_s: float) -> str | None: ...
    def teardown(self) -> None: ...

class ResultExtractor(Protocol):
    def extract(self, plan: IterPlan, run_id: str, exit_code: int) -> list[dict]: ...

class SweepRunner:
    def __init__(
        self,
        *,
        tag: str,
        planner: IterPlanner,
        triage: TriageMonitor,
        extractor: ResultExtractor,
        retrospective_spec: RetrospectiveSpec | None = None,
        experiments_dir: str | Path = "experiments",
        iter_timeout_min: int = 30,
        triage_poll_s: int = 5,
        pause_between_iters_s: int = 15,
        sigint_grace_s: int = 60,
        sigterm_grace_s: int = 30,
    ): ...

    def run(self) -> None:
        """The main loop — plan → launch → monitor → log → retrospective → repeat."""
```

**Per-project glue shrinks dramatically**:

```python
# orak's experiments/autoresearch.py becomes ~30 lines:

class OrakPlanner:
    def plan_iters(self, history):
        for iter_n in range(self.max_iterations):
            params = propose_next_params(self.games, history)        # existing fn
            apply_params_to_yaml(params)                             # existing fn
            yield IterPlan(
                cmd=[PYTHON, "run.py", "--config-name=" + self.config, "--local",
                     "--games", *self.games],
                description=f"iter {iter_n + 1} | {param_summary(params)}",
                config_name=self.config,
            )

class OrakTriage:
    def setup(self, plan, baseline):
        self.run_id = wait_for_run_id_in_game_logs(...)
        self.baseline = baseline
    def check(self, elapsed_s):
        return _triage_check(self.run_id, self.games, {self.games[0]: self.baseline})
    def teardown(self): ...

class OrakExtractor:
    def extract(self, plan, run_id, exit_code):
        return [extract_one_game(run_id, g) for g in self.games]

runner = SweepRunner(
    tag="my_sweep",
    planner=OrakPlanner(...),
    triage=OrakTriage(...),
    extractor=OrakExtractor(...),
    retrospective_spec=load_retrospective_spec("configs/schedules/my_sweep.yaml"),
)
runner.run()
```

g4-rlvr's glue similarly shrinks: schedule-driven planner, stdout-line-grep triage, heldout-eval extractor.

## What this v1 *doesn't* do

Out of scope for the first PR — listed here so they're explicit:

- **No GPU watcher in core.** g4-rlvr's `_gpu_watcher` is RL-specific. Ship it as a separately-importable helper (`autoresearch.gpu_watcher.GPUWatcher`) that integrates with `TriageMonitor`, but don't bake it into `SweepRunner`.
- **No declarative YAML sweep spec.** Both projects currently use Python for the iter loop. A YAML-only declarative form ("autoresearch run --schedule v1_explore") is interesting but a v3 thing — first get the imperative API right.
- **No autoresearch-loop CLI command.** The runner is intended to be called from a project's `experiments/autoresearch.py`, which keeps its own `typer.Typer` CLI surface. Shipping a CLI on the package would force consensus on `--tag` / `--config-name` / `--max-iters` flag names that should stay project-side.
- **No SIGINT-aware sweep abort.** Both projects currently handle this differently (orak just lets the exception propagate; g4-rlvr catches `KeyboardInterrupt` and cleans the sidecar). v1 would do the cleanest version of g4-rlvr's pattern; the project-side `try/except` becomes unnecessary.

## Open questions

1. **Should `SweepRunner.run()` return a summary?** Yes for in-process use (sweep results, kill counts, etc). Currently both projects' `main()` returns `None` and prints. A `SweepResult` dataclass that aggregates everything seems cheap.
2. **`pause_between_iters_s` default**: g4-rlvr uses 15s (GPU memory release). orak has no pause. Default to 5? 0? Make it project-set?
3. **Where does `run_id` come from?** The two projects discover it differently — orak from `game_logs/<game>/<latest>` post-Popen, g4-rlvr from the subprocess's stdout (training script prints it). For the package, either: (a) `IterPlanner` returns `IterPlan.run_id` (forces planner to know in advance — bad for g4-rlvr), (b) `TriageMonitor.setup()` returns it (works for both), or (c) accept None and let `ResultExtractor.extract()` re-discover it. Leaning (b).
4. **Should we extract the iter-timeout-relabel as a no-op when `kill_reason is None`?** Currently `_relabel_last_as_early_kill` is called conditionally. Cleaner to always call it with `kill_reason | None` and let it skip internally.

## Scope estimate

- **Code**: ~250 LOC total — `sweep_runner.py` (~150), `subprocess_utils.py` (~80, the kill ladder), `current_run_writer.py` (~20, the sidecar helper).
- **Tests**: ~30 — Protocol implementations stubbed in tests; `SweepRunner` tested with fakes; relabel + sidecar tested in isolation.
- **Migration**: orak's `experiments/autoresearch.py` shrinks from 1305 LOC to ~150 LOC (the project-specific `propose_next_params` / `analyze_trajectory` stays). g4-rlvr similarly shrinks from 640 to ~120.
- **Risk**: medium. Both projects have working loops with subtle bugs already shaken out (`_find_run_id` mtime sort, prev_run_id checkpoint plumbing, EARLY_KILL relabel for timeouts, etc). Migration must preserve these — extensive integration testing on a real sweep before declaring done.

## Recommendation

**Land this as two PRs**:

1. **First PR — extract Bucket 1 only** (`_relabel_last_as_early_kill`, sidecar writer, subprocess-kill helper). Per-project `autoresearch.py` files import them but keep their own loop. Low-risk extraction; flushes out the API for the small utilities. ~3-5 days of work.

2. **Second PR — `SweepRunner` + Protocols + reference adapters** (`OrakPlanner` etc as test fixtures). Per-project `autoresearch.py` files migrate to the runner. Higher-risk; depends on (1).

Splitting like this means we can ship value (Bucket 1) without committing to the full API shape until both projects' adapters compile cleanly against the proposed Protocols.

## Want input on

- Are the 3 Protocols (`IterPlanner` / `TriageMonitor` / `ResultExtractor`) the right slicing? Could collapse to 2 or expand to 5; current count is "what feels minimal but not so coarse that adapters become awkward".
- Are there any other downstream projects in flight that should be surveyed before locking the API?
- Is the two-PR split worth the extra coordination, or should it land as one big PR?
