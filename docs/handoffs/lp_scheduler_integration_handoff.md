# LPServe Live `LPScheduler` Integration Handoff

Evidence record for the live `LPScheduler(BaseScheduler)` integration and its
focused CPU validation. The normative contract is recorded in
`docs/lp_scheduler_design.md` §13 ("Live scheduler integration"), §15.4
traceability, and the D-17 register entry. The mathematical layer, mapper,
executor, and their tests are reused unchanged.

## Provenance

| Item | Value |
|---|---|
| Base revision | `d1bd69c2d5c359f542f086d07de3340b5e0e5e7c` on `main`, in sync with `origin/main`; no newer commits |
| Reviewed commits included | `0afba0d`, `74f5d09`, `f053afd`, `d1bd69c` (full hashes confirmed as ancestors with `git merge-base --is-ancestor`) |
| Tested revision | Base plus uncommitted changes (not committed). `git diff d1bd69c -- sarathi docs/lp_scheduler_design.md \| sha256sum` = `8f743fde…afcbc`; `sha256(lp_scheduler.py)` = `debfd4ec…eb0c3`; `sha256(tests/test_lp_scheduler.py)` = `0db8d438…4eec`. This handoff file was written after testing. Re-verified after commit at implementation commit `276e5f4c188b6137fd5f953eda67503d41b0718f` (tested paths clean against HEAD): the four suites again ran 5, 7, 9, and 13 tests, all OK, on `gpu048`, SLURM job `65250285`. |
| Pre-existing unrelated state | Untracked `CLAUDE.md`, left untouched |
| Host / allocation | Unity node `gpu048`, SLURM job `65249627` (post-commit re-run: job `65250285`), partition `gpu-preempt`; CPU-only work. No model, CUDA, GPU, or benchmark was run |
| Environment | `module load Python/3.10.8-GCCcore-12.2.0`, then `source env/bin/activate`, in the same shell as each command |
| Interpreter / dependencies | `/home/atjoshi_umass_edu/LPServe/env/bin/python`, Python 3.10.8; NumPy 2.2.6, SciPy 1.15.3, torch 2.3.0+cu121, transformers 4.57.6 |

## Changed paths

- `sarathi/config.py`: `SchedulerType.LP = 8` (no renumbering) and
  `LPSchedulerConfig`.
- `sarathi/core/scheduler/lp_scheduler.py` (new): `LPScheduler` and
  `LPSchedulingError`.
- `sarathi/core/scheduler/scheduler_registry.py`: registers `LP` →
  `LPScheduler`.
- `sarathi/core/block_space_manager/block_space_manager_registry.py`:
  registers `LP` → existing `VLLMBlockSpaceManager`.
- `docs/lp_scheduler_design.md`: §13 integration subsection, §15.4
  traceability, D-17 pointer.
- `tests/test_lp_scheduler.py` (new). **Gitignored** by the `test*` rule
  (`.gitignore:205`), like the other tracked tests, so `git status` does not
  list it. A later commit needs `git add -f tests/test_lp_scheduler.py`.
- `docs/handoffs/lp_scheduler_integration_handoff.md` (new, this file).

## Implemented behavior

- **Entry guard.** `schedule()` raises `LPSchedulingError` (stage
  `scheduler_entry`, category `unsupported_state`) for a stage count other
  than one or a nonzero `num_running_batches`. It does this *before*
  `BaseScheduler.schedule()`, so `_iteration_id` is unchanged and the
  inherited empty output cannot occur.
- **Supported call.** The inherited method advances `_iteration_id` once.
  `_schedule()` reads `time.monotonic()` once. It returns the D-18 idle output
  (running empty and waiting empty or head not yet arrived) without building
  utilities or calling any LP component. Otherwise it builds uniform utilities
  and runs mapper → `solve_and_extract` → `execute_plan`, with each stage
  running only after the previous one succeeded.
- **Failure conversion.** A returned `MappingFailure` or `Failure` is raised
  as `LPSchedulingError` carrying the frozen record unchanged. There is no
  broad handler. Post-mutation exceptions propagate unchanged.
- **Configuration.** All LP inputs are required constructor arguments, with no
  defaults. `max_num_batched_tokens` returns `b_max`. A multi-stage
  configuration raises `ValueError`.
- **Import placement.** The scheduler imports the repository-root modules
  `lp_relaxation_scheduler`, `lpserve_state_mapping`, and
  `lpserve_plan_execution` inside its methods. `sarathi` is an editable
  install that exposes only the package. A module-level import would have made
  every `SchedulerRegistry` import fail unless the repository root is on
  `sys.path` (for example `python examples/offline_inference.py`). Observed:
  importing the registry from `/tmp` succeeds, while
  `lp_relaxation_scheduler` is not importable there. **Consequence:**
  `LPScheduler.schedule()` itself still requires the repository root on
  `sys.path`, as in the documented `python -m sarathi.benchmark.main` run from
  the repository root.

## Scoped inputs

The fixture uses `b_max=8`, `c_max=4`, `s_max=3`, resident limit 4,
`max_model_len=32`, block size 4, 10 blocks, reserve 1,
`conservative_one_block_v1`, uniform utilities 1/1/1, and numerical policy
`1e-7`/`1e-6`/`1e-9`/`1e-9`. These are provisional plumbing inputs; the related
OPEN decisions are unchanged. Case-specific overrides, each documented in the
test that uses it:

- decode policy `exact_gap_v1` (real mapping failure);
- reserve 11 (real solver infeasibility);
- reserve 9 (real all-zero plan → `no_progress`);
- resident limit 1 (real executor `resident_capacity` rejection).

## Exercised guarantees (`tests/test_lp_scheduler.py`, 13 tests)

These tests use the real registered constructor, the native
`VLLMBlockSpaceManager`, and the existing disabled `MetricsStore` mode (set up
and torn down at test-module scope). Requests are real `Sequence` objects:
the central `EngineSequenceManager` shares them with the scheduler, and a real
`WorkerSequenceManager` holds `deepcopy` copies. Replay follows the engine's
single-stage order: engine `on_schedule` → worker `on_schedule` → worker
`on_step_completed` → engine `on_step_completed` → scheduler
`on_step_completed`. Synthetic `SamplerOutput` entries are supplied for every
scheduled entry, including prefill. Only detokenization is stubbed (an engine
manager subclass with `_decode_seq` as a no-op). This is not a test of text
generation.

- **Idle.** Empty and future-only states return empty native output. The
  output ID equals the singly incremented `_iteration_id`, no
  utility/mapper/solver/executor call is made, and the state fingerprint is
  otherwise unchanged.
- **Lifecycle.** Admission (4 of 8 tokens, 2 blocks) → resident prefill
  completing the prompt → decode at gap 0 → decode at gap 1 (allocates a
  third block) → `FINISHED_LENGTH_CAPPED`. Checked after each replay:
  - call order, one clock read, one decision ID shared by snapshot, problem,
    result, plan, and output, and zero running batches at mapping and at
    execution;
  - running-batch count 1 after `schedule()` and 0 after completion;
  - prompt progress and status on both the central and worker copies;
  - central and worker block tables equal.

  At the end the request has been removed from the scheduler and from both
  sequence managers, all 10 blocks are free centrally and on the worker, and
  the next call is idle.
- **Mixed and unselected resident.** Staggered arrivals give unique optima.
  Output `[(1, 4), (0, 0)]` shows the admission ordered before request 0's
  decode, although request 0 has the smaller ID. The next step emits
  `[(2, 4), (0, 0), (1, 0)]`. With three decode-ready residents, an 8-token
  arrival, and `s_max=3`, the admission is first, exactly two decodes follow
  in ascending order, and the third resident stays in `running` with its block
  table and sequence state unchanged. Which decode is left out is a solver
  tie, so only these invariants are asserted.
- **Pre-mutation failures.** The tests cover:
  - a real mapping failure, with no solve or execute call;
  - a real solver infeasibility, with no execute call and solver diagnostics
    retained (raw status 2);
  - a real `resident_capacity` executor rejection;
  - a real all-zero plan that becomes `no_progress` and is not converted into
    idle output.

  Each raises `LPSchedulingError` with a frozen record, advances
  `_iteration_id` exactly once, and leaves the scheduler fingerprint
  unchanged.
- **Unsupported entry.** Changing `num_pipeline_stages` after construction,
  and calling `schedule()` a second time before completion
  (`num_running_batches=1`), both raise before base scheduling: the iteration
  is unchanged and there is no clock read, LP call, or output.
- **Destructive failure.** The plan is to admit 1, admit 2, then decode 0.
  The second `_allocate` raises. The `RuntimeError` propagates unchanged (not
  as `LPSchedulingError`), and no output or append is produced. The first
  admission remains mutated and request 2 has left `waiting` without becoming
  resident; nothing is rolled back. The fixture is then discarded.

Sanity check (scratch script outside the repository, not retained): patching
out the entry guard, the idle shortcut, or failure propagation in memory made
the corresponding tests fail (2/2, 2/2, 5/5), while the unmodified code passed.

## Commands and observed results

Before implementation (base `d1bd69c`, same shell setup), each suite was run
separately:

```text
test_lp_relaxation_scheduler.py   Ran 5 tests  OK
test_lpserve_state_mapping.py     Ran 7 tests  OK
test_lpserve_plan_execution.py    Ran 9 tests  OK
```

After implementation:

```text
python -m py_compile sarathi/config.py sarathi/core/scheduler/lp_scheduler.py \
  sarathi/core/scheduler/scheduler_registry.py \
  sarathi/core/block_space_manager/block_space_manager_registry.py \
  tests/test_lp_scheduler.py                                           exit 0
python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v  Ran 5   OK
python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v    Ran 7   OK
python -B -m unittest discover -s tests -p 'test_lpserve_plan_execution.py' -v   Ran 9   OK
python -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v             Ran 13  OK
git diff --check                                                       exit 0, no output
rg -n -i 'phase' sarathi/core/scheduler/lp_scheduler.py tests/test_lp_scheduler.py   exit 1, no matches
added lines in config/registries piped to rg -i 'phase'                exit 1, no matches
git diff --name-only   docs/lp_scheduler_design.md, sarathi/config.py,
                       sarathi/core/block_space_manager/block_space_manager_registry.py,
                       sarathi/core/scheduler/scheduler_registry.py
git status --short --branch --untracked-files=all
                       ## main...origin/main; M on the four files above;
                       ?? CLAUDE.md; ?? sarathi/core/scheduler/lp_scheduler.py
```

Passed: everything above. Failed: none. Skipped: none.

**Not run:** model initialization, CUDA, GPU smoke test, benchmark, and
`EngineArgs`/CLI construction (out of scope).

## Limitations and unresolved items

- **Not verified:** GPU execution, the model runner, sampler correctness,
  generation quality, performance, arbitrary-workload completion, and full
  acceptance of native execution.
- **Inherited and not repaired:** mixed-batch/sampler association (§16.4),
  non-transactional mutation (§16.5), and the head-of-queue idle assumption.
  Native completion also discards the sample from the prompt-completing
  prefill entry; the first output token comes from the first decode.
- **Runtime preemption** is still rejected by the executor. A positive penalty
  does not prevent the LP from selecting it, for example when memory binds.
- **Clocks.** The decision clock is `time.monotonic()`, as in the
  vLLM/Sarathi schedulers. The engine defaults `arrival_time` to
  `time.perf_counter()`. On this Linux CPython both read `CLOCK_MONOTONIC`;
  this was not verified on other platforms.
- **GPU follow-up.** `ModelRunner` memory profiling has no `SchedulerType.LP`
  branch, so the LP type takes its default profiling path. This is unexamined
  and is relevant only to a later GPU task.
- **Stale wording.** The guide, status, and the D-18 row still describe live
  integration as future work and are left for post-review updates.
- **Committing.** Implementation, config, design, and tests are in
  `276e5f4` (the test file was added with `git add -f`); this handoff is in
  `ec15af2`.
