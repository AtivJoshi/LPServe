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


## Post-review provenance addendum (2026-10-05)

Mac review at `4f553b4f238fe93e87623de3612d20362888899f` confirmed the
implementation commit `276e5f4` contains the six implementation/design/test
paths listed above. The evidence commit `ec15af2d1879028152cc25cea6ead936143e46ef`
contains this handoff **and `CLAUDE.md`**, newly tracked with the single line
`@AGENTS.md`. That file was pre-existing and untracked before implementation;
tracking it was outside the prompt's allowed file list. The earlier statements
that it was left untouched describe the pre-commit working-tree state, not the
complete committed path list. No scheduler-runtime effect is attributed to it.

Commit `4f553b4` changes only this handoff to record post-commit verification.
The test file is tracked as of `276e5f4`; the earlier ignored-file staging note
is historical. This addendum corrects provenance only and records no new test
run. Historical commands and results above are preserved.


## Corrected connected CPU framework (2026-10-13; awaiting review)

Scope: the corrected planner connected to state mapping, immutable ranks and
coefficients, executor validation and native execution, live coordination,
LP-only pre-registration admission rejection, and obsolete
configuration/`EngineArgs` rejection (design §§12–14, §15.7; discussion
P10 step 9). Earlier sections of this handoff are historical evidence for their
revisions and are not current validation.

**Provenance.** Tested base `9a6ca3cf338f4212679eeaaabd0929cce4c0f26a` plus
the uncommitted changes in `working_tree.diff` (21 files; SHA-256 per file in
`sha256sums.txt`). Evidence:
`validation_output/lp_scheduler_integration/20261013T034740Z/` (`summary.json`,
`run_checks.sh`, per-check outputs with exit statuses). Host `gpu048`,
allocation `65616965`, CPU only; Python 3.10.8 (`env`), NumPy 2.2.6, SciPy
1.15.3. The work began on `gpu049` (allocation `65615469`), which Unity
preempted; development runs there were not retained and are not reused. This
section was written after the run; documentation is not part of the tested
code.

**Observed.** Compile, warnings-as-errors import, standalone smoke, and 138
focused tests passed (pure 38, mapping 12, executor 16, live scheduler and
submission 20, `EngineArgs` 10, script CPU tests 24 + 18); `git diff --check`
clean. Bounded enumeration: 150 seeded abstract states, 10,348 integer plans.
No check failed or was skipped. GPU, model, policy, and performance runs were
not executed.

### Coverage (design §15.7, P10 step 9)

| Obligation | Implementation | Check (tests/…) | Observed | Remaining limitation |
|---|---|---|---|---|
| Every amended row; raw/normalized validation | `lp_relaxation_scheduler` `_build_lp`, `_residuals`, `validate_integer_plan` | `test_lp_matrix_matches_independent_residuals`, `test_relaxed_validation_rejects_each_row`, `test_relaxed_bounds_projection_and_raw_versus_normalized`; enumeration: LP rows at integer points ⇔ native-order simulation | pass | — |
| Watermark zero/positive/shared; fixed admission charge | planner; mapper reads `watermark_blocks` | `test_watermark_zero_positive_and_shared`, `test_fixed_admission_charge_despite_short_chunk`; mapping equality case (8 + 2 = 10) | pass | h > 0 fixtures in small pools are abstract |
| Omitted residents; replacement; zero-recovery victims | planner resident row; executor resident check | `test_omitted_resident_counts_and_zero_recovery_replacement`; executor `test_prefills_then_zero_then_one_block_decodes` (omitted D keeps order) | pass | — |
| Zero/one/multiple free blocks; decode demands 0/1; decode-start gate | planner; mapper exact gap; executor append gate | `test_zero_free_blocks_zero_demand_decode`, `test_one_free_block_demand_one_then_zero`, `test_free_blocks_versus_one_block_decodes`, `test_mixed_decode_demand_groups`; executor A/B case (B before A); live `test_solver_reported_infeasibility_stops_before_executor` | pass | — |
| Full-context rounding; zero resident-prefill charge; recovery; supported/unsupported gaps; malformed allocation | `lpserve_state_mapping._capture_request`, pool accounting | `test_full_context_rounding_recovery_and_decode_demand`, `test_unsupported_gap_and_malformed_allocation` | pass | — |
| Many fractional; integral-idle/zero-support; self-victim; failed-trial cleanup; preservation; chunk floor/extension | whole-action constructor | nine-waiting construction (solver I = 2/(n+1)), integral-idle (ABSTRACT), self-victim, cleanup, chunk, extension tests | pass | integral-idle fixture is abstract |
| Necessary and avoidable negative first work; strict improvement on both sides of τ; zero/positive mass preemption | constructor (one pass, no two-pass policy) | named tests; avoidable case −3 vs integer optimum 0; positive case 101 victims, 99.5 | pass (accepted limitations reproduced) | utility quality, victim count, ties |
| No useful plan; abstract F=3,h=2; independent enumeration and returned-plan validation | constructor; test-only oracle (`oracle_choice` + native-order simulation) | `test_lp_feasible_integer_infeasible_abstract_fixture`, `test_bounded_enumeration_against_constructor`, `test_enumeration_on_named_fixtures`, `test_independent_oracle_rejects_malformed_decisions` | pass | bounded abstract states; F=3,h=2 has no validated-runtime counterpart |
| Outcomes, diagnostics, nonfinite and oversized numbers | `Failure.outcome`; checked arithmetic | solver-outcome, nonfinite, oversized-vector tests | pass | status 2 is a solver report, not a certificate |
| One capture; ranks; changing/fixed coefficients; uniform/per-request policies; invalid coefficients; no mutation | `capture_scheduler_state`, `vllm_oriented_ranks`, `construct_coefficients`; `LPScheduler._coefficient_policy` | `test_one_capture_and_state_dependent_policy_inputs`, `test_malformed_utilities_return_mapping_failure`, mapper fingerprints; live spies require exactly capture→map→solve→execute | pass | state-dependent fixture is not a production utility policy |
| Declared order; identical central/metadata order; multiple-victim reinsertion; running-tail readmission; arrived work behind a future head | executor declared orders; `LPScheduler._is_idle` | executor order, victim, and readmission tests; live and mapping future-head tests | pass | — |
| Planned charge/demand/recovery/watermark equal current state | executor `plan_state_mismatch` | `test_planned_resources_must_match_current_state` | pass | — |
| Admission: equality boundaries, total pool despite contention, no registration, other schedulers | `LPScheduler.check_new_request`; `BaseLLMEngine.add_request` | `SubmissionBoundaryTest` (real `add_request`, doubles for sequence manager, workers, metrics) | pass | real engine initialization not run |
| Registered or recomputation-expanded invalid context | mapper admissibility check (fail-stop) | `test_resident_ledger_and_inadmissible_registered_context` | pass | mapper level only |
| Obsolete arguments, including zero; positional order; serialization | `LPSchedulerConfig` reject-only detector; `EngineArgs` reject-only fields | `test_obsolete_arguments_fail_with_migration_guidance`, config-file and legacy-positional tests | pass | — |
| Defensive `no_progress`; zero mutation on rejection; post-mutation termination | executor gate before mathematical validation | executor control-only and precommit tests; live `test_no_progress_plan_is_not_ordinary_idle` (induced empty plan), post-mutation tests | pass | live no_progress requires an induced plan |
| Existing admission, partial prefill, decode, mixed, contention, preemption/recompute, `EngineArgs` paths | all layers | `LivePipelineTest`, `LivePreemptionTest`, `LiveDecodePreemptionTest`, `EngineArgs` tests | pass | relaxed values changed (below); inherited §16 limits |
| Active scripts migrated | config/record fields, reserve preconditions, utility hook | compile; script CPU tests | pass | GPU workloads unexecuted; two scripts' relaxed expectations re-derived analytically only. 2026-10-13 GPU (`82609eb`, `gpu051`): contention and preemption passed one dummy-weight run each (preemption boundary relaxed objective observed 15; evidence `lp_scheduler_contention/20261013T064744Z`, `lp_scheduler_preemption_gpu/20261013T065022Z`; see those handoffs); decode-preemption reference GPU unexecuted |
| GPU-helper order checks derive the approved order (added 2026-10-13, evidence `20261013T041256Z`) | `check_declared_order` in `scripts/check_lp_scheduler_preemption_gpu.py`, used by the contention and decode-preemption helpers | `tests/test_check_lp_scheduler_ordering.py` (real CPU pipeline records: B before A with one free block, prefills by service rank, victims by victim rank; obsolete order fails) | pass | helpers' GPU runs not executed. 2026-10-13: contention and preemption helpers ran on GPU with the derived order matching every step, including zero-demand before one-block decodes in one batch (contention steps 6, 8); decode-preemption helper unexecuted |
| Explicit obsolete `None` and positional supply rejected (added 2026-10-13) | `arg_utils` private omission sentinel | `test_obsolete_arguments_fail_with_migration_guidance`, `test_obsolete_arguments_omitted_or_positional` | pass | — |
| Oversized arrival reaches the mapper's structured rejection, not idle (added 2026-10-13) | `LPScheduler._is_idle` guarded conversion | `test_oversized_arrival_is_mapping_failure_not_idle` | pass | capture-stage failures carry no snapshot ID |

### F01–F04 dispositions

- **F01 (useful execution through relaxation/extraction):** implemented and
  CPU-checked for the supported model: constructor progress matched
  independent enumeration in every bounded state, and live paths executed.
  Closure is left to review; not marked closed here.
- **F02 (repeated lost work, omission):** OPEN; unchanged by this batch.
- **F03 (consistent allocation feasibility):** exact demand, decode-start
  gate, watermark rows, declared order, and executor current-state checks
  implemented and CPU-checked. Identical central/worker physical block IDs
  remain unestablished (§13 clarification). Closure left to review.
- **F04 (resident capacity and request rejection):** resident row,
  omitted-resident counting, and pre-registration rejection implemented and
  CPU-checked. Closure left to review.

### Findings recorded for review

- *Admission equality can admit a request that can never decode.* With
  h = 0 (pools under 100 blocks), a request whose full context equals the
  pool passes `A_i + h <= T`; after prefill it holds every block, its decode
  needs one free block, and it cannot be its own victim. The LP is then
  solver-infeasible and D-17 stops the whole run. Executed in
  `test_solver_reported_infeasibility_stops_before_executor` (2-block pool,
  8-token prompt). Not changed: equality acceptance is the approved rule, and
  any stricter check is a new admission decision.
- *Changed relaxed values in existing cases.* The resident row forces
  `z_A >= I_B` when the resident limit is 1, so earlier fractional victims
  (`z_A = 0.5`) become `z_A = 1` (CPU objectives 3.5 → 3 and 79.5 → 79). The
  integer plans and all 11 decode-preemption steps are unchanged on CPU. The
  same derivation updated `check_lp_scheduler_preemption_gpu.py`
  (15.5 → 15) and `check_lp_scheduler_decode_preemption_reference_gpu.py`
  (319.5 → 319, A's decode demand 1 → 0); these are analytical, not GPU
  evidence.
- *Provisional interface choices.* `EngineArgs.lp_memory_reserve` and
  `lp_decode_memory_policy_id` remain as reject-only fields (any value fails
  with migration guidance) to keep later positional bindings; they are not
  written to `config.yml`. A precommit failure carries outcome
  `implementation_or_validation_failure`.
- *Clocks.* Engine arrival uses `time.perf_counter` and the scheduler
  `time.monotonic`; on this node both are `clock_gettime(CLOCK_MONOTONIC)`.
  Inherited; not changed.

Not established: real engine initialization, GPU behavior, sampler/output
semantics beyond §16, physical block-ID equality, policy recovery, fairness,
restart freedom, completion, or performance. Coefficient formulas and scales,
optional κ, and Q11 remain OPEN.


## Review-finding fixes (2026-10-13; awaiting review)

**Provenance.** Tested base `13c2df4cca7df38f06d24460b48d142aba6f1136` plus
the uncommitted changes in `working_tree.diff` (seven modified files and the
new `tests/test_check_lp_scheduler_ordering.py`, which the repository's
`test*` ignore rule hides from `git status`; SHA-256 per file in
`sha256sums.txt`). Evidence:
`validation_output/lp_scheduler_integration/20261013T041256Z/`. Host
`gpu048`, allocation `65616965`, CPU only; Python 3.10.8 (`env`), NumPy
2.2.6, SciPy 1.15.3. This section and the table rows above marked 2026-10-13
were written after the run and are not tested code.

**Findings, reproduced on the starting code** (`repro/starting_code_observed.txt`):

1. *Obsolete helper order.* The contention, preemption, and decode-preemption
   GPU helpers rebuilt execution order as prefill/decode then identity, and
   victims by identity. Executed: the starting preemption helper's
   `check_decision` rejected a correct corrected-order record (one free
   block; zero-demand B emitted before one-block A). Fixed: one
   `check_declared_order` derives victims by victim rank, then identity, and
   execution as prefills, zero-demand decodes, then one-block decodes, each by
   service rank, then identity, from the mapped request facts, and requires
   the plan's declared lists and the emitted controls and metadata to equal
   it. Action, chunk, operation, replay, and state checks are unchanged.
2. *Explicit obsolete `None` accepted.* `EngineArgs(lp_memory_reserve=None)`
   and `lp_decode_memory_policy_id=None` were treated as omission. Fixed with
   a private, copy-stable omission default: any supplied value, including
   `None`, zero, or the old policy string, by keyword or position, fails with
   migration guidance before model construction or `config.yml`. Omission
   succeeds; the default is not serialized; field positions are unchanged.
3. *Oversized arrival.* `arrival_time=10**400` made the idle check raise
   `OverflowError`. Fixed by guarding that conversion: malformed arrival is
   not idle, and the mapper returns its structured failure (stage
   `state_mapping`, no solve or execution, no execution mutation; the
   inherited iteration counter advances).

**Observed after the fixes.** Ordering helpers 5, `EngineArgs` 11, live
scheduler and submission 21, existing script CPU tests 24: all pass (61).
Compile, warnings-as-errors import, and whitespace checks pass (the new file
checked separately: 0 problems). The pure, mapping, executor, and
`run_lp_inference` suites were not rerun: no changed file is imported by
them. No GPU, model, policy, or performance run.

**Remaining limitations.** The migrated helpers' GPU runs are unexecuted.
A mapping failure raised during capture carries no snapshot ID (existing
mapper behavior, unchanged). Admission equality, OPEN coefficient decisions,
and Q11 are unchanged.
