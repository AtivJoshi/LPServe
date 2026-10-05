# LPServe Plan Execution: Initial Native Executor Handoff

Evidence record for the first native executor of validated integer plans. It
records observed facts only and does not replace `docs/lp_scheduler_design.md`
(the normative source). It claims no phase acceptance.

## Provenance

| Item | Value |
|---|---|
| Base commit | `d464168b65ccf8ee12d5280f19db62a493ec736a` (`Refactor documentation for LP Scheduler and Unity setup`), matching the Mac planning checkout |
| Tested revision | Base commit plus the uncommitted changes listed below; re-verified after commit at implementation commit `0afba0d7a7d7f5952975332086e1cca953aff5c4` (`Add native executor for validated LP integer plans`) |
| Branch | `main`, tracking `origin/main`; no pull, reset, or other Git operation was performed |
| Host / allocation | Unity node `gpu048`, SLURM job `65249627`, partition `gpu-preempt`, 8 CPUs on node, `CUDA_VISIBLE_DEVICES=0`. All checks were CPU-only. No model, CUDA, GPU kernel, or benchmark was run. |
| Environment | `module load Python/3.10.8-GCCcore-12.2.0` then `source env/bin/activate`, run in the same shell as each command |
| Interpreter | `sys.executable` = `/home/atjoshi_umass_edu/LPServe/env/bin/python`; Python 3.10.8 (GCC 12.2.0) |
| Dependencies | NumPy 2.2.6, SciPy 1.15.3, torch 2.3.0+cu121, `sarathi` 0.1.7 (editable checkout) |

Pre-edit state: HEAD `d464168`; the working tree had only the pre-existing
untracked `CLAUDE.md`, which this task did not touch. `git diff --stat c3e0143
HEAD -- sarathi` was empty: the framework code is unchanged since the commit
the architecture audit is pinned to.

## Exact changed paths

Implementation commit `0afba0d` contains `lpserve_plan_execution.py`,
`tests/test_lpserve_plan_execution.py` (added with `git add -f`), and
`docs/lp_scheduler_design.md`. This handoff is added in a separate following
commit; its own hash is reported outside this file. Paths as originally
changed:

- `lpserve_plan_execution.py`: new.
- `tests/test_lpserve_plan_execution.py`: new. `.gitignore:205` (`test*`)
  matches it, so `git status` does not list it. The existing test files are
  tracked anyway; committing this one needs `git add -f`.
- `docs/lp_scheduler_design.md`: modified.
- `docs/handoffs/lpserve_plan_execution_handoff.md`: new (this file).

No other file was changed. The mathematical module, mapper, existing tests,
framework, configuration, guide, status, architecture audit, and historical
handoffs are all unchanged.

## Implemented behavior

The public operation is `lpserve_plan_execution.execute_plan(scheduler,
snapshot, result)`. It is one synchronous call. On success it returns native
`SchedulerOutputs`. On a rejection before mutation it returns an immutable
`lp_relaxation_scheduler.Failure` record. It never invokes the mapper or
solver.

Supported actions:

- **Waiting admission.** The executor removes the waiting entry by identity
  (preserving the order of other waiting entries), calls the inherited
  `_allocate`, and appends the request to `running` once. Allocation demand is
  `block_manager.get_num_initial_blocks(seq)`, which is the full logical
  context for `VLLMBlockSpaceManager` regardless of chunk size.
- **Resident partial prefill.** The block gap must be zero. Nothing is
  allocated; positive metadata is emitted.
- **Decode.** The executor calls the inherited `_append_slot` and emits a zero
  chunk. The block gap must be 0 or 1.
- **Prompt-first mixed output.** All prefills come first, then all decodes,
  each group ascending by `order_key` (D-21). Central operations and emitted
  metadata use this same order.

**Precommit validation** reads state only and does the following:

- Checks input and result types. An upstream `Failure` or `MappingFailure` is
  rejected.
- Checks that the snapshot, problem, result, and plan IDs, plus
  `snapshot.scheduler_iteration_id`, all match the current `_iteration_id`.
- Requires one pipeline stage, zero running batches, and a current resident
  count within the limit.
- Reuses `validate_problem` and `validate_integer_plan` unchanged.
- Applies the progress gate.
- Checks that snapshot request records associate one-to-one with the problem.
- Requires current `waiting`/`running` ownership to be unique by `seq_id`.
- For each selected request, checks: ownership; status (`WAITING` or
  `PAUSED`); arrival at or before `snapshot_time`; not finished; prompt
  progress consistent with completion; allocation; block gap; and chunk at most
  min(current remainder, `c_max`).
- Before an admission, checks the length against `max_model_len` (read-only;
  `_check_request_prompt_length` is never called) and checks resident capacity.
- Proves memory feasibility with a running count of free blocks, applied in
  execution order:
  - Admission gate: `free - demand >= block_manager.watermark_blocks`.
  - Append gate: `free > 0`, then subtract the actual gap.

The allocator and block tables are not copied.

After mutation, the final checks are:

- `len(running) <= max_num_seqs`;
- emitted `(seq_id, prompt_chunk_len)` equals the validated ordered actions;
- the control lists are empty;
- emitted token and action counts, from the native output properties, are
  within `b_max`/`s_max`.

A failure in these checks raises `RuntimeError`. An exception from a native
operation propagates without being caught.

Not done by the executor:

- incrementing `_iteration_id` or `num_running_batches`;
- changing sequence status or prompt progress;
- auditing ownership or allocation after execution;
- retry, rollback, recovery, or partial output;
- classifying SLAI criticality (the counter fields keep their defaults).

**Precommit rejection boundaries.**

- Stage `precommit_validation`, with one of these categories:
  - `upstream_failure`, `malformed_input`, `snapshot_mismatch`;
  - `unsupported_state` (pipeline stages, in-flight batch);
  - `no_progress` (all-zero or preempt-only plan);
  - `unsupported_preemption` (execution plus any selected preemption);
  - `ownership_mismatch`, `ineligible_action`, `chunk_bound`;
  - `overlength_admission`, `resident_capacity`;
  - `admission_gate`, `append_gate`.
- `validate_problem`/`validate_integer_plan` failures are returned unchanged
  (stage `input_validation`/`plan_validation`).
- Ordinary exceptions raised by malformed scheduler objects before mutation
  are not converted into failures. They propagate, and no mutation has
  happened at that point.

## Approved decisions recorded in `docs/lp_scheduler_design.md`

- §1.5: removed metadata ordering from the "no implicit default" list, and
  added a pointer to D-21.
- §13: added "Execution and metadata order (D-21, approved 2026-10-05)" and
  "Initial executor supported subset (approved 2026-10-05)". The subset
  records that runtime preemption and ignore controls are deferred; that this
  changes neither mathematical eligibility nor extraction; and that live
  integration, idle handling, and D-17 exception conversion are outside the
  executor.
- §17: added a D-21 RESOLVED row with its rationale and focused test, and
  removed D-21 from the OPEN table.

All other OPEN decisions and BLOCKERs are unchanged. D-03 (preemption penalty)
remains OPEN "before runtime preemption is enabled". The inherited
mixed-batch/sampler limitations are referenced to §16.4 and D-25 and were not
investigated.

## Scoped fixture inputs (not production defaults)

Common inputs:

- `B_max=8`, `C_max=4`, `S_max=3`;
- resident limit 4, block size 4, 10 total blocks, `max_model_len=32`;
- planning reserve 1, decode policy `conservative_one_block_v1`;
- numerical policy `lp_relaxation_mvp_v1` (`1e-7`, `1e-6`, `1e-9`, `1e-9`);
- iteration 7, snapshot time `100.0`;
- native watermark 0.01, which is 0 blocks.

The combined-memory case uses watermark 0.2 (2 blocks). The planning reserve
is not the native watermark.

Utilities are given as `RequestUtility(decode, prefill_token, penalty)` keyed
by raw ID:

- **Admission + resident prefill:** 0 → (0, 2, 0); 1 → (0, 1, 0.25);
  2 → (0.5, 0, 0.5). This is the mapper fixture except that request 2's decode
  utility is lowered from 3.0 to 0.5, so the unique optimum leaves request 2
  unselected.
- **Decode gaps / append gate:** (1, 0, 1) for each request.
- **Mixed:** 0 → (3, 0, 0.5); 1 → (0, 2, 0); 2 → (0, 1, 0.25).
- **Combined memory:** 0, 1 → (0, 1, 0); 2 → (0, 0.1, 1); 3 → (0.1, 0, 1).

The approved later live demonstration values (uniform decode utility 1,
prefill-token utility 1, preemption penalty 1) are not implemented here.

## Focused tests (`tests/test_lpserve_plan_execution.py`)

The tests use a real `Sequence`, a real `VLLMBlockSpaceManager`, and the
native `SchedulerOutputs`/`SequenceScheduleMetadata`. The scheduler holder
borrows `BaseScheduler._allocate`/`_append_slot` without initializing an
engine.

1. `test_admission_and_resident_prefill_through_mapper_and_solver`: runs the
   real mapper and `solve_and_extract`.
   - Emitted `[(0,4),(1,4)]`; output ID 7; 8 prompt tokens, 0 output tokens.
   - `waiting` goes from `[0, 3]` to `[3]`, where 3 is an unrelated future
     request. `running` goes from `[1, 2]` to `[1, 2, 0]`.
   - The admission allocates 2 blocks (the full context of a 6-token prompt,
     for a 4-token chunk). The resident prefill allocates 0. Free blocks go
     7 → 5.
   - Unrelated request 2's table, and the fingerprints of all four sequences
     (status and progress), are unchanged.
   - `_iteration_id` and `num_running_batches` are unchanged.
2. `test_decode_with_block_gap_zero_and_one`: runs the mapper and solver.
   Gap 0 appends no block; gap 1 appends one. Free blocks go 8 → 7. Emitted
   `[(0,0),(1,0)]`.
3. `test_append_gate_rejects_gap_zero_decode_without_free_block`: the plan is
   validated as legitimate first. The test then holds all 9 free blocks
   outside the snapshot, so current state differs from the snapshot. Result:
   `append_gate` with zero mutation (fingerprint).
4. `test_mixed_output_is_prompt_first_in_native_execution_order`: emitted
   `[(1,4),(2,3),(0,0)]`, which is not the global-ID order `[0,1,2]`. The
   recorded native calls are `[allocate 1, append_slot 0]`. Free blocks go
   7 → 4. This test does not establish model or sampler correctness.
5. `test_combined_memory_rejection_with_native_watermark`: the planning
   residual memory is 0, and the plan is feasible.
   - Checked one at a time against the original free count, native
     `can_allocate` is True for both admissions.
   - In execution order, the second admission sees 3 free blocks, and
     3 − 2 < watermark 2, so the result is `admission_gate` on seq 1 with zero
     mutation.
6. `test_compact_precommit_rejections`: 12 subtests, each with zero mutation
   and problem ID `"7"`. The cases are:
   - iteration mismatch;
   - upstream failure;
   - in-flight batch;
   - pipeline stages;
   - resident not owned;
   - waiting request already allocated;
   - arrival after the decision;
   - chunk above the current remainder;
   - resident capacity;
   - overlength admission;
   - plan chunk bound (reported by the validator);
   - plan token limit (reported by the validator).
7. `test_no_progress_and_selected_preemption_gate`: each constructed plan is
   first confirmed legitimate with `validate_integer_plan`. Results:
   all-zero → `no_progress`; preempt-only → `no_progress`; admission plus
   preemption → `unsupported_preemption`. All have zero mutation.
8. `test_post_mutation_exception_propagates_without_recovery`: a test double
   makes `_allocate` raise after the admission was popped from `waiting`.
   - The `RuntimeError` propagates and no output is returned.
   - `_append_slot` is never called.
   - The popped request is not restored.
   - The fixture is discarded.

## Commands and observed results

Before the edits (same shell, on `gpu048`):

```text
$ python -B lp_relaxation_scheduler.py
lp_relaxation SUCCESS problem_id=smoke-problem ... exit 0
$ python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
Ran 5 tests ... OK
$ python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
Ran 7 tests ... OK
```

This is the first retained mapper test evidence at a revision that includes
the `2c31d5b`, `1996b68`, and `78d663a` simplifications.

After the implementation (same shell, on `gpu048`):

```text
$ python -m py_compile lpserve_plan_execution.py tests/test_lpserve_plan_execution.py
exit 0
$ python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
Ran 5 tests in 0.013s  OK
$ python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
Ran 7 tests in 0.011s  OK
$ python -B -m unittest discover -s tests -p 'test_lpserve_plan_execution.py' -v
Ran 8 tests in 0.047s  OK
$ git diff --check
exit 0 (no output)
$ git diff --no-index --check /dev/null <each new file>
no output (exit 1 only because the file differs from /dev/null)
$ rg -n -i 'phase' lpserve_plan_execution.py tests/test_lpserve_plan_execution.py
no matches, exit 1 (expected clean result)
$ git diff --name-only
docs/lp_scheduler_design.md
$ git status --short --branch --untracked-files=all
## main...origin/main
 M docs/lp_scheduler_design.md
?? CLAUDE.md
?? lpserve_plan_execution.py
```

The handoff file was added after these commands. The test file is not listed
because it is ignored.

Post-commit re-verification at `0afba0d` on `gpu048` (same environment):
`py_compile` of both new files exit 0; the math tests (5), mapper tests (7),
and executor tests (8) each ran OK; `git show --check HEAD` reported no
whitespace errors. The working tree then contained only the untracked
`CLAUDE.md` and this handoff.

Check summary:

- **Passed:** all of the above.
- **Failed:** none.
- **Skipped:** none.
- **Not run:** live scheduler integration, engine/worker replay, model
  execution, GPU, and performance. All of these are outside this task.

## Inherited limitations and unverified scope

The inherited limitations are §16.3 (control-only outputs), §16.4
(mixed-batch/sampler association), §16.5 (non-transactional mutation), and
§16.6 (pipeline). They are referenced here, not repaired.

The following remain **unverified**:

- live `LPScheduler` integration;
- the ordinary idle path;
- D-17 exception conversion;
- central/worker replay equality;
- model execution and GPU correctness;
- general mixed-sampling correctness;
- performance.

The state-change case in the append-gate test violates the D-19 contract on
purpose. It shows that current-state validation catches the change; it does
not establish support for concurrent changes.
