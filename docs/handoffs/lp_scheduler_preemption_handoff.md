# LPServe Native Preemption Execution Handoff

## Objective

Let the LP executor run every selected, legally eligible resident preemption
in a plan that also schedules prefill or decode work. The executor reuses
LPServe's native recomputation preemption and replay. Before this change it
rejected such plans as `unsupported_preemption`. Show the path on one live
CPU workload in which the real mapper, solver, and integer extraction select
the preemption, and the victim is later readmitted, recomputed, and
completed.

This is CPU-only evidence. No model was initialized, and no CUDA, GPU test,
or benchmark ran. The supported boundary is unchanged: one pipeline stage,
zero batches in flight, no overlapping state-changing engine calls, and one
synchronous decision from mapping through execution.

## Changed paths

- `lpserve_plan_execution.py`: victim prevalidation, native preemption
  execution, emitted `preempted_seq_ids`, and the updated final control
  check. The `unsupported_preemption` category was removed, and a new
  `preemption_recovery` category was added.
- `tests/test_lpserve_plan_execution.py`: the test holder gained the
  inherited `_preempt`/`_free_seq` helpers and a pool-size argument. One test
  was renamed and revised, and four tests were added (see below).
- `tests/test_lp_scheduler.py`: the harness gained one pool-size argument,
  used for both the central and worker managers. One live test was added.
- `docs/lp_scheduler_design.md`: §13 (native preemption execution, table
  row, scoped-values note), §15.4 traceability, the D-21 extension, and the
  D-03 row note.
- `docs/handoffs/lp_scheduler_preemption_handoff.md` (this file).
- `validation_output/lp_scheduler_preemption/20261010T054052Z/` (evidence).

Unchanged: the mathematical module, the mapper, the live scheduler,
configuration, native framework code, `docs/math/`, historical handoffs and
artifacts, and the pinned audit. The native paths this work relies on
(`BaseScheduler._preempt`/`_free_seq`, the block managers, the sequence
managers, `Sequence.reset_for_recompute`, `SchedulerOutputs`, and the engine
step) have no diff from the audit baseline `c3e0143`
(`git diff --stat c3e0143 HEAD -- <those paths>` printed nothing).

## Behavior implemented

- **Victims.** All victims come from the plan's `preempt` flags, regardless
  of the dominant, safety, or integral label. The executor checks them and
  calls `_preempt` in ascending `order_key`, before any scheduled action.
  `preempted_seq_ids` uses the same order. Because native `_preempt` inserts
  each victim at the front of `waiting`, several victims end in reverse call
  order. There is no cap on victims and no restriction on generated tokens.
- **Prevalidation (no mutation).** For each victim, the executor checks:
  - it is in the mapped legal preemption set;
  - it is currently owned by `running`, `PAUSED` (native executing),
    arrived, unfinished, allocated, and prompt-consistent;
  - its current physical block table has no repeated block and shares no
    block with another victim. Otherwise the plan fails as
    `preemption_recovery`.

  The table length is credited to a temporary free-block count, and one
  resident slot is released. Every scheduled action is then checked in
  execution order against that account: admissions need their full context,
  a resident slot, and the native watermark, and the decode append gate is
  enforced. A later failing action rejects the whole plan before any victim
  is freed. All earlier checks are retained.
- **Mutation.** Each victim is popped from `running` once, then
  `_preempt(seq)` runs. The executor does not reset the sequence; central
  replay does. Prefills and decodes follow as before (D-21).
- **Unchanged.** `no_progress` rejection of all-zero and preempt-only plans,
  prompt-ignore non-support, D-17 pre-mutation failure, and D-20
  post-mutation fail-stop.

## Environment

Unity node `gpu052.unity.rc.umass.edu`, SLURM job `65493676`. The work used
CPU only; the allocated GPU was not used. Modules: `uri/main`,
`Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`, and their dependencies (the
full list is in the evidence). Interpreter `env/bin/python` (Python 3.10.8),
numpy 2.2.6, scipy 1.15.3, torch 2.3.0+cu121. The repository root was on
`PYTHONPATH`. Tested base `d7e0c7d5d6f883b2fecd88455bcee594a411d8d1` on
`main`, plus the uncommitted changes listed above. The concurrent `docs/math`
edits reported on the Mac checkout were not present on Unity, whose tree was
clean at the start.

## Commands and observed results

Every execution shell ran `module load Python/3.10.8-GCCcore-12.2.0`,
`source env/bin/activate`, and
`export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`.

| Check | Before change (`baseline_checks/`) | After change (`cpu_checks/`) |
|---|---|---|
| `python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v` | 5 OK | 5 OK |
| `... -p 'test_lpserve_state_mapping.py' -v` | 7 OK | 7 OK |
| `... -p 'test_lpserve_plan_execution.py' -v` | 9 OK | 13 OK |
| `... -p 'test_lp_scheduler.py' -v` | 14 OK | 15 OK |

All exit statuses were 0. No tests were skipped or failed. `5-static.log`
records the remaining checks:

- `py_compile` of the three Python files and the trace script: exit 0.
- `git diff --check` of the four allowed tracked files: exit 0.
- A phase-terminology scan of the added Python lines and the trace script:
  no match (exit 1, as expected).

New or revised executor tests:

- `test_preemption_recovery_enables_admission`: a 6-block pool. The
  admission fails the native gate before recovery. The recorded call order
  is `_preempt(0)` then `_allocate(2)`. The test checks the complete queue
  and block transitions and that the omitted resident and omitted future
  waiting request are unchanged.
- `test_multiple_victims_use_ascending_native_and_control_order`: victims 1
  and 3, where 3 has two generated tokens. Calls are `_preempt(1)`,
  `_preempt(3)`, `_allocate(0)`. `preempted_seq_ids` is `[1, 3]` and
  `waiting` is `[3, 1, 4]`. The plan is hand-constructed and validated, so
  this is not live-selection evidence.
- `test_preemption_rejections_have_zero_mutation`: three rejections, each
  with no native call and an unchanged fingerprint:
  - an admission that fails after both recoveries are credited
    (`admission_gate`, "needs 4 blocks with 3 free");
  - a later victim that has finished (`ineligible_action`);
  - a repeated block in a victim's table (`preemption_recovery`).
- `test_control_only_plans_fail_progress_gate` (replaces the earlier
  progress-gate test): all-zero, single, and double preempt-only plans fail
  as `no_progress` without calling `_preempt`.
- `test_exception_after_native_preemption_propagates`: the second `_preempt`
  raises. Victim 1 stays freed and waiting; victim 3 has left `running` but
  keeps its blocks. There is no allocation, append, output, or recovery,
  and the fixture is discarded.

The live test is
`LivePreemptionTest.test_solver_selected_preemption_is_recomputed_to_completion`.

## Live CPU witnesses

The trace was produced by `python -B
validation_output/lp_scheduler_preemption/20261010T054052Z/trace_live_preemption.py
validation_output/lp_scheduler_preemption/20261010T054052Z` (exit 0, all
checks true). It uses the same harness and provisional inputs as the live
test:

- block size 4, a 4-block pool in both managers, `max_model_len` 32,
  resident limit 1;
- `b_max=4`, `c_max=4`, `s_max=1`, reserve 0, `conservative_one_block_v1`;
- utilities: prefill-token 1, decode 10, penalty 1; numerical policy
  `lp_relaxation_mvp_v1`;
- A: ID 0, prompt tokens 0..4. B: ID 1, prompt tokens 0..11. Both use
  `max_tokens=1`, `ignore_eos=True`, and synthetic token 7;
- arrival times 1.0 and 2.0, with decision times 1.0, 2.0, 3.0, and so on.

| Step | Emitted (preempted; scheduled) | Central state after replay |
|---|---|---|
| 0 | `[]`; `(0,4)` | A PAUSED, 4/5 processed, 2 blocks; 2 free in each manager |
| 1 | `[0]`; `(1,4)` | A WAITING, 0 processed, unallocated; B 3 blocks |
| 2–3 | `[]`; `(1,4)` | B prompt 8, then 12 |
| 4 | `[]`; `(1,0)` | B finishes (output `[7]`), freed |
| 5 | `[]`; `(0,4)` | A readmitted: full 2-block context, from token 0 |
| 6 | `[]`; `(0,1)` | A prompt complete |
| 7 | `[]`; `(0,0)` | A finishes (output `[7]`); all 4 blocks free in both managers |
| idle | none | no mapping or solver calls; iteration 8 |

At the step 1 boundary:

- The snapshot had `m_free=2`, `w=0`, legal set `{"0"}`, A's recovery 2,
  and B's fixed charge 3.
- The relaxed solution was A: x=0, y=0, I=0, z=0.5; B: x=4, y=0, I=1, z=0.
  The normalized objective was 3.5, with projection count 0. This matches
  the predicted bound `1 + 3q - max(0, (3q-2)/2)`, which is maximized at
  q=1.
- Extraction marked A as a dominant preemption. The plan was A preempt and
  B prefill 4, with integer objective 3.0.
- During replay, the worker block-manager calls were `free(0)` then
  `allocate(1)`. Central and worker copies of A were reset to WAITING with
  0 prompt tokens processed and the prompt unchanged.

After the multi-block frees, central and worker physical block IDs differ
(for example B was central `[3, 2, 1]` and worker `[2, 3, 1]`). This is the
inherited set-order free behavior, not a new defect. The tests compare
allocated sets, per-request counts, free counts, and per-manager pool
integrity instead.

## Evidence

All evidence is in `validation_output/lp_scheduler_preemption/20261010T054052Z/`:

| Path | Content |
|---|---|
| [`baseline_checks/`](../../validation_output/lp_scheduler_preemption/20261010T054052Z/baseline_checks/) | The four suites at unmodified HEAD (`*.log` files match an ignore rule and were force-added) |
| [`cpu_checks/environment.txt`](../../validation_output/lp_scheduler_preemption/20261010T054052Z/cpu_checks/environment.txt) | Host, job, modules, branch, HEAD, `git status`, versions, file hashes, and the hash of `git diff HEAD` |
| [`cpu_checks/tested_tracked_changes.diff`](../../validation_output/lp_scheduler_preemption/20261010T054052Z/cpu_checks/tested_tracked_changes.diff) | The exact tracked diff that was tested |
| `cpu_checks/1-…5-*.log` | Each command, its verbose output, and exit status |
| [`trace_live_preemption.py`](../../validation_output/lp_scheduler_preemption/20261010T054052Z/trace_live_preemption.py) | The trace script (attempt 2) |
| [`decision_trace.json`](../../validation_output/lp_scheduler_preemption/20261010T054052Z/decision_trace.json) | Per step: state before, snapshot, relaxed values, plan, outputs, state after execution, worker block calls, and state after replay; plus the idle record |
| [`summary.json`](../../validation_output/lp_scheduler_preemption/20261010T054052Z/summary.json) | Provenance (HEAD, `git status`, diff hash, file hashes), environment, inputs, witnesses, checks, and the pass flag |
| `trace_console.log` | Trace command output and exit status |
| [`trace_attempt1_comparison_bug/`](../../validation_output/lp_scheduler_preemption/20261010T054052Z/trace_attempt1_comparison_bug/README.md) | The failed first trace attempt, kept as recorded |

The first trace attempt exited 1 because the script compared tuples with
lists. The scheduler outputs it recorded were the same; only the script line
was fixed. `cpu_checks/environment.txt` therefore hashes the attempt-1
version of the trace script, and `summary.json` hashes the version that
produced the passing trace. The test and production hashes are identical in
both records.

## Limitations and open items

- **D-03 remains OPEN.** Runtime preemption now executes, with the penalty
  supplied only as an explicit scoped input; it is not a penalty policy.
  The design's D-03 row records this.
- **Generated tokens.** Interrupting a request after it has generated tokens
  is allowed but not validated live. The focused multi-victim test covers
  only the executor's central side of that case. The inherited native reset
  can extend the prompt IDs, clear the current generated IDs, permit
  overgeneration, and produce inconsistent output representations
  (§§16.1–16.2). This work does not repair or validate those semantics.
- **Not established:** GPU preemption, arbitrary-workload completion,
  fairness, pipeline support, prompt-ignore controls, corrected output
  semantics, independent framework correctness, or performance.
  Physical-ID equality between central and worker managers is not claimed.
- The `docs/math/` work, historical handoffs and artifacts, and the pinned
  audit were not modified.
