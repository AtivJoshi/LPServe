# LPServe Plan Execution: Pre-Mutation Failure-Interface Correction Handoff

Evidence record for a narrow correction to `execute_plan()`. The correction
implements the existing pre-mutation failure contract
(`docs/lp_scheduler_design.md` §§13–14.3 and D-17). It changes no design
decision. The earlier `docs/handoffs/lpserve_plan_execution_handoff.md` is
preserved unchanged.

## Provenance

| Item | Value |
|---|---|
| Base revision | `74f5d095f249abbe91267f315869f283ea65d2a0` (`docs: record LPServe plan-execution handoff`), on `main` and in sync with `origin/main` |
| Reviewed commits included | `0afba0d7a7d7f5952975332086e1cca953aff5c4` (implementation) and `74f5d09` (handoff), both ancestors of HEAD, confirmed with `git merge-base --is-ancestor`; there are no newer commits |
| Tested revision | Base plus the uncommitted changes to `lpserve_plan_execution.py` and `tests/test_lpserve_plan_execution.py`; re-verified after commit at correction commit `f053afd74f8f91db5d8feefd91dfc1f8d428717e` (`Return structured failure for malformed nested executor inputs`) |
| Pre-existing unrelated state | Untracked `CLAUDE.md`, left untouched |
| Host / allocation | Unity node `gpu048`, SLURM job `65249627`, partition `gpu-preempt`; CPU-only work, with no model, CUDA, GPU, or benchmark run |
| Environment | `module load Python/3.10.8-GCCcore-12.2.0`, then `source env/bin/activate`, in the same shell as each command |
| Interpreter / dependencies | `/home/atjoshi_umass_edu/LPServe/env/bin/python`, Python 3.10.8; NumPy 2.2.6, SciPy 1.15.3, torch 2.3.0+cu121 |

## Defect and correction

Before the fix, `execute_plan()` checked the outer `StateSnapshot` and
`SchedulingSuccess` types and then read `snapshot.lp_problem.problem_id` and
`result.plan.problem_id` without validating those nested records. A `None`
nested record therefore raised `AttributeError` instead of returning a
structured failure. No mutation could occur, because this happens before
execution, but the result violated the failure contract.

The correction adds two explicit `isinstance` checks immediately after the
outer-input checks and before either record is dereferenced:

- `snapshot.lp_problem` must be an `lrs.LPProblem`;
- `result.plan` must be an `lrs.IntegerPlan`.

A failed check returns `lrs.Failure` with stage `precommit_validation`,
category `malformed_input`, and the snapshot ID (read with `getattr` as
before). The reason names the record (`snapshot.lp_problem` or `result.plan`)
and the type it received.

The following are unchanged:

- the outer-input checks and the reused mathematical validators;
- valid-input behavior and execution order;
- the post-mutation path, which still catches nothing from `_execute()`.

No broad exception handler, retry, fallback, rollback, recovery, or partial
output was added.

## Exact changed paths

- `lpserve_plan_execution.py`: two type checks (9 added lines).
- `tests/test_lpserve_plan_execution.py`: one new test,
  `test_malformed_nested_records_return_failure`.
- `docs/handoffs/lpserve_plan_execution_failure_interface_handoff.md`: new
  (this file).

## Focused regression

`test_malformed_nested_records_return_failure` builds the valid
admission/resident-prefill fixture through the real mapper and solver. It then
uses `dataclasses.replace` to make two subcases:

1. `plan_none`: a valid snapshot with `SchedulingSuccess(plan=None)`;
2. `lp_problem_none`: `StateSnapshot(lp_problem=None)` with an otherwise valid
   result.

Each subcase asserts:

- the call returns a `Failure`, with stage `precommit_validation`, category
  `malformed_input`, and problem ID `"7"` (the snapshot ID);
- the reason contains the name of the malformed record;
- the complete scheduler-state fingerprint is unchanged;
- the spies on `_allocate` and `_append_slot` were never called.

## Commands and observed results

Expected failure before the fix, with only the new test added:

```text
$ python -B -m unittest discover -s tests -p 'test_lpserve_plan_execution.py' -k test_malformed_nested_records_return_failure -v
ERROR: ... [plan_none]       AttributeError: 'NoneType' object has no attribute 'problem_id'
ERROR: ... [lp_problem_none] AttributeError: 'NoneType' object has no attribute 'problem_id'
Ran 1 test ... FAILED (errors=2)
```

Final verification after the fix:

```text
$ python -m py_compile lpserve_plan_execution.py tests/test_lpserve_plan_execution.py
exit 0
$ python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
Ran 5 tests in 0.014s  OK
$ python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
Ran 7 tests in 0.010s  OK
$ python -B -m unittest discover -s tests -p 'test_lpserve_plan_execution.py' -v
Ran 9 tests in 0.044s  OK
$ git diff --check
exit 0 (no output)
$ rg -n -i 'phase' lpserve_plan_execution.py tests/test_lpserve_plan_execution.py
no matches, exit 1 (expected clean result)
$ git diff --name-only
lpserve_plan_execution.py
tests/test_lpserve_plan_execution.py
$ git status --short --branch --untracked-files=all
## main...origin/main
 M lpserve_plan_execution.py
 M tests/test_lpserve_plan_execution.py
?? CLAUDE.md
```

This handoff was written after these commands. The mathematical suite ran in
its own process.

Post-commit re-verification at `f053afd` on `gpu048` (same environment):
`py_compile` exit 0; math (5), mapper (7), and executor (9) tests each OK;
`git show --check HEAD` reported no whitespace errors. This handoff is added
in a separate following commit.

Check summary:

- **Passed:** all of the final checks above. All 8 earlier executor tests pass
  unchanged, including the valid-execution cases and
  `test_post_mutation_exception_propagates_without_recovery`.
- **Final failures:** none. The two pre-fix errors are the expected regression
  evidence, not final failures.
- **Skipped:** none.
- **Not run:** live `LPScheduler` integration, engine/worker replay, model
  execution, GPU validation, and performance.

No live integration, GPU correctness, or phase acceptance is claimed.
