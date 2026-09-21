# LP-Relaxation Zero-Valued Action-Tie Fix — Handoff

## Provenance

- Branch: `main`
- Base commit: `6ddba39e12db885c5baa8b4e5262fd0b62155886`
- Fix commit: `32036f6a0ddb9bf2b55fc81f48967d61d558b4c2`
- Environment: local; `module load uri/main Python/3.10.8-GCCcore-12.2.0` then `source env/bin/activate` (Python 3.10.8, SciPy 1.15.3, NumPy 2.2.6). Bare `python` without the module load fails with a missing `libpython3.10.so.1.0`.

## Changed paths (fix commit)

- `lp_relaxation_scheduler.py` (+2)
- `tests/test_lp_relaxation_scheduler.py` (+31, no deletions)
- `docs/lp_scheduler_design.md` (§10.6)
- `docs/math/main-llm-serving.tex` (Safe Fractional Extraction, step 3)

This handoff document is the only file in the second commit.

## Approved behavior

For an unpreempted fractional request, if normalized relaxed `y == 0.0` and normalized relaxed prefill indicator `I == 0.0`, no execution action is selected. Otherwise `y` and `I` are compared as before; an exact positive tie still chooses decode. Eligibility, capacity, ordering, preemption, validation, and failure rules are unchanged. No new tolerance, fallback, helper, or configuration was added. This resolves the conflict between design §10.6 (every exact tie chooses decode) and §11.4 (every selected decode must be eligible).

## Deviation from the specified regression scenario

The instruction's scenario (two legal candidates with `z=0.5`, `y=I=0`, recovery `2`, safety repair after the locked prefill) cannot reach the zero/zero branch. Under design §10.4, `z=0.5` strictly exceeds `y=I=0` with `c^Z=2>0`, so both candidates are selected as dominant preemptions before packing. Against the base code, that scenario failed only on `safety_preemption_ids == ()`, not because of the `y >= I` promotion.

With user approval, the regression instead uses one legal-preemption candidate with recovery `0` (never a dominant or safety victim, per §§10.4–10.5), decode-ineligible, relaxed `x=0, y=0, I=0, z=0.5` (with preemption penalty `0.5` in the original commit; superseded by the Addendum below, which sets the penalty to `0`), beside a locked prefill (fixed charge 2, `m_free=2`). This scenario does not exercise safety repair. The safety-repair path with a tie remains covered by the pre-existing `test_mixed_case_and_tied_extraction`. Design §10.4/§10.5 were not changed.

## Commands and observed results

Saved output from a single run:

```
$ python --version
Python 3.10.8
$ python -m py_compile lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py
exit=0
$ python lp_relaxation_scheduler.py
lp_relaxation SUCCESS problem_id=smoke-problem
  numerical_policy: lp_relaxation_mvp_v1
  solver: highs-ds scipy=1.15.3 status=0 iterations=2
  relaxed objective (max): 12.0
  integer objective (max): 12.0
  plan (ascending order_key):
    smoke-prefill order_key=(0, 10) action=prefill prefill_tokens=2 decode=0 preempt=0 prefill_indicator=1
    smoke-decode order_key=(0, 20) action=decode prefill_tokens=0 decode=1 preempt=0 prefill_indicator=0
  totals: prefill_tokens=2 decode_actions=1 preemptions=0
  residual capacities: tokens=0 actions=0 memory=0
  fractional_request_count: 1
  dominant_preemption_ids: ()
  safety_preemption_ids: ()
exit=0
$ python -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
test_fractional_prefill (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_main_smoke (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_mixed_case_and_tied_extraction (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_visible_infeasibility (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_zero_zero_execution_tie_is_no_action (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.013s

OK
exit=0
$ grep -n -i "phase" lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py
exit=1   (no matches)
```

Regression sensitivity check (not a repository command): the new test file run against `git show 6ddba39:lp_relaxation_scheduler.py` in a scratch directory gave `Ran 5 tests ... FAILED (failures=1)`; the failing test was `test_zero_zero_execution_tie_is_no_action`, with `(0, 1, 0, 0) != (0, 0, 0, 0)` (the candidate was promoted to an ineligible decode). The other four passed.

## Check status

- Passed: `py_compile` (exit 0); `python lp_relaxation_scheduler.py` (exit 0, SUCCESS); unit tests (5 of 5 pass); `grep -i phase` on both Python files (no matches); regression fails on base code and passes on fixed code.
- Failed: none.
- Skipped: formatter/lint check. `ruff`, `flake8`, `black`, `pycodestyle`, and `pyflakes` are not installed, and nothing was installed for this.
- Unexecuted: LaTeX compilation of `docs/math/main-llm-serving.tex` (the edit is two lines inside an existing `algorithmic` block; not built); any state-mapping, integration, or GPU test.

## Test-suite confirmation

- The four original tests are unchanged (the test-file diff is additions only: +31, 0 deletions) and pass.
- Exactly one regression was added: `test_zero_zero_execution_tie_is_no_action`.

## Unresolved issues and limitations

- The regression covers only the recovery-0 route to a zero/zero pair; no other route exists under §10.4 (a fractional request with `y=I=0` must have fractional `z`, which is a dominant preemption whenever `c^Z>0`).
- The `tests/` directory matches an ignore rule reported by `git add` on the first attempt (`The following paths are ignored ... tests`), although `git check-ignore` reports no match for the file and it is tracked. The four paths were staged and committed normally. The cause was not investigated.

## Scope confirmation

No state mapping, execution, framework integration, or unrelated repair was performed. The LP formulation, objective, constraints, tolerances, and eligibility definitions were not changed.

## Working tree before committing this handoff

```
?? CLAUDE.md
```

`CLAUDE.md` was untracked at the start of the task and is unrelated.

## Addendum: regression optimality correction

- Correction commit: `db3a64afbe1aec2b6b11d477ced4f2f888288a06` (base `d0dc4280bbd45c658fda15664af24d1e023a4d8b`; changes only `tests/test_lp_relaxation_scheduler.py`, one line: `cand` utilities `(0, 0, 0.5)` -> `(0, 0, 0)`).
- Why the former point was not optimal: with recovery `0`, `z=0.5` frees no memory but costs `0.5 * 0.5 = 0.25` in the objective. The point was feasible with objective `0.75`, while the LP optimum was `1.0` (solver `z=0`), so an optimal-only solver pipeline could never deliver it to extraction. The statement above that the former point was a valid extraction input is superseded.
- Current status: with recovery `0` and penalty `0`, `cand.z` has no effect on feasibility or objective, so `z=0.5` is a degenerate optimal relaxed solution. Observed in a scratch check (not added to the repository): the solver returned `optimal_candidate`, status `0`, minimization objective `-1.0` (maximization `1.0`, raw vector `(1, 0, 0, 0, 1, 0, 0, 0)`). The hand-authored point `(1, 0, 0, 0, 1, 0, 0, z)` passed `validate_relaxed_solution` with maximization objective `1.0` for each of `z = 0.0, 0.5, 1.0`, equal to the solver optimum.
- No test was added; the existing regression is unchanged except for the one-line utility change, so the suite still has five tests.

### Commands and observed results (after correction)

Same environment as above (Python 3.10.8 via the Unity module and `env`).

```
$ python -m py_compile lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py
exit=0
$ python lp_relaxation_scheduler.py
exit=0   (output not repeated; unchanged production code)
$ python -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
test_fractional_prefill ... ok
test_main_smoke ... ok
test_mixed_case_and_tied_extraction ... ok
test_visible_infeasibility ... ok
test_zero_zero_execution_tie_is_no_action ... ok
Ran 5 tests in 0.013s
OK
$ grep -n -i "phase" lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py
exit=1   (no matches)
```

### Pre-fix sensitivity (corrected test file vs. `git show 6ddba39e12db885c5baa8b4e5262fd0b62155886:lp_relaxation_scheduler.py` in a scratch directory)

`Ran 5 tests in 0.031s`, `FAILED (failures=1)`. The four original tests passed. `test_zero_zero_execution_tie_is_no_action` failed with `AssertionError: Tuples differ: (0, 1, 0, 0) != (0, 0, 0, 0)` (the candidate was promoted to an ineligible decode).

### Check status

- Passed: `py_compile`, smoke run, 5 of 5 unit tests on the fixed code, `grep` (no matches), pre-fix failure of the corrected regression only, and the scratch optimality check.
- Failed: none other than the expected pre-fix regression failure.
- Skipped/unexecuted: formatter/lint (tools not installed); LaTeX build; anything beyond CPU unit checks.

### Working tree

Before this handoff update, after the correction commit: `?? CLAUDE.md` only (pre-existing, unrelated).
