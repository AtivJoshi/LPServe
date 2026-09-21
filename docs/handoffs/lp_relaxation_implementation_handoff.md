# LP-Relaxation Scheduling Layer: Implementation Handoff

Evidence record for the standalone LP-relaxation module. It records observed
facts only; it does not amend `docs/lp_scheduler_design.md` or
`docs/math/main-llm-serving.tex`.

## Provenance

| Item | Value |
|---|---|
| Implementation commit | `f406eeaac7bafc4478304744c9187316b75b8f67` |
| Base commit | `8240f0667e2f9045eccf9ec042cd3f902cffc6f4` |
| Branch | `main` (local was ahead of `origin/main` by the implementation commit only) |
| Host used | Unity node `gpu048`, CPU-only work; no GPU, CUDA, or model touched |
| Python | 3.10.8 (`/home/atjoshi_umass_edu/LPServe/env`, after `module load uri/main` and `module load Python/3.10.8-GCCcore-12.2.0`) |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 (pinned) |

State before editing: branch `main`, HEAD `8240f06`, working tree with only
untracked `CLAUDE.md` (pre-existing, not part of this work, left untouched and
uncommitted).

Note: the venv interpreter fails with `libpython3.10.so.1.0: cannot open shared
object file` unless the Python module from `docs/unity_setup.md` §3 is loaded
first. This is an environment fact, not a change.

## Implementation paths changed (commit `f406eea`)

- `lp_relaxation_scheduler.py` (new)
- `tests/test_lp_relaxation_scheduler.py` (new)
- `requirements.txt` (one line added: `scipy == 1.15.3`)

`.gitignore` line 205 (`test*`) ignores the tests file, so it was added with
`git add -f`. `.gitignore` was not modified.

## Behavior implemented

- Frozen dataclasses for request input, problem, numerical policy, solver
  diagnostics and result, relaxed decisions, integer decisions and plan, and
  `Failure`/`SchedulingSuccess`. Input validation sorts requests by ascending
  `order_key` and rejects duplicates, malformed or nonfinite values,
  inconsistent bounds or eligibility, and inconsistent legal-preemption
  membership.
- LP per design §7 through `scipy.optimize.linprog` with `method="highs-ds"`,
  `options={"presolve": True}`, and no other controls. The maximization is
  converted to minimization explicitly. Variable layout is
  `[x.., y.., I.., z..]` in sorted request order.
- Raw SciPy output is normalized to a project-owned `SolverResult`. Only
  status `0` with `success is True`, a finite vector of the right length, and a
  finite objective that agrees with the recomputed objective proceeds. Statuses
  1-4, unknown status, exceptions, interruption, and malformed output become
  non-success categories.
- Relaxed-solution validation recomputes bounds and all constraints from
  project data on both raw and normalized vectors. It allows only the
  single-variable projection of design §11.2, preserves the raw vector, and
  records projection count and worst violations.
- Extraction follows design §10 (locking with floor of `x`, residuals,
  dominant preemption with strict inequality, safety preemption by largest `z`
  then smallest `order_key`, fractional packing, canonicalization). Plan
  validation uses exact integer arithmetic (design §11.3, and the semantic
  checks of §11.4 that apply without live state).
- `main()` runs the hard-coded smoke case and prints the plan.

Numerical policy (visible, provisional, ID `lp_relaxation_mvp_v1`): absolute
feasibility tolerance `1e-7`, absolute indicator integrality tolerance `1e-6`,
objective absolute and relative tolerances `1e-9`. These and the fixture values
in the tests and `main()` are scoped MVP inputs, not permanent project policy.

## Commands run and complete observed output

Retained output (run after the final source edit and before the implementation
commit):

```text
$ python -m py_compile lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py
[exit 0]

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
[exit 0]

$ python -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
test_fractional_prefill (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_main_smoke (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_mixed_case_and_tied_extraction (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_visible_infeasibility (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.007s

OK
[exit 0]
```

Other checks run:

- `grep -n -i "phase"` over both Python files: no matches (naming rule).
- `awk 'length>88'` over both Python files: no lines over 88 characters.
- `grep -n -P "\t| +$"` over both Python files: no tabs or trailing whitespace.
- An inline AST check over both Python files found no unused imports.

Observations from an ad-hoc inline script (not retained in the repository)
that printed the solver vectors for the test problems:

- Mixed case: HiGHS returned an integral vertex (`admit` prefill 2, `decode`
  decode, `victim-b` preempted, `victim-a` untouched), relaxed and integer
  objective both `10.5`, fractional count `0`. The end-to-end run therefore
  did not exercise tied extraction; the tied relaxed point in the test is a
  hard-coded input to validation and extraction, as specified.
- Fractional case: vector `y_decode=0.5`, `x_prefill=1.5`, `I_prefill=0.5`,
  relaxed objective `3.5`, integer objective `2.0` (prefill 2 tokens, decode
  skipped because `M_curr=1 < c^D=2`), fractional count `2`. No approximation
  claim is made.
- Infeasible case: HiGHS status `2`, message `The problem is infeasible.
  (HiGHS Status 8: model_status is Infeasible; primal_status is None)`; result
  is a `Failure` at stage `solver`, category `infeasible`, with no plan. The
  message text is non-contractual.

## Check status

- **Passed:** `py_compile`; `python lp_relaxation_scheduler.py`; four unit
  tests; the manual checks listed above.
- **Failed:** none observed after the final source edit.
- **Skipped:** a formatter or linter. `flake8`, `ruff`, `black`, `pyflakes`,
  `pycodestyle`, `pylint`, `mypy`, and `isort` are not installed in the
  environment (`pip list` showed only NumPy and SciPy among the searched
  names); nothing was installed.
- **Unexecuted:** state mapping, execution, integration, GPU smoke, and
  performance checks (out of scope). Test coverage is the four specified cases
  only; no numerical-boundary, degeneracy, projection-path, or other
  failure-path tests (only the infeasible solver path) were added, per design
  §15.1.

## Dependency and environment changes

- `requirements.txt`: added `scipy == 1.15.3`, the exact version observed
  installed in the Unity environment. Nothing was installed or upgraded.
  `numpy` remains unpinned as it was before.

## Assumptions and limitations

No exact conflict between `docs/lp_scheduler_design.md` and
`docs/math/main-llm-serving.tex` was found for the read passages (design §§7,
9-11, 14-17; TeX myopic ILP and Primal Heuristic 1). Points the sources leave
open, resolved minimally in code and visible here:

- An empty request set is rejected as invalid input rather than producing an
  empty plan.
- Counts, charges, capacities, `M_free`, and `W` must be non-negative
  integers; `C_max` must be positive; `W > M_free` is accepted. Utilities must
  be finite (no sign restriction). Request IDs are non-empty strings.
- A prefill-ineligible request must have `prefill_upper_bound == 0`; a
  prefill-eligible one must have positive remainder and
  `prefill_upper_bound == min(remainder, C_max)`; decode-eligible requires
  zero remainder; recovery must be zero outside the legal-preemption set.
- Extraction follows the literal rule "larger of `y` and `I`, tie chooses
  decode". A fractional request whose relaxed `y` and `I` are both zero would
  therefore be tried as decode; if it is decode-ineligible, plan validation
  fails visibly rather than the request being skipped. Not exercised by tests.
- Design §10.2/§11.2 reject a locked indicator inconsistent with `floor(x)`
  (for example `I=1` with `x` slightly below 1). Implemented as specified; it
  was not triggered by any test.
- `KeyboardInterrupt` during the solve is normalized to an `interruption`
  failure rather than propagated, following design §9.3's table.
- The solver's `nit` is recorded; `crossover_nit` and HiGHS version are not.
- Design decisions D-13 (remaining solver controls), D-17 and D-18 (live
  failure response and empty-plan liveness) remain OPEN and are not affected.
  No design or mathematical document was changed.

## Scope confirmations

- No state-mapping or execution work was performed. No `sarathi` import, no
  `SchedulerOutputs`, no queue, block, sequence, or GPU access. The smoke test
  asserts `sarathi` is not in `sys.modules` after running `main()`.
- No existing framework behavior was repaired.
- Fixture and smoke values (capacities, utilities, charges, decode policy ID
  `fixture_supplied_charge_v1`, numerical policy) are scoped provisional
  inputs and are not presented as permanent project policy.
- No Python identifier, filename, docstring, comment, printed text, or result
  field uses project phase names or numbers.

## Working-tree status before the handoff-document commit

`git status --short --branch --untracked-files=all` at that point:

```text
## main...origin/main [ahead 1]
?? CLAUDE.md
```

`CLAUDE.md` is the pre-existing untracked file, not part of this work.
