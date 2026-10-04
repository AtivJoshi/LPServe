# LPServe State Mapping: Ordinary-Idle Result Handoff

Evidence record for the D-18 idle-result correction to the read-only state
mapper. It records observed facts only; it does not amend
`docs/lp_scheduler_design.md`, the mathematical layer, or any other normative
document.

## Provenance

| Item | Value |
|---|---|
| Base commit (HEAD before the implementation commit) | `7ca0ed350aa28ea5445ada848e8acc933b55b4be` |
| D-18 documentation commit | `7ca0ed350aa28ea5445ada848e8acc933b55b4be` (`refine LPServe scheduler design: clarify failure responses and idle state handling`; modifies only `docs/lp_scheduler_design.md`) |
| Implementation commit | `d49b305196c3425d1db78ff3b77e39ca0947c2df` |
| Branch | `main` |
| Host used | Unity node `gpu048`, allocated GPU session (`CUDA_VISIBLE_DEVICES=0`); verification was CPU-only and no model or GPU kernel was run |
| Python | 3.10.8 (`/home/atjoshi_umass_edu/LPServe/env`, activated after `module load Python/3.10.8-GCCcore-12.2.0`) |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |

### D-18 documentation check (performed before editing)

`docs/lp_scheduler_design.md` at `7ca0ed3` records D-18 as resolved for the
single-stage MVP: a coherent future-only or empty state returns an immutable
ordinary-idle result without constructing an `LPProblem`, and arrived
unfinished work with no prefill or decode action is handled later as
`no_progress`. The text matched the task's prerequisite. The working tree
before editing contained only the pre-existing untracked `CLAUDE.md`.

### Environment note

The first verification attempts ran on login/compute node `gpu059`. There, the
Python 3.10.8 module crashed with `Illegal instruction` (exit 132) on
`import numpy`, even with `env/bin/activate` sourced and `python --version`
working. Two standard NumPy/OpenBLAS CPU-dispatch workarounds
(`OPENBLAS_CORETYPE=Haswell`, `NPY_DISABLE_CPU_FEATURES=...`) did not help. No
verification output from `gpu059` is claimed. All verification reported below
ran on `gpu048`, where the same module and venv imported NumPy 2.2.6 and SciPy
1.15.3 successfully.

## Exact changed paths

- Implementation commit `d49b305`:
  - `lpserve_state_mapping.py` (modified)
  - `tests/test_lpserve_state_mapping.py` (modified)
- Handoff commit: `docs/handoffs/state_mapping_idle_result_handoff.md` (new).
  This commit's own hash is recorded in the final response, because a document
  cannot contain the hash of the commit that adds it.

`tests/` matches the `test*` pattern in `.gitignore`. The already-tracked test
file was staged by `git add` without `-f`; the command printed the ignored-path
hint and exited nonzero, but the modification was staged (confirmed with
`git diff --cached --stat` before committing). The same hint appeared in the
earlier correction.

## Implemented result type and fields

```python
@dataclass(frozen=True)
class IdleStateSnapshot:
    snapshot_id: str
    snapshot_time: float
    scheduler_iteration_id: int
```

It contains only these three primitive fields. It holds no `LPProblem`,
`Sequence`, scheduler, block manager, list, or dict.

The public contract of `map_scheduler_state` is now annotated as:

```text
-> "StateSnapshot | IdleStateSnapshot | MappingFailure"
```

The snapshot ID for an idle result is produced by the same
`_compute_snapshot_id` used for nonempty snapshots, with an empty request
tuple. This is a design choice of this correction: one identity function is
retained, and the ID is deterministic for the observed state.

## Idle classification and rejection boundaries

The idle decision happens only after all existing read-only validation has
completed, at the point where the arrived unfinished request list is empty.
That validation covers the scalar and policy arguments, the single-stage and
zero-in-flight checks, resident-count limit, the scheduler-owned ID shape and
uniqueness, arrival-time validity for every owned request, block-table
ownership in both directions (before future/finished filtering), utility
rows (missing/duplicate/malformed/extra), and per-request construction.

Returns `IdleStateSnapshot` when:

- no arrived, unfinished scheduler-owned request exists, and
- `len(scheduler.running) == 0` (no resident owner).

Both coherent states qualify: no owned unfinished request at all, or one or
more well-formed future waiting requests with no resident owner. Finished
requests are excluded as before and do not block idle.

Returns `MappingFailure` (unchanged or new) for:

- malformed or non-finite arrival time, including on any owned request;
- duplicate or contradictory ownership;
- missing, waiting-owned, orphan, or malformed block-table entries;
- invalid IDs, utility rows, configuration, or supported-state fields;
- a resident owner when the arrived unfinished universe is empty. This is new
  in this correction. It is rejected as incoherent, not treated as idle,
  because residency implies prior arrival. This case is reachable: a
  `running` entry whose `arrival_time` is in the future is excluded from the
  request list by the filter, so the list can be empty while `running` is not.

The nonempty-universe path is unchanged. For an idle result, `LPProblem` is not
instantiated, `validate_problem` is not called, no solver or extraction runs,
and no mutation occurs. The idle `return` precedes the `LPProblem(...)` line.

## Focused fixture values

Test: `test_idle_result_for_future_only_universe`.

- block size `4`; total GPU blocks `10`; maximum model length `32`
- scheduler iteration ID `7`; pipeline stages `1`; running batches `0`
- resident limit `4`; snapshot time `100.0`
- `b_max=8`, `c_max=4`, `s_max=3`, memory reserve `1`
- decode policy `conservative_one_block_v1`
- existing numerical policy (`lp_relaxation_mvp_v1`, `1e-7`, `1e-6`, `1e-9`, `1e-9`)
- one waiting request, raw ID `0`, six prompt tokens, arrival `101.0` (future)
- no running request; utilities `()`

The test also calls the mapper a second time on the same unchanged state and
asserts the same `snapshot_id`.

## Nonmutation evidence

- The test computes `_fingerprint(holder)` before mapping and asserts equality
  after the first call and after the repeated call. The fingerprint covers
  waiting/running identities and order, scheduler iteration ID, running-batch
  count, per-sequence status, token IDs, processed prompt count, completion
  flag, logical blocks, the central `block_tables`, and the free-block order.
- `mock.patch.object(lsm.lrs, "LPProblem", wraps=...)` and
  `mock.patch.object(lsm.lrs, "validate_problem", wraps=...)` are test-only
  spies. Both `assert_not_called()` passed. No production fault-injection hook
  was added.
- `_assert_no_mutable_or_framework_objects(result)` passed.
- `dataclasses.FrozenInstanceError` is asserted on assignment to
  `result.snapshot_id`.

## Commands run and observed output

All commands ran on `gpu048` after `module load` and `source env/bin/activate`
in the same shell.

```text
$ python -m py_compile lp_relaxation_scheduler.py lpserve_state_mapping.py \
    tests/test_lp_relaxation_scheduler.py tests/test_lpserve_state_mapping.py
exit: 0

$ python -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
test_fractional_prefill ... ok
test_main_smoke ... ok
test_mixed_case_and_tied_extraction ... ok
test_visible_infeasibility ... ok
test_zero_zero_execution_tie_is_no_action ... ok
Ran 5 tests in 0.007s
OK
exit: 0

$ python -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
test_idle_result_for_future_only_universe ... ok
test_malformed_arrival_time_is_rejected ... ok
test_none_utilities_returns_mapping_failure ... ok
test_orphan_block_table_is_rejected ... ok
test_supported_mapping_and_nonmutation ... ok
Ran 5 tests in 0.002s
OK
exit: 0

$ grep -n -i 'phase' lpserve_state_mapping.py tests/test_lpserve_state_mapping.py
(no output)
exit: 1

$ git diff --check
(no output)
exit: 0

$ git status --short --branch --untracked-files=all
## main...origin/main [ahead 1]
?? CLAUDE.md
exit: 0
```

Earlier, on `gpu059`, `python -m py_compile` and any `import numpy` exited
with 132 (`Illegal instruction`). Those attempts are recorded only as the
environment note above. They are not verification results.

## Checks: passed, failed, skipped, unexecuted

- **Passed:** `py_compile` on all four files; all five existing
  `test_lp_relaxation_scheduler.py` tests; all five `test_lpserve_state_mapping.py`
  tests, including the new idle test and the three prior regressions; the
  naming grep (exit `1`, no matches); `git diff --check` (exit `0`).
- **Failed:** none in the final runs. The `gpu059` `Illegal instruction`
  crashes are an environment failure on that node and are not counted as test
  failures.
- **Skipped:** formatters and linters (none installed; none installed for this
  correction).
- **Unexecuted:** the resident-owner-with-empty-universe rejection has no
  dedicated regression test. The task asked for a single focused idle test and
  no broader matrix. The branch is covered by code review only. Also unexecuted:
  GPU, solver, extraction, scheduler integration, and performance checks, all
  out of scope.

## Unresolved issues and limitations

- The resident-owner rejection path (`running` non-empty while the arrived
  universe is empty) is implemented but not separately exercised by a test.
- The idle snapshot ID includes the capacities and policy fields through the
  shared canonicalization, even though `IdleStateSnapshot` does not store them.
  The ID is deterministic but is not a minimal idle-only identity.
- `docs/lp_scheduler_design.md` and the D-18 text were read, not edited.
  D-09, D-10, D-01 through D-07, and the other OPEN items remain OPEN.
- `utilities` remains an iterable of `(raw_seq_id, RequestUtility)` pairs. The
  idle path requires an empty utilities iterable for an empty included set,
  and it rejects extra rows. This is consistent with the existing validation.

## Scope confirmations

- No solver was invoked. No extraction was performed.
- No live `LPScheduler`, scheduler registration, or `SchedulerOutputs`
  construction was added.
- No execution, precommit `no_progress` gate, or Phase F-style logic was added.
- No GPU work, model load, or benchmark was run.
- No pipeline-parallel support was added. No Sarathi/SLAI repair was made.
- No mathematical-layer file (`lp_relaxation_scheduler.py`) or its tests were
  changed. No normative document was edited.
- No Python identifier, filename, comment, docstring, or output in either
  changed Python file uses project phase names or numbers (grep verified).

## Working-tree status

After the implementation commit `d49b305`:

```text
## main...origin/main [ahead 1]
?? CLAUDE.md
```

After the handoff commit, the expected state is `ahead 2` with only
`?? CLAUDE.md` untracked. The final response records the observed value.
