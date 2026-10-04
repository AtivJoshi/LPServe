# LPServe State Mapping: Ordinary-Idle Contract Alignment Handoff

Evidence record for the D-18 contract change that moves ordinary
future-only or completely empty scheduling out of the mapper and into the
future live `LPScheduler`. It records observed facts only. It does not replace
`docs/lp_scheduler_design.md` (the normative source). The earlier
`docs/handoffs/state_mapping_idle_result_handoff.md` is preserved unchanged as
historical evidence.

## Provenance

| Item | Value |
|---|---|
| Base commit | `b15031e596f341a05019d5fa145aa36aac96c300` (`docs: record state-mapping ordinary-idle result handoff`) |
| Contract-and-implementation commit | `1c9141a29c289dcb01441a8740458980d5fddea9` (`Align D-18 idle contract with inherited scheduler arrival gating`) |
| Branch | `main` |
| Host | Unity node `gpu048`, allocated GPU session. Verification was CPU-only. No model, GPU kernel, or benchmark was run. |
| Python | 3.10.8, `/home/atjoshi_umass_edu/LPServe/env`, after `module load Python/3.10.8-GCCcore-12.2.0` and `source env/bin/activate` in the same shell |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |

Pre-edit checks: branch `main`; HEAD `b15031e`; working tree contained only
the untracked pre-existing `CLAUDE.md`. `b15031e` was an ancestor of HEAD.

Environment note: an earlier attempt on node `gpu059` crashed with
`Illegal instruction` on `import numpy`. That is recorded in the prior
handoff. All verification in this record ran on `gpu048`.

## Exact changed paths

Commit `1c9141a` (contract-and-implementation), confirmed with
`git show --stat --oneline HEAD` and `git diff HEAD^..HEAD --name-only`:

- `docs/lp_scheduler_design.md`
- `lpserve_state_mapping.py`
- `tests/test_lpserve_state_mapping.py`

Commit for this handoff (`docs/handoffs/state_mapping_idle_contract_handoff.md`,
new). Its own hash is recorded in the final response, because a document
cannot contain the hash of the commit that adds it.

The earlier handoff `docs/handoffs/state_mapping_idle_result_handoff.md` was
not modified or deleted. `git add` on the tracked test file printed the
`.gitignore` `test*` hint and exited `1`. The file was staged anyway, which
`git diff --cached --stat` confirmed before the commit.

## Exact D-18 contract change

Before (commit `b15031e`): a coherent empty arrived universe returned an
explicit immutable `IdleStateSnapshot` from the mapper. The mapper decided
ordinary idle.

After (commit `1c9141a`):

- Ordinary future-only or completely empty scheduling is decided by the future
  live `LPScheduler` before mapper invocation. The condition is that `running`
  is empty and `waiting` is empty or its head has `arrival_time > now`. The
  scheduler then returns an ordinary empty `SchedulerOutputs`, with no mapping,
  LP construction, solving, extraction, precommit validation, or execution.
- This matches the audited schedulers. `vllm_scheduler.py` (lines 51-52) and
  `sarathi_scheduler.py` (lines 204-205) both test the head of `waiting` with
  `if seq.arrival_time > now:` followed by `break`. The grep of
  `sarathi/core/scheduler/` also shows the same `arrival_time > now` test in
  `faster_transformer_scheduler.py`, `simple_chunking_scheduler.py`,
  `orca_scheduler.py`, and `slai_scheduler.py`. This change inherits that queue-order and future-request-validity
  assumption and adds no separate validation subsystem.
- The mapper accepts only a nonempty arrived unfinished universe. Direct
  mapping that produces an empty arrived universe returns `MappingFailure`.
- A future request may still be filtered from a mixed nonempty snapshot, as
  before, without further validation.
- Arrived work with no prefill or decode action still produces `no_progress`
  with zero mutation (unchanged D-18 rule).
- No retry, fallback, forced action, or mathematical-layer change was added.

Design-document locations updated for consistency:

- §12.1 (state-mapping boundary): mapper returns `StateSnapshot` or
  `MappingFailure`; ordinary future-only or empty scheduling is handled before
  mapping.
- §13 "Do nothing" table row: ordinary empty output before mapping; arrived
  no-action work fails `no_progress`.
- §13 classification paragraph: replaces the mapper-owned idle classification
  with the scheduler-owned convention, the inherited-assumption note, and the
  mapper's empty-universe `MappingFailure` rule.
- §14.2 D-18 paragraph: idle is decided before mapping; an empty output must
  not suppress `no_progress`.
- §15.4 focused tests: the direct-mapper future-only case is a `MappingFailure`
  test. The live future-only empty-output case is deferred to the future
  `LPScheduler` integration tests.
- Decision register D-18 entry: rewritten to match the above.

No other decision was modified.

## Mapper behavior before and after

| Situation | Before (`b15031e`) | After (`1c9141a`) |
|---|---|---|
| Nonempty arrived unfinished universe | `StateSnapshot` with a validated `LPProblem` | Unchanged |
| Empty arrived universe after future/finished filtering, no resident | `IdleStateSnapshot` | `MappingFailure` (stage `state_mapping`, category `mapping_failure`, reason states the arrived unfinished universe is empty and the accepted `LPProblem` cannot represent an empty set). No `LPProblem` is constructed and `validate_problem` is not called. |
| Empty arrived universe with a resident owner | `MappingFailure` (new in `b15031e`) | `MappingFailure` with the existing empty-universe reason. The separate resident-specific message was removed with the idle path. |
| All other validation (ownership, arrival times, block tables, utilities, numerical policy, single-stage and zero-in-flight checks) | Existing behavior | Unchanged |

Public result contract: `StateSnapshot | MappingFailure`. `IdleStateSnapshot`
was removed from `lpserve_state_mapping.py`, and the idle-specific docstring
text was removed. The `map_scheduler_state` return annotation is
`"StateSnapshot | MappingFailure"`.

## Focused fixture values

Test `test_future_only_universe_is_mapping_failure`:

- block size `4`; total GPU blocks `10`; maximum model length `32`
- scheduler iteration ID `7`; pipeline stages `1`; running batches `0`
- resident limit `4`; snapshot time `100.0`
- `b_max=8`, `c_max=4`, `s_max=3`; memory reserve `1`
- decode policy `conservative_one_block_v1`
- existing numerical policy (`lp_relaxation_mvp_v1`, `1e-7`, `1e-6`, `1e-9`, `1e-9`)
- one waiting request, raw ID `0`, six prompt tokens, arrival `101.0`
  (future relative to `100.0`)
- no running request; utilities `()`

## Nonmutation evidence

- `_fingerprint(holder)` (waiting/running identity and order, iteration ID,
  running-batch count, per-sequence status, token IDs, processed count,
  completion flag, logical blocks, central `block_tables`, and free-block
  order) is captured before the call and asserted equal after it.
- Test-only spies, `mock.patch.object(lsm.lrs, "LPProblem", wraps=...)` and
  `mock.patch.object(lsm.lrs, "validate_problem", wraps=...)`, each
  `assert_not_called()`. No production hook was added.
- `_assert_no_mutable_or_framework_objects(result)` passes. The
  `MappingFailure` contains only `None`/`str` fields.

## Commands and complete observed results

All run on `gpu048` in one shell after `module load` and `source env/bin/activate`.

Pre-commit verification (on the working tree before commit `1c9141a`):

```text
$ python -m py_compile lp_relaxation_scheduler.py lpserve_state_mapping.py tests/test_lp_relaxation_scheduler.py tests/test_lpserve_state_mapping.py
exit: 0

$ python -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
test_fractional_prefill ... ok
test_main_smoke ... ok
test_mixed_case_and_tied_extraction ... ok
test_visible_infeasibility ... ok
test_zero_zero_execution_tie_is_no_action ... ok
Ran 5 tests in 0.104s
OK
exit: 0

$ python -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
test_future_only_universe_is_mapping_failure ... ok
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
exit: 0

$ git status --short --branch --untracked-files=all
## main...origin/main
 M docs/lp_scheduler_design.md
 M lpserve_state_mapping.py
 M tests/test_lpserve_state_mapping.py
?? CLAUDE.md
exit: 0
```

Post-commit verification (after commit `1c9141a`, before this handoff commit):

```text
$ python -m py_compile lp_relaxation_scheduler.py lpserve_state_mapping.py tests/test_lp_relaxation_scheduler.py tests/test_lpserve_state_mapping.py
exit: 0

$ python -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
Ran 5 tests in 0.007s
OK
exit: 0

$ python -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
Ran 5 tests in 0.002s
OK
exit: 0

$ grep -n -i 'phase' lpserve_state_mapping.py tests/test_lpserve_state_mapping.py
exit: 1

$ git diff --check
exit: 0

$ git status --short --branch --untracked-files=all
## main...origin/main [ahead 1]
?? CLAUDE.md
exit: 0
```

## Check status

- **Passed:** `py_compile` over all four files; all five
  `test_lp_relaxation_scheduler.py` tests (unchanged mathematical layer); all
  five `test_lpserve_state_mapping.py` tests, including the replaced
  future-only test and the three prior regressions; the phase-naming grep
  (exit `1`, no matches, the expected result); `git diff --check` (exit `0`).
- **Failed:** none in the reported runs.
- **Skipped:** formatters and linters (none installed; none installed).
- **Unexecuted:** the live `LPScheduler` future-only empty-output case, which
  does not exist yet and belongs to future integration tests. Also unexecuted:
  the `no_progress` gate, solver, extraction, execution, GPU, and benchmark
  checks. These are out of scope for this task.

## Unresolved issues and limitations

- Live scheduler arrival gating is not implemented. It is specified in the
  design and will be tested with future `LPScheduler` integration. It must not
  be treated as verified.
- The arrival gate inherits the audited schedulers' waiting-order and
  future-request-validity assumptions. It does not validate a future request
  until that request reaches an arrived mapping boundary. This is intentional
  per the task, but it is a known limitation.
- The resident-owner case with an empty arrived universe now returns the
  generic empty-universe `MappingFailure`. The earlier resident-specific reason
  text was removed. No test covers that path directly.
- Nothing in this change alters D-01 through D-10, D-13, D-20, D-21 through
  D-24, or any other OPEN decision.

## Scope confirmations

- `LPScheduler` was not implemented. No existing SLAI, Sarathi, VLLM, Orca, or
  other scheduler was modified.
- No `SchedulerOutputs` was constructed.
- The `no_progress` guard was not implemented.
- No solver, integer extraction, execution, or precommit mutation was added.
- No fallback, retry, forced scheduling, rollback, or recovery was added.
- No pipeline-parallel support was added.
- The mathematical layer (`lp_relaxation_scheduler.py`) and its tests were not
  changed.
- No inherited framework behavior was repaired.
- No GPU work or benchmark was run.
- The historical `docs/handoffs/state_mapping_idle_result_handoff.md` was
  preserved.

## Working-tree status

After commit `1c9141a`, before this handoff commit:

```text
## main...origin/main [ahead 1]
?? CLAUDE.md
```

`CLAUDE.md` is the pre-existing untracked file, left unchanged. The final
response reports the status after the handoff commit.
