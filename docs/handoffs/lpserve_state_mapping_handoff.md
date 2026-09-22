# LPServe State-Mapping Layer: Implementation Handoff

Evidence record for the standalone, read-only LPServe state-mapping module. It
records observed facts only; it does not amend `docs/lp_scheduler_design.md`
or `docs/math/main-llm-serving.tex`.

## Provenance

| Item | Value |
|---|---|
| Implementation commit | `a0e35d7c7eb9bf04e709956fc4edfe894a8dcb68` |
| Base commit | `bc95dc4e56e55e645400bbb4f01928db9f764b35` |
| Branch | `main` (local was ahead of `origin/main` by the implementation commit only) |
| Host used | Unity node `gpu048`, CPU-only work; no GPU, CUDA, or model touched |
| Python | 3.10.8 (`/home/atjoshi_umass_edu/LPServe/env`, after `module load Python/3.10.8-GCCcore-12.2.0`) |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |

State before editing: branch `main`, HEAD `bc95dc4`, working tree with only
untracked `CLAUDE.md` (pre-existing, not part of this work, left untouched and
uncommitted).

Confirmed before editing that `docs/lp_scheduler_design.md` at `bc95dc4`
contains the resolved D-08, D-16, D-17, and D-19 rules required by the task
(single-stage `BaseScheduler` ownership; raw non-negative integer `seq_id`
supplying `request_id`/`order_key`; pre-mutation fail-stop; mapping at a
quiescent, zero-batch-in-flight boundary), so implementation proceeded.

Note: the venv interpreter fails with `libpython3.10.so.1.0: cannot open
shared object file` unless the Python module from `docs/unity_setup.md` §3 is
loaded first. This is an environment fact, not a change.

## Implementation paths changed (commit `a0e35d7`)

- `lpserve_state_mapping.py` (new)
- `tests/test_lpserve_state_mapping.py` (new)

`.gitignore` line 205 (`test*`) ignores the tests file, so it was added with
`git add -f`. `.gitignore` was not modified. No other file was changed by this
commit.

## Implemented interface and supported state shape

Public interface in `lpserve_state_mapping.py`:

- `RequestUtility` (frozen): one explicit `(decode_utility,
  prefill_token_utility, preemption_penalty)` triple.
- `RequestStateSnapshot` (frozen): primitive-only observation of one included
  request (raw ID, request ID, order key, ownership, status name, arrival
  time, prompt length/processed/remaining, completion flag, logical block
  count, physical block-number tuple or `None`, eligibility flags, charges,
  utility).
- `StateSnapshot` (frozen): snapshot ID, snapshot time, scheduler iteration
  ID, pipeline-stage/running-batch counts, resident count/limit, free
  physical blocks, memory reserve, decode-memory policy ID, numerical
  policy, the ordered `requests` tuple, and the validated `LPProblem`.
- `MappingFailure` (frozen): optional snapshot ID, fixed `stage="state_mapping"`,
  fixed `category="mapping_failure"`, and a precise `reason` string.
- `map_scheduler_state(scheduler, *, snapshot_time, b_max, c_max, s_max,
  memory_reserve, decode_memory_policy_id, utilities, numerical_policy) ->
  StateSnapshot | MappingFailure`. `utilities` is an iterable of
  `(raw_seq_id, RequestUtility)` pairs (a provisional, visible shape chosen so
  a duplicate row is structurally detectable, unlike a plain dict).

Supported state, matching the task's exact specification and
`docs/lp_scheduler_design.md` §12.4: `scheduler_config.num_pipeline_stages ==
1`; `num_running_batches == 0`; `waiting`/`running` are the complete
authoritative ownership collections; raw `seq_id` values are exact
non-boolean, non-negative integers, unique across both collections (checked
before arrival/completion filtering, so a duplicate that would otherwise be
filtered as future or finished is still rejected); arrived means finite
`arrival_time <= snapshot_time`; future and finished requests are silently
excluded (not a failure); an included `waiting` request must have status
`WAITING`, be unallocated, and have positive, incomplete prompt remainder; an
included `running` request must have status `PAUSED`, be allocated, and (if
prompt-incomplete) have a zero logical/physical block gap, or (if
prompt-complete) a gap of zero or one; block sizes must agree between the
sequence and the block manager; resident count (`len(running)`) must not
exceed `max_num_seqs`.

Eligibility/charge construction follows the task specification exactly:
waiting requests are prefill-eligible only, with
`prefill_fixed_charge=len(seq.logical_token_blocks)`; resident partial
prefills are prefill- and preemption-eligible, with `prefill_fixed_charge=0`;
resident decode-ready requests are decode- and preemption-eligible, with
`decode_charge=1` under the sole accepted decode-memory policy ID
`conservative_one_block_v1`; preemption recovery is
`len(block_manager.get_block_table(seq))` for preemption-eligible requests
and `0` otherwise. The snapshot ID is a SHA-256 hex digest over a JSON
canonicalization of primitive scheduler/request fields (no `hash()`, object
identity, mutable representation, or internally read clock); it is used
directly as `LPProblem.problem_id`. `lp_relaxation_scheduler.validate_problem()`
is run on the constructed problem; `solve_and_extract()` is never called.

## Provisional fixture and decode-policy values used by the focused test

Exactly the values specified for this task: block size `4`; total GPU blocks
`10`; maximum model length `32`; scheduler iteration ID `7`; pipeline stages
`1`; running batches `0`; resident limit `4`; snapshot time `100.0`;
`b_max=8`, `c_max=4`, `s_max=3`, `w=1`; decode-memory policy ID
`conservative_one_block_v1`; numerical policy `policy_id="lp_relaxation_mvp_v1"`,
`feasibility_tol=1e-7`, `integrality_tol=1e-6`, `objective_abs_tol=1e-9`,
`objective_rel_tol=1e-9` (these match `NumericalPolicy()`'s existing
defaults). Three requests: raw ID `0` waiting (6 prompt tokens, 0 processed,
unallocated, utilities `(0.0, 2.0, 0.0)`); raw ID `1` resident paused partial
prefill (6 prompt tokens, 2 processed, allocated, utilities `(0.0, 1.0,
0.25)`); raw ID `2` resident paused decode (4 prompt tokens, 4 processed,
allocated, utilities `(3.0, 0.0, 0.5)`). Residents were constructed through
the legal `WAITING -> RUNNING -> PAUSED` transitions and allocated through the
real `VLLMBlockSpaceManager` before the pre-mapping fingerprint was taken.
These values are scoped MVP/test inputs, not permanent project policy; the
decode-memory policy remains a scoped fixture input to D-09, not its
resolution.

## Commands run and complete observed output

```text
$ python -m py_compile lp_relaxation_scheduler.py lpserve_state_mapping.py \
    tests/test_lp_relaxation_scheduler.py tests/test_lpserve_state_mapping.py
[exit 0]

$ python -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
test_fractional_prefill (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_main_smoke (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_mixed_case_and_tied_extraction (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_visible_infeasibility (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok
test_zero_zero_execution_tie_is_no_action (test_lp_relaxation_scheduler.LPRelaxationSchedulerTest) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.022s

OK
[exit 0]

$ python -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
test_supported_mapping_and_nonmutation (test_lpserve_state_mapping.LPServeStateMappingTest) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
[exit 0]

$ grep -n -i 'phase' lpserve_state_mapping.py tests/test_lpserve_state_mapping.py
[no output, exit 1]

$ git diff --check
[no output, exit 0]

$ git status --short --branch --untracked-files=all
## main...origin/main [ahead 1]
?? CLAUDE.md
[exit 0]
```

## Check status

- **Passed:** `py_compile` over all four listed files; the pre-existing five
  `test_lp_relaxation_scheduler.py` tests (unchanged, no regression); the one
  new `test_lpserve_state_mapping.py` test; the phase-naming grep (exit `1`,
  no matches); `git diff --check` (no whitespace/EOL issues).
- **Failed:** none observed.
- **Skipped:** a formatter or linter; none is installed in the environment,
  and none was installed to run this check.
- **Unexecuted:** execution-layer, live-scheduler-integration, pipeline,
  stale-snapshot, GPU, and performance checks (all out of scope for this
  task). No broader status/ownership/block-layout matrix was added beyond the
  one focused supported-state case, per the task's testing scope.

## Non-mutation evidence

The test captures a primitive fingerprint of the scheduler holder and block
manager before calling `map_scheduler_state`, covering: `waiting`/`running`
list order and Python object identities; scheduler iteration ID and
running-batch count; per-sequence status name, prompt token IDs, output token
IDs, processed-prompt count, completion flag, and full logical-block contents
(block number, size, token count, token IDs); the block manager's central
`block_tables` (sorted by `seq_id`, each a tuple of physical block numbers);
and the exact order of the GPU allocator's free-block list. The identical
fingerprint is asserted after the mapping call and the test passed, so no
observable scheduler, sequence, or block-manager state changed. The mapper
also calls only documented read-only accessors (`is_allocated`,
`get_block_table`, `get_num_free_gpu_blocks`, `.block_size`) and never an
allocation, append, free, preemption, status-transition, or queue-mutation
method.

The test also recursively walks the returned `StateSnapshot` (dataclass
fields, tuples, frozensets) and asserts no `Sequence`, scheduler-holder, or
block-manager instance, and no mutable `list` or `dict`, is reachable from it.

## Unresolved issues, open decisions, and limitations

- No conflict was found between `docs/lp_scheduler_design.md` §§3–6, 8–9, 12,
  14–17 and the current `sarathi` source for the read paths (`BaseScheduler`,
  `Sequence`, `SequenceState`, `SequenceStatus`, `BaseBlockSpaceManager`,
  `VLLMBlockSpaceManager`, `config.py`). D-08/D-16/D-17/D-19 remain resolved
  as recorded; no other decision was resolved by this task.
- `utilities` as an iterable of `(raw_seq_id, RequestUtility)` pairs (rather
  than a plain mapping) is a visible, provisional interface choice made so a
  duplicate row is structurally rejectable rather than silently overwritten.
  This is a scoped implementation-shape choice, not a change to
  `docs/lp_scheduler_design.md` §9's semantic contract.
- Resident count for the `max_num_seqs` check uses `len(scheduler.running)`
  (the raw collection length), matching design §6.4's "authoritative resident
  set before the plan"; this was not separately exercised against a case
  containing a lingering finished resident, since the design states finished
  requests are normally already removed at the quiescent boundary.
- D-09 (decode planning charge), D-10 (memory reserve), D-01–D-04 (utility
  policy/scaling), D-05–D-07 (capacity bindings), and all other OPEN items in
  `docs/lp_scheduler_design.md` §17.1 remain OPEN; this task supplied only the
  scoped, visible fixture values listed above for its own test, not a
  project-wide resolution.
- Only the single focused supported-state case specified by the task was
  added. Exhaustive status/ownership/duplicate-conflict/stale-snapshot/
  alternative-decode-policy variants are deferred, per
  `docs/lp_scheduler_design.md` §15.3.

## Scope confirmations

- No LP solve, integer extraction, live `LPScheduler`, scheduler
  registration, `SchedulerOutputs` construction, execution-layer logic, fresh
  precommit validation, pipeline-parallel support, GPU test, or inherited
  Sarathi/SLAI behavior repair was performed. `lrs.solve_and_extract()` is
  never called by the new module.
- No scheduler, sequence-manager, or block-manager mutation occurred, per the
  non-mutation evidence above.
- No OPEN decision in `docs/lp_scheduler_design.md` §17.1 was resolved
  permanently; no normative design or mathematical document was changed.
- No Python identifier, filename, docstring, comment, printed text, or result
  field in either new file uses project phase names or numbers (verified by
  the grep check above).

## Working-tree status after the implementation commit

`git status --short --branch --untracked-files=all` at that point:

```text
## main...origin/main [ahead 1]
?? CLAUDE.md
```

`CLAUDE.md` is the pre-existing untracked file, not part of this work, and was
left untouched throughout.
