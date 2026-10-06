# Exact central/worker block-ID check: original failure and characterization

The original CPU contention case compared complete central and worker block
tables (exact physical block IDs) after each replay. That check **failed**;
it never passed as a gate. It was replaced by the user-approved count,
ownership, and per-manager pool-integrity checks (see the handoff).

## Tested code and environment

- Repository `/home/atjoshi_umass_edu/LPServe`, branch `main`, HEAD
  `c69cddec5f86d721f6ca2d7563c243774205910f`, plus uncommitted
  `tests/test_lp_scheduler.py` exactly as `test_lp_scheduler.exact_id_version.py`
  here (sha256 `2985406cf3c4b3210aac7ade3586ea35a4cb314ec254e41796b8f22ebbb43a72`;
  diff vs HEAD in `test_lp_scheduler.exact_id_version.diff`). No production
  file changed. The new GPU script did not exist for the first two runs and
  is not imported by any of them.
- Host `gpu051`, SLURM job `65298930` (partition `gpu-preempt`),
  modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`,
  `source env/bin/activate`; interpreter
  `/home/atjoshi_umass_edu/LPServe/env/bin/python` (Python 3.10.8);
  `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`. CPU only; no model, no GPU.

## Attempts (in order)

1. Single-test run (output transcribed from the session; not captured to a
   file at the time):

   ```
   timeout 120 python -B -m unittest tests.test_lp_scheduler.LivePipelineTest.test_three_requests_contend_for_two_action_slots_until_completion -v
   ```
   Result: `FAIL` at the `assertEqual(harness.central_tables(),
   harness.worker_tables())` after replay:
   `AssertionError: {0: [7, 8]} != {0: [7, 6]}` (central first). `Ran 1 test`,
   `FAILED (failures=1)`.

2. `python -B trace_cpu.py` (scratch per-step trace, script retained here;
   output transcribed from the session). Decisions (seq_id, chunk):
   1 `[(1,2),(2,2)]`, 2 `[(1,2),(2,2)]`, 3 `[(0,2),(2,0)]`, 4 `[(0,2),(2,0)]`,
   5 `[(1,0),(2,0)]`, 6 `[(1,0),(2,0)]` (seq 2 finishes, frees `[8,6]`),
   7 `[(0,0),(1,0)]`, 8 `[(0,0),(1,0)]` (seq 1 finishes), 9 `[(0,0)]`,
   10 `[(0,0)]` (seq 0 finishes). In this process tables stayed equal at
   every step; only the final free-list order differed
   (central `[0,1,2,3,4,6,5,9,8,7]`, worker `[0,1,2,3,4,6,5,9,7,8]`).

3. `for i in $(seq 1 12); do python -B diverge.py 2>/dev/null | tail -1; done | sort | uniq -c`
   (script and raw output `diverge_12_runs.output` retained here). Each
   process runs the workload and records the first step whose central and
   worker tables differ; it asserts per-request block counts are equal at
   every step (never failed). Result: **5 of 12 diverged** (4 x step 8
   central `{0: [7, 6]}` vs worker `{0: [7, 8]}`; 1 x the reverse), 7 equal.

## Cause (source-confirmed, inherited framework behavior)

`BaseBlockSpaceManager._free_block_table`
(`sarathi/core/block_space_manager/base_block_space_manager.py`) frees
`for block in set(block_table)`. `PhysicalTokenBlock`
(`sarathi/core/datatypes/block.py`) defines no `__hash__`/`__eq__`, so set
order follows object identity (memory address). The central and worker
managers own different block objects, so a freed two-block table can be
pushed onto each free list in a different order, and the next `pop()`
allocation returns different block numbers. Counts are unaffected. Not
repaired (out of scope).
