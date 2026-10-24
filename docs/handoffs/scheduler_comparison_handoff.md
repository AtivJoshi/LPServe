# Scheduler comparison — bounded GPU handoff

**Scope.** First bounded dummy-weight GPU comparison of the adapted vLLM
reference (`AdaptedVLLMScheduler`), the uniform LP (1/1/1), and the
rank-based LP (`rank_based_utility`, D = P = G = 1) on one two-request
workload with resident limits 2 and 1. This is a scheduling observation:
coefficients → LP proposal → integer plan → execution → progress. It draws no
TTFT, throughput, latency, SLO, quality, fairness, or recovery conclusion.
Q11 remains OPEN. No production file, contract, coefficient, or design
document changed.

**Evidence:** [`validation_output/scheduler_comparison/20261018T063322Z/`](../../validation_output/scheduler_comparison/20261018T063322Z/)
(`cpu_checks/` gate; one directory per run with `command.txt`, `console.log`,
`exit_status.txt`, `summary.json`, `decision_trace.json`, `cleanup.json`;
`comparison.txt` from `scripts/summarize.py`; `run_gpu.sh`,
`shell_environment.txt`, `git_state_gpu.txt`, `sha256sums*.txt`).
Tested at committed HEAD `0aa6cb741f81406032b15c3e1753e13678b4c0fc` with no
tracked changes (`git diff HEAD` hash = empty-input hash) plus the two new
untracked files below; the driver hash `c627bd10…` is identical in the CPU
gate, every run's `summary.json`, and the file at report time. Host `gpu049`,
Slurm job `65821002`, NVIDIA A16
(`GPU-550e163c-e6cb-b9a9-8f68-342d097c2226`, driver 595.91.07), modules
`uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`,
`env/bin/python` 3.10.8, torch 2.3.0+cu121, ray 2.58.0, numpy 2.2.6, scipy
1.15.3. Model snapshot
`models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1…`, dummy
weights, float16, flash attention, seed 42, offline.

## Files

| File | Change |
|---|---|
| `scripts/compare_scheduler_gpu.py` | New driver: one (policy, resident limit) per process; policy-specific contract checks; `drained` / `bounded_unfinished` / `failure` outcome separate from validation |
| `tests/test_compare_scheduler_gpu.py` | New, 15 CPU tests; needs `git add -f` (`.gitignore` `test*`) |
| `docs/handoffs/scheduler_comparison_handoff.md` | This handoff |

Reused unchanged: `EngineRuntime`, `build_engine_class`, the validation
worker, `instrument_block_manager`, `central_state`, `check_quiescent`,
`check_capacity` (`check_adapted_vllm_gpu.py`); `CacheCapacitySelection`,
`check_declared_order`, `select_prompt_ids` (`check_lp_scheduler_preemption_gpu.py`);
`check`, `to_jsonable`, record helpers (`check_lp_scheduler_gpu.py`). The
reference's `run_case` was not used (it prescribes one schedule). No helper
file changed. The driver registers `check_adapted_vllm_gpu` for by-value
pickling so the Ray worker does not need to import that script (Ray would
otherwise rely on its propagated script directory).

Configuration is native: `EngineArgs(scheduler_type="adapted_vllm",
max_num_batched_tokens=8)` or `EngineArgs(scheduler_type="lp", lp_b_max=8,
lp_c_max=8, lp_s_max=1, lp_*_utility=1.0, lp_numerical_policy=…)`. This is
the first real engine initialization through the LP `EngineArgs` path. The
rank-based run replaces `_coefficient_policy` on its scheduler instance; its
configured 1/1/1 triple is then unused. Numerical policy
`lp_relaxation_mvp_v1`: feasibility 1e-7, integrality 1e-6, objective abs and
rel 1e-9. Prompts A = tokens 1000–1004, B = 2000–2007 (equal to the accepted
mixed-case selection, rechecked per run), two outputs each, budget 8, block
size 16, 8 blocks, `max_model_len` 32, native watermark 0, bound 8 nonempty
decisions.

## Critical review (analytical unless stated)

1. **This fixture has no memory pressure.** Each request needs one 16-token
   block for its whole life (5+2 and 8+2 tokens), and 8 are available, so
   every decode demand is 0 and no preemption can be allocation-driven. All
   observed preemptions are therefore voluntary (utility-driven); the
   traces confirm free blocks ≥ 6 and demand 0 at each one. The comparison
   cannot show memory-driven behavior of either policy. The earlier CPU
   comparison (block size 4) is a different resource state.
2. **`s_max = 1` determines limit-2 timing.** The LP needs at least six
   decisions (two prefills, four decodes, one action each); the reference
   co-schedules and needs four. At limit 1 the cap never binds: one resident
   allows one action, and preempt-plus-admit is one action. Limit-1
   differences therefore come from the objective and coefficients.
3. **Both LP policies cycle at limit 1 on this workload.** The objective has
   no discarded-work cost (κ is OPEN under Q11). Uniform: replacing the
   resident is worth β·prompt − γ = 5 − 1 or 8 − 1 against a decode worth 1.
   Rank-based: 0.5·8 − 1 = 3 or 0.5·5 − 1 = 1.5 against 1. Because the
   inherited final-prefill sample is discarded, a resident has no output at
   that moment, so the LP inputs repeat with period 2. With a deterministic
   solver the same choices would repeat indefinitely. This is an analytical
   consequence; only 8 decisions were observed, which is bounded
   non-completion, not proof of starvation.
4. **The rank-based family at unit scales is not FCFS-like here.** Prefill
   reward is per token while decode reward is per action, so an 8-token
   prompt at rank 1 (4.0) outweighs the rank-0 resident's decode (1.0). This
   is a property of the candidate family (D-01–D-04 OPEN), not an
   implementation bug, and it was not tuned.
5. **Clocks (inherited).** Arrivals use `time.perf_counter()` and both
   schedulers use `time.monotonic()`; on this host both are
   `clock_gettime(CLOCK_MONOTONIC)` (checked with `time.get_clock_info`). A
   platform where they differ could make arrived requests look future-dated.

No defect was found in the LP or reference contracts used here. Review scope:
the driver, the reused helpers, `lp_scheduler.py`, `adapted_vllm_scheduler.py`,
the mapper's rank/coefficient path, LP construction, the executor order, and
worker replay. Kernels, cache contents, and multi-worker paths were not
reviewed.

## Results (executed)

All six processes printed `RESULT: PASS`, exited 0 (`timeout`/Python and
`tee` statuses recorded under `pipefail`), and shut down their own Ray
runtime (`ray_initialized_after_shutdown: false`). One attempt each; no
failure, timeout, or retry. Profiling ran once per process and reported
15616 blocks each time; exactly 8 were initialized through native
`_init_cache` (central manager, worker allocator, and cache tensors). The
only console warning was `Casting torch.bfloat16 to torch.float16`.
Arrivals preceded the first decision in every run (`comparison.txt`).

Per request (A/B, decision indices; `comparison.txt`):

| Run | Outcome | Dec. | First prefill | First decode | Done | Token events | Preemptions | Recomputed prefill tokens | Omitted decisions |
|---|---|---|---|---|---|---|---|---|---|
| reference, limit 2 | drained | 4 | 0/0 | 1/2 | 2/3 | 2/2 | 0/0 | 0/0 | 0/0 |
| LP uniform, limit 2 | drained | 6 | 1/0 | 4/2 | 5/3 | 2/2 | 0/0 | 0/0 | 3/1 |
| LP rank-based, limit 2 | drained | 6 | 0/1 | 2/4 | 3/5 | 2/2 | 0/0 | 0/0 | 1/3 |
| reference, limit 1 | drained | 6 | 0/3 | 1/4 | 2/5 | 2/2 | 0/0 | 0/0 | 0/3 |
| LP uniform, limit 1 | bounded_unfinished | 8 | 1/0 | –/– | –/– | 0/0 | 3/4 | 15/24 | 1/0 |
| LP rank-based, limit 1 | bounded_unfinished | 8 | 0/1 | –/– | –/– | 0/0 | 4/3 | 15/24 | 0/1 |

"Recomputed" counts prefill tokens executed beyond the original prompt;
the discarded work was prompt progress only (no generated token was ever
discarded). In the unfinished runs at the bound, the final state was
consistent (queues, per-manager conservation, equal central/worker tables
and free counts); the final record keeps the resident (B or A), the waiting
request, and the accounting. Nothing was forced, freed, or cleared.

The six-run decision sequences equal the earlier block-size-4 CPU
comparison row for row. Every relaxed solution was integral
(`fractional_request_count` 0, HiGHS status 0), so extraction never changed
a proposal. Central and worker physical block IDs were equal after every
decision in all runs (each request held one block); the earlier CPU
rank-based limit-1 difference came from multi-block tables and the inherited
set-order free, which this fixture cannot reach.

**First material differences** (from the recorded before-states):

- **Limit 2, decision 0** (identical state: nothing resident, waiting A,B,
  8 free). Reference: A:5 then B:3 (FCFS, B chunked to the remaining
  budget). LP uniform: B:8 (reward 8 > 5). LP rank-based: A:5 (5 > 0.5·8).
  The cap `s_max = 1` excludes the reference's two-action batch; the
  coefficients pick which single action. Execution matched each plan.
- **Limit 2, decision 1.** Reference and rank-based LP are in different
  states (reference has B partially prefilled). Rank-based: A resident with
  prompt done, B waiting; it admits B:8 (4.0) over A's decode (1.0), then
  decodes A at 2–3 (1.0 > 0.5) and B at 4–5. Uniform at 2–3 decoded B over
  A; that was a tie (1 = 1), resolved by the solver vertex and not
  prescribed.
- **Limit 1, decision 1** (reference and rank-based LP in the same state:
  A resident with prompt done and demand 0, B waiting, 7 free). Reference
  decodes A (no allocation failure, so no preemption; stop
  `resident_limit`). Rank-based LP preempts A and admits B:8 (objective 3
  vs 1). This is the LP proposal itself (voluntary replacement); extraction
  and execution matched it. Uniform differed already at decision 0 (B:8),
  then alternates replacements with objectives 4 and 7.

The reference's B was omitted for three decisions at limit 1 because of the
resident limit, with A served; that is the policy, not starvation.

**Inherited behavior seen (not repaired):** each request's first decode
sample equals its discarded final-prefill sample (for example A: 17573),
consistent with the earlier GPU run. The output counter reset is not
exercised: no request was preempted after generating a token.

## Coverage

| Obligation | Check | Result |
|---|---|---|
| Policy plan/proposal validity and eligibility | `check_lp_decision` (mapped inputs = before-state, coefficients and ranks per policy, plan arithmetic, eligibility, limits, memory, objective) / `check_reference_decision` (proposal inputs, action eligibility, budget, resident limit) | pass, every decision, 6 runs |
| One mapping, one solve, one execution per LP decision; native results unchanged | `DecisionObserver`; `test_wrappers_return_the_native_result_unchanged` | pass |
| Central calls and emitted order per policy contract | `check_declared_order` (LP) / prompt-first layout (reference); `check_execution` | pass |
| Worker replay matches emitted work; conservation; central/worker state, table lengths, free counts | `check_execution`, `check_quiescent` | pass; physical IDs recorded, equal here |
| Sample/request association; finished removal and release | `check_execution` | pass (greedy path) |
| Drained: full release, one ordinary idle call | `run_comparison` | pass, 4 runs |
| Bounded unfinished: visible, consistent final state, no forcing | `run_comparison` | pass, 2 runs |
| **Not established** | — | cache contents, attention correctness, sampler generality, quality, physical-ID equality in general, memory-pressure behavior, recovery, starvation, fairness, performance |

## Commands and results

CPU gate: `bash validation_output/scheduler_comparison/20261018T063322Z/cpu_checks/run_checks.sh <dir>/cpu_checks`
(`CUDA_VISIBLE_DEVICES=""`):

| Step | Command | Result |
|---|---|---|
| 01 | `python -B -m py_compile` on the two new files | exit 0 |
| 02 | `python -B -W error scripts/compare_scheduler_gpu.py --help` | exit 0 |
| 03 | `python -B -m unittest tests.test_compare_scheduler_gpu -v` | 15 passed |
| 04 | trailing-whitespace grep on the new files | exit 0 |
| 05 | `git diff --check` | exit 0 |

The 15 tests cover import/`--help` without torch, Ray, engine, worker, or
LP modules; all six native configurations (block size, capacities,
coefficients, caps); native central/worker CPU replay of all six runs
against the earlier CPU rows; voluntary-only limit-1 preemptions; the
decision bound; an injected solver failure (partial trace, no mutation);
two contract violations detected (reference order, rank-based
coefficients); capacity selection (8 initialized; 7 fails before init);
trace serialization; failure recording without a GPU and no overwrite; and
the worker serialization boundary (by-reference pickle fails with
`ModuleNotFoundError` in an interpreter without `scripts/`; the by-value
pickle loads and its reused helpers run there, Ray not started).

Not rerun (unchanged, not imported differently): the LP mathematical,
mapper, executor, live-scheduler, reference, and earlier driver suites.

GPU: `bash validation_output/scheduler_comparison/20261018T063322Z/run_gpu.sh <dir>`
ran, sequentially, for limit in 2 1 and policy in adapted_vllm, lp_uniform,
lp_rank_based:

```text
PYTHONPATH="$PWD" timeout 600s python -B scripts/compare_scheduler_gpu.py \
  --policy <policy> --resident-limit <limit> --model-path <model dir> \
  --output-dir validation_output/scheduler_comparison/20261018T063322Z/<policy>_resident<limit>
```

All six: exit 0, `RESULT: PASS`, outcomes as in the table above.

## Addendum 2026-10-18: memory-pressure case

**Scope.** One new case, `--case memory_pressure`, in the same driver: the
reference check's `preemption` workload. A, B, C, D have 16 prompt tokens
each (IDs 1000–1015, 2000–2015, 3000–3015, 4000–4015, rechecked against the
tokenizer in every run) and 3 outputs each. Pool 4 blocks of 16,
`max_model_len` 64, budget 64, resident limit 4, bound 16 nonempty decisions.
LP caps are `b_max` 64, `c_max` 64, `s_max` 4. They add no restriction here,
because every action's request is resident afterwards, so actions ≤ 4. The
coefficients, numerical policy, model settings, and seed are unchanged. The
original case, its constants, and its command (`--case` defaults to
`two_request`) are unchanged. Q11 remains OPEN. No production file, contract,
or design document changed.

**Evidence:** [`validation_output/scheduler_comparison/20261018T195933Z/`](../../validation_output/scheduler_comparison/20261018T195933Z/)
(`cpu_checks/`; `memory_pressure_<policy>/` with `command.txt`, `console.log`,
`exit_status.txt`, `summary.json`, `decision_trace.json`, `cleanup.json`;
`comparison.txt` from `scripts/summarize.py`; `run_gpu.sh`,
`shell_environment.txt`, `git_state_gpu.txt`, `sha256sums*.txt`). Tested at
HEAD `d988ed1643689bfff0a4de05aa62a774bfa63190` plus uncommitted changes to
the driver and test only (`cpu_checks/working_tree.diff`; `git diff HEAD`
sha256 `3e0a2224…`, identical at the CPU gate and GPU start). Driver sha256
`d1d16aef…` and test sha256 `d792f336…` are the same in the CPU gate, the GPU
start, and every `summary.json`. Host `gpu055`, Slurm job `65832130`
(partition `gpu-preempt`), NVIDIA A16
`GPU-0f8ce0e5-4e07-31f1-d565-6353513166db`, driver 580.178.04. Modules
`uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`; `env/bin/python`
3.10.8, torch 2.3.0+cu121, ray 2.58.0, numpy 2.2.6, scipy 1.15.3. Same
TinyLlama snapshot `fe8a4ea1…`, offline.

### Driver changes (validation only)

`case_inputs` selects the case. The checks now cover several actions and
victims, decode demands 0 and 1, and preemption after output:

- one plan decision per mapped request;
- actual memory and the decode prerequisite (`H ≥ 1` if any decode);
- the relaxed objective, recomputed from the relaxed values, bounds the
  integer objective;
- the running order is retained, victims are front-inserted in preemption
  order, and admissions are appended in emitted order;
- each decode's actual sampled token is checked against the sequence state
  and the request output, including at completion;
- a victim's prompt grows by its outputs and keeps its logical blocks.

The accounting is driver-only. It counts cumulative generated-token events
and the native counter (reset by recomputation), and reports overgeneration.
It counts discarded prompt progress, generated tokens moved into the prompt,
and prefill actions. It splits executed prefill tokens into first-time,
replayed, and expanded-context positions. The field
`discarded_generated_tokens` was renamed `generated_tokens_moved_to_context`,
because those tokens are not discarded.

Each decision also records `victim_need` (which constraint, if any, needed
its victims for the selected batch, and whether a single action with no
victim existed under that policy's contract), `lp_comparison` (relaxed vs.
integer values and objectives), and whether central and worker physical IDs
are equal. Two test-fixture defects were fixed: the CPU runtime sampled one
constant token, which could not detect a sample/request mix-up, and the
helpers could not take a case. Helper scripts are unchanged.

### Critical review

Analytical unless marked observed.

**Clarified/corrected 2026-10-19 after source and recorded-input review at
`2c61d3e`.** These interpretation updates leave the tested revision and retained
run evidence unchanged. Best integer utilities below are analytical review
results, not outputs of an executed exact-ILP solver.

1. **The LP decode prerequisite follows an inherited scheduler convention.**
   The vLLM-style, Sarathi, SLAI, and SimpleChunking schedulers enforce
   `can_append_slot()` (free > 0); the LP preserves that convention. It is not
   a physical allocator requirement: `append_slot()` allocates only for an
   actual block gap, and worker replay ignores the query's return value.
   The adapted vLLM reference uses actual additional demand and permits
   zero-demand decodes with no free blocks (observed on GPU). At decision 1,
   all four residents have demand 0 and no free blocks: the LP convention
   forces a victim, while the reference decodes all four. See
   [design §8.3](../lp_scheduler_design.md#83-decode-demand) for source links.
   Both implementations remain unchanged. Removing the convention from LP
   or adding it to the reference is deferred; either requires an explicit
   later decision. Cache-content correctness remains unverified.
2. **Separate relaxation gaps from extraction loss.** The prerequisite row
   `Σy ≤ S_max·H` can be funded fractionally; a lower returned utility than
   the LP optimum alone does not establish extraction loss.

   | Policy / decision (zero-based) | LP optimum | Best integer utility | Returned utility | Attribution |
   |---|---|---|---|---|
   | Uniform / 1 | 2.4 | 2 | 2 | Relaxation gap |
   | Rank-based / 3 | ≈4.1944 | ≈3.4167 | ≈3.4167 | Relaxation gap |
   | Uniform / 4 | 16 | 16 | 15 | Extraction loss |

   For uniform decision 4, the constructor evicts A and C (one block each),
   then prefills B's 17 tokens: utility 17 − 2 = 15. Evicting D alone releases
   two blocks for the same prefill: utility 17 − 1 = 16. This is an accepted
   constructor limitation. The earlier attribution of rank-based decisions
   3, 4, 6, … to extraction loss was incorrect: their returned plans attain
   the best integer utility in the recorded states; their gaps are due to
   the relaxation.
3. **Both LP policies cycle with period 2 from decision 4/5 (observed to the
   bound).** A readmitted request finishes its prompt with no output (the
   inherited final-prefill sample is discarded), decode demand 0, and 0 free
   blocks. The objective then prefers preempting both residents and
   readmitting the two waiting requests. Uniform: 17 + 17 − 2 = 32 against
   decoding both = 2. Rank-based: 17/3 + 16/4 − 1.5 = 8.17 against 1.5. The
   replacement wins even without the prerequisite, so the cycle comes from
   the objective and coefficients: per-token prefill reward and no
   discarded-work cost (κ is OPEN under Q11). It is bounded noncompletion,
   not proof of starvation; nothing was tuned.
4. **A preemption at 0 free blocks is not always allocation pressure.**
   Reference decision 2 is pure decode demand. LP decision 1 is the
   prerequisite alone. From LP decision 4 on, the blocks are needed by the
   policy's chosen admissions (`admission_charges`, replacement). At
   rank-based decisions 2, 5, 7, … a preemption-free single action existed.

No validation-driver defect remained after the CPU gate. Review scope: the
driver and reused helpers, both schedulers, the mapper's rank and
coefficient path, LP construction, extraction and plan validation, the
executor's order and append gate, native preempt/recompute, the worker
sequence manager, and the block manager. Kernels, cache contents, and
multi-worker paths were not reviewed.

### Results (executed)

The three fresh processes ran sequentially, once each. All printed
`RESULT: PASS`, with `timeout`/Python and `tee` exit statuses 0, and shut
down their own Ray (`ray_initialized_after_shutdown: false`). Profiling ran
once (15603 blocks); exactly 4 blocks went through native initialization
(central, worker allocator, cache tensor dimension 4). The only warning was
the bfloat16→float16 cast. GPU decision sequences equal the CPU-gate traces
decision for decision.

| Policy | Outcome | Dec. | Finished | Token events A/B/C/D (native counter) | Overgen. | Preemptions (after output) | Prefill exec. = first + replayed + expanded | Resources |
|---|---|---|---|---|---|---|---|---|
| reference | drained | 8 | all 4 | 3/3/4/4 (3 each) | C, D +1 | 2 (2) | 98 = 64 + 32 + 2 | full release, idle ordinary |
| LP uniform | bounded_unfinished | 16 | none | 1/1/1/2 (0 each) | – | 26 (4) | 476 = 64 + 407 + 5 | consistent; residual A:2, C:2 blocks, 0 free |
| LP rank-based | bounded_unfinished | 16 | none | 2/1/1/0 (0 each) | – | 28 (3) | 505 = 64 + 437 + 4 | consistent; residual A:2, C:2 blocks, 0 free |

In every decision of all three runs, the central and worker states, table
lengths, and free counts agreed. Physical block IDs differed after 3
(reference), 11 (uniform), and 12 (rank-based) decisions; they are recorded,
not required. Request D (rank-based) was preempted 8 times and generated no
token within the bound.

**First material divergence: decision 1, identical state.** All four are
resident with complete prompts, demand 0, and 0 free blocks. The reference
decodes all four with no allocation (its contract). Both LPs preempt one
request and decode three, with H = 1. The preemption comes from the
formulation (decode prerequisite, finding 1). The victim comes from the
coefficients for rank-based: tail D is the unique optimum, 1 + 1/2 + 1/3 −
1/4. For uniform it comes from the solver vertex: every single victim ties
at 2; the vertex put z = 0.8 on head A, and extraction kept it. Execution
matched each plan. Later indices are not corresponding states: the
reference's genuine one-block tail preemption (victims 3, 2) and its
17-token readmissions never occur in the LP runs.

**Inherited behavior seen (not repaired):** the first decode sample equals
the discarded final-prefill sample (reference: 10935, 3591, 10123, 14666).
Recomputation moves outputs into the prompt and resets the counter; after
restart, completion took 3 more events (C, D: 4 in total).

### Coverage

| Obligation | Check | Result |
|---|---|---|
| Inputs, coefficients/ranks, mapped physical state | `check_lp_decision` / `check_reference_decision` | pass, every decision |
| Integer plan: arithmetic, eligibility, one decision per request, caps, memory, prerequisite, objective, relaxed bound | `check_lp_decision` | pass |
| Emitted order and central calls per policy | `check_declared_order` / prompt-first layout; `check_execution` | pass |
| Worker replay, conservation, central/worker agreement | `check_execution`, `check_quiescent` | pass; IDs recorded |
| Running order, victim reinsertion, readmission, completion removal | `check_execution` | pass |
| Sample IDs and token values vs state updates and outputs | `check_execution`; CPU swap test | pass; mix-up detected |
| Recomputation (prompt growth, counter reset, logical blocks) | `check_execution` | pass (reference 2, uniform 26, rank-based 28 victims) |
| Drained / bounded-unfinished / failure reporting | `run_comparison`; CPU tests | pass |
| **Not established** | — | cache contents, attention correctness, sampler generality, quality, physical-ID equality, recovery, starvation, fairness, performance |

### Commands

CPU gate (`CUDA_VISIBLE_DEVICES=""`):
`bash validation_output/scheduler_comparison/20261018T195933Z/cpu_checks/run_checks.sh <dir>/cpu_checks`.
Steps 01 `py_compile`, 02 `--help` under `-W error`, 04 trailing whitespace,
and 05 `git diff --check` exited 0. Step 03,
`python -B -m unittest tests.test_compare_scheduler_gpu -v`, passed 24 tests:
the 15 earlier tests, with the six two-request rows unchanged, plus 9 new
ones. The new tests cover the three native configurations and CLI case
limits, reference replay (zero-demand decodes at 0 free blocks, one-block
growth, victims 3, 2, reset and expansion), the LP prerequisite at decision
1, LP accounting consistency and the bounded-unfinished record,
multiple-action sample association, sample mix-up detection, mid-run solver
failure with the partial trace, and 4-block capacity selection (3 fails
before initialization). The worker serialization test is unchanged and
passes. Not rerun: unchanged LP, mapper, executor, scheduler, and reference
suites, and the six two-request GPU runs.

GPU: `bash validation_output/scheduler_comparison/20261018T195933Z/run_gpu.sh <dir>`
ran, sequentially, for each policy:

```text
PYTHONPATH="$PWD" timeout 600s python -B scripts/compare_scheduler_gpu.py \
  --case memory_pressure --policy <policy> --model-path <model dir> \
  --output-dir validation_output/scheduler_comparison/20261018T195933Z/memory_pressure_<policy>
```
