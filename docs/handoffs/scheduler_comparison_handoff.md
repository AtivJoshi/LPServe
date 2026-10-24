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

## Addendum 2026-10-24: coordinated coefficient policy and integer optimum

**Scope.** One provisional coefficient policy, `coordinated_rank_utility`
(policy name `lp_coordinated`), studied against the adapted reference, the
existing uniform and rank-based LPs, and a CPU-only experimental integer
optimum of the same coordinated problem (`ilp_coordinated`). The batch has
three parts: CPU closed-loop runs, same-state comparisons, and four bounded
GPU confirmation runs. The formulation, LP extraction, resource conventions
(including both schedulers' free-block conventions), numerical policy,
native order, and failure behavior are unchanged. Q11, D-01–D-04, κ, and F02
remain OPEN. These values are demonstration inputs, not selected
coefficients, and they were not tuned after the results were seen.

**Evidence:** [`validation_output/scheduler_comparison/20261024T182347Z/`](../../validation_output/scheduler_comparison/20261024T182347Z/)
(`cpu_checks/` with `run_checks.sh`, step logs 01–13, `working_tree.diff`,
`new_files/`, `sha256sums.txt`; `cpu_comparison/` with `comparison.txt`,
`rollouts/<case>_limit<n>/<method>/{summary,decision_trace}.json`,
`same_state/`, `diagnostic_states.json`, `divergences.json`, `run_record.json`;
`gpu_<case>_<policy>_limit<n>/` with `command.txt`, `console.log`,
`exit_status.txt`, `summary.json`, `decision_trace.json`, `cleanup.json`;
`run_gpu.sh`, `run_gpu_stdout.txt`, `shell_environment.txt`,
`git_state_gpu.txt`, `sha256sums_gpu.txt`). Tested at HEAD
`870d017dac687211d834ac14cb6aecade9397248` plus the uncommitted changes
below: tracked diff `git diff HEAD` sha256 `6da0fbd2…`, the same at the CPU
gate and the GPU start, plus three new files, copied under
`cpu_checks/new_files/`. Every file hash in `cpu_checks/sha256sums.txt`
equals `sha256sums_gpu.txt` and the files at report time. Host `gpu050`,
Slurm job `66022154` (partition `gpu-preempt`), NVIDIA A16
`GPU-685ce071-19c9-a4f8-0bde-2abc602c98d2`, driver 595.91.07; modules
`uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`; `env/bin/python`
3.10.8, numpy 2.2.6, scipy 1.15.3 (HiGHS through `milp`/`linprog`), torch
2.3.0+cu121, ray 2.58.0. Same TinyLlama snapshot `fe8a4ea1…`, offline.

**Evidence review 2026-10-24.** At the review of `9b03a78`, the recorded
CPU and GPU source-file hashes matched the committed files, but the four
GPU `console.log` files were absent from the checkout. Commit `43aa837`
subsequently preserved them. Review of all four logs confirmed one
`RESULT: PASS` per run and only the bfloat16-to-float16 cast warning. The
retained exit-status files also record zero Python/timeout and `tee` exit
statuses. Console-log preservation is complete; no GPU rerun was needed.

### Files

| File | Change |
|---|---|
| `lpserve_state_mapping.py` | `coordinated_rank_utility` (new callable policy for the existing hook); uniform and rank-based policies unchanged |
| `lp_relaxation_scheduler.py` | Metadata only: `DirectIntegerSolution` provenance record; `validate_integer_plan` accepts it in `plan.extraction`; `SchedulingSuccess` documents `relaxed`/`solver` as `None` for a direct integer plan. Construction, solve, extraction, and every other validation are unchanged |
| `scripts/exact_integer_plan.py` | New, CPU-only: `solve_integer_plan`, `evaluate_assignment`, `compare_same_state`, `integer_decision` |
| `scripts/compare_coefficient_policies_cpu.py` | New CPU batch entrypoint |
| `scripts/compare_scheduler_gpu.py` | `lp_coordinated` policy (GPU and CPU); CPU-only `ilp_coordinated` method and representative cases (not accepted on the command line); per-request output limits; virtual arrival clock |
| `tests/test_compare_scheduler_gpu.py` | Fixture: sequence-ID counter (finished requests leave `seq_map`, so `len(seq_map)` reused IDs once arrivals were staggered); earlier rows pinned to the earlier three policies; 10 new tests |
| `tests/test_lpserve_state_mapping.py` | 3 new policy tests |
| `tests/test_exact_integer_plan.py` | New, 8 tests; needs `git add -f` (`.gitignore` `test*`) |

The executor, live scheduler, registry, allocator, `EngineArgs`, and
dependencies are unchanged. There is no registered or GPU ILP path.

### Policy and solver inputs (provisional, recorded in `run_record.json` and each `summary.json`)

With `r` the zero-based position in running order followed by arrived
waiting order: `w = (1/4) ** r`; for a prefill-eligible request
`q = min(remaining, c_max, b_max)`. Decode reward is `w` (decode-eligible
requests, else 0), prefill token reward `w / q` (prefill-eligible, else 0),
and preemption penalty `w` (eligible resident victims, else 0). `c_max` and
`b_max` are bound from the case's LP caps, and the driver checks them
against the scheduler's configuration. The mapper's service and victim
ranks are unchanged and are not read by the policy.

The integer solver uses `scipy.optimize.milp` with options `mip_rel_gap=0`,
`presolve=True`, and `time_limit=10` s per solve. The objective, bounds, and
rows come from `_build_lp` unchanged; every variable is integer. A
conversion tolerance of 1e-6 is applied. A result is accepted only after
checks of the status, finite values, near-integrality, objective agreement,
`I = 1{x>0}`, every row at the integer point, and `validate_integer_plan`.
Fractional or non-optimal results are never rounded into a plan, and there
is no retry. Objectives of different methods are compared with
COMPARISON_TOL 1e-7. "Optimal" means HiGHS-reported optimality within its
tolerances.

The CPU rollout uses `integer_decision`, installed on one scheduler
instance. It follows the live coordinator's order (capture → coordinated
coefficients → one integer solve → unchanged `execute_plan`). No LP is
solved for its inputs. A failed integer solve raises `LPSchedulingError`
before mutation, with no fallback; this was tested with an injected time
limit.

The virtual clock is the number of nonempty decisions executed. A request is
submitted before the first decision at or after its arrival time. When
nothing is unfinished, the clock jumps to the next arrival without a
scheduler call; this was tested with an induced gap. In the staggered cases,
request k arrives at time 2k. No staggered run went idle.

### Critical review (analytical unless marked observed)

1. **"Protect earlier residents" holds per request, not per token.** A
   full chunk and a decode are worth the same `w`, so one prefill token is
   worth `w_i/q_i`. A later decode-ready request j outbids a single token of
   an earlier prefill i whenever `q_i > 4^(r_j−r_i)`. When the budget binds,
   the policy therefore shortens the earlier request's chunk by one token to
   serve the later decode. The reference never does this: it gives the head
   its full chunk first. Observed in the `mixed_ample` diagnostic: A:7 plus
   B's decode, 7/8 + 1/4 = 1.125 > 1, and the same in the LP and the ILP.
   With q up to 32, the gap is up to two ranks. This is a consequence of the
   specified policy, not a defect, and it is not corrected here.
2. **Voluntary replacement by a waiting request is never profitable on its
   own.** The waiting rank is at least |R|, so its full-chunk value is below
   the penalty plus the lost action value of any resident. Observed: zero
   `resident_capacity` victims in every coordinated run. The uniform and
   rank-based LPs had 1–7 such decisions per run. The only coordinated
   replacement (memory-pressure decision 3) comes from the LP decode
   prerequisite: A and B have demand 0 and the pool has 0 free blocks, so a
   victim is forced. Evicting B (penalty 1/4) frees 2 blocks; one funds D's
   admission (+16/1024) and one gives H = 1. Utility 0.765625, matched by
   enumeration. C (rank 2) was skipped because its 2-block charge would leave
   H = 0.
3. **The weights shrink geometrically with rank.** The smallest coefficients
   observed were 3.05e-5 (`mixed_burst`: r = 5, q = 32), well above HiGHS's
   documented default dual feasibility tolerance (1e-7) and the extraction
   threshold (≈1e-9). By arithmetic, `4^-r/32` reaches about 1.2e-7 at r = 9
   and 1e-9 near r = 13. The policy as specified is therefore numerically
   meaningful only for small universes (about ten ranked requests at q ≤ 32)
   under the unchanged numerical policy. No weight was clamped or rescaled.
   Larger universes need an explicit decision.
4. **The LP decode prerequisite remains the main source of LP-only
   preemption** (inherited convention, design §8.3, unchanged). Coordinated
   memory-pressure victims: 4 decisions cite `lp_decode_prerequisite` and 1
   cites `decode_block_demand`. The reference had 1.
5. **Driver weakening caught and fixed before the gate.** An early draft of
   the arrival loop drained the worker operation log after every decision.
   That would have masked the "worker operations outside a step" check. The
   log is now drained only after an actual submission.
6. **Inherited defect seen, not repaired:** `SchedulerOutputs.__repr__`
   names a missing `new_seqs` attribute. It surfaced when formatting a check
   message, and the entrypoint now formats only the type.

No defect was found in the formulation, extraction, or executor paths used.
Review scope: the policy, mapper construction, `_build_lp`/`_residuals`, the
constructor and plan validator, the executor's precommit path for a
`SchedulingSuccess` with `relaxed = None` (it reads only `plan` and
`problem_id`), the driver checks, and the reference proposal. Kernels, cache
contents, and multi-worker paths were not reviewed.

### Results (executed)

**Diagnostic single states.** These were constructed with native allocation
and progress methods and mapped once per method. The LP plan and the integer
plan were each executed by the native executor on a separate copy. The
reference proposal was evaluated under the LP contracts, and bounded
enumeration ran.

| State | LP relaxed | LP plan | Integer optimum (= enumeration) | Reference (LP-contract evaluation) |
|---|---|---|---|---|
| `mixed_ample` (A 8 left, B decode-ready, budget 8, s 2) | 1.125 | A:7 + B decode, 1.125 | same | A:8, 1.0: strict objective preference |
| `mixed_pressure` as run (+ 0 free, later C legal victim; B had **demand 0**, not the intended 1: C's eviction funded only the LP decode prerequisite; corrected 2026-10-24, see below) | 1.1042 | A:7 + B decode, preempt C, 1.0625 | same | A:8, 1.0: strict objective preference |
| `zero_free_decodes` (4 decode-ready, 0 free, demand 0) | 1.3031 | decode A,B,C, preempt tail D, 1.296875 | same (unique by enumeration) | decode all four: contract difference (`decode_prerequisite`) |
| `fractional_admission` (charges 16, 1; pool 16; cap 1) | 1.1875 (A 15/16, B 1) | B:1, 0.25 | A:1, 1.0 | A:2: contract difference (chunk cap) |

**Initial diagnostic review 2026-10-24 at `9b03a78`.** Source and retained
state showed that the original `mixed_pressure` fixture had B demand 0,
although its description said 1. The original artifacts and utilities remain
unchanged. The subsequent fixture correction, explicit demand assertion, and
fresh CPU evidence are recorded in the correction section below; the other
workload results and scheduler contracts are unaffected.

In the last row, the relaxation gap is 0.1875 and the extraction loss is
0.75. Extraction tries B first because its e = 1 is larger than A's 0.9375,
and A can then no longer be admitted.

**CPU closed-loop runs** (35 runs, all `validation_passed`; full table in
`cpu_comparison/comparison.txt`). Columns: total preemptions, preemptions
after output, discarded prompt progress, replayed prefill tokens, and
overgeneration.

| Workload | Reference | Uniform LP | Rank-based LP | Coordinated LP = ILP |
|---|---|---|---|---|
| two_request, limit 1 | drained 6; 0 | bounded 8; 7 victims, 39 replayed | bounded 8; 7, 39 | drained 6; 0 |
| two_request, limit 2 | drained 4 | drained 6 | drained 6 | drained 6; 0 |
| memory_pressure | drained 8; 2 (2), 32 discarded, 32 replayed, +2 | bounded 16; 26, 407 replayed | bounded 16; 28, 437 | drained 15; 5 (3), 83 discarded, 83 replayed, +4 |
| equal_burst_ample | drained 9; 0 | drained 14; 11, 180 | drained 9; 4, 60 | drained 9; 0 |
| mixed_burst_ample | drained 10; 0 | drained 13; 6, 123 | drained 9; 1, 17 | drained 10; 0 |
| mixed_staggered_ample | drained 16; 0 | drained 16; 0 | drained 16; 0 | drained 16; 0 |
| mixed_staggered_tight | drained 16; 0 | drained 18; 1, 32 | drained 21; 2, 63 | drained 17; 0 |

The two_request limit-1 hypothesis held: six decisions, no preemption. In
memory_pressure, the coordinated LP drained where the uniform and rank-based
LPs stayed bounded-unfinished. It still preempted 5 times against the
reference's 2 and replayed 83 tokens against 32. Completion is observed here,
not guaranteed.

**Same-state, coordinated LP vs integer optimum** (79 identical problems):
all 79 have identical selected actions, and extraction loss is 0
everywhere. A relaxation gap appears in 8 problems: memory_pressure
decisions 1, 3, 4, 6, 8, 9, 10 (0.006–0.389) and tight-staggered decision
10 (0.047). The LP bound ≥ integer optimum ≥ LP plan holds in every problem.
Bounded enumeration agreed with the integer optimum in all 136 same-state
problems small enough to enumerate (12 were skipped as too large). The
largest node count was 1.

**Same-state, reference vs integer optimum.** Each of the 69 reference
states was mapped with coordinated coefficients and the case's LP caps. In
60, the integer optimum selects the reference's actions. The other 9 are
contract differences: `action_width` (s_max = 1) at three two_request
limit-2 states, and `decode_prerequisite` at five memory_pressure states and
one tight-staggered state. No rollout state showed a tie or a strict
objective preference.

**Does integer optimization change the trajectory?** Not in this batch.
The ILP rollouts emitted exactly the coordinated-LP decisions in all seven
workloads, so the relaxation gaps never changed a selected plan. The
`fractional_admission` state shows that the two can separate, but no
rollout reached such a state.

**Where the coordinated runs first deviate from the reference** (identical
before-states in every case): two_request limit 2, decision 0 (action cap
s_max = 1); memory_pressure decision 1 (LP decode prerequisite: tail D
preempted, the reference decodes all four); mixed_staggered_tight decision
10 (prerequisite: the LP omits the reference's F:30 admission). equal_burst
and mixed_burst first differ at decision 2 only in emitted order, with the
same selected set: the LP declared order puts zero-demand decodes before
one-block decodes. mixed_staggered_ample and two_request limit 1 never
diverge.

**GPU confirmation** (four fresh processes, sequential, one attempt each).
Every run printed `RESULT: PASS`, drained, exited 0 (timeout/Python and tee),
and shut down its own Ray. Profiling reported 15616 blocks (two_request) and
15606 (memory_pressure); exactly 8 and 4 blocks were initialized natively.
The only warning was the bfloat16→float16 cast. Each GPU decision sequence
equals its CPU trace decision for decision.

| Run | Outcome | Decisions | Preemptions | Physical IDs differ after |
|---|---|---|---|---|
| coordinated LP, two_request limit 1 | drained | 6 | 0 | 0 decisions |
| reference, two_request limit 1 | drained | 6 | 0 | 0 |
| coordinated LP, memory_pressure | drained | 15 | 5 | 0 |
| reference, memory_pressure | drained | 8 | 2 | 3 (recorded, not required) |

### Checks (CPU gate, `CUDA_VISIBLE_DEVICES=""`)

`bash validation_output/scheduler_comparison/20261024T182347Z/cpu_checks/run_checks.sh <run dir>`. Every step exited 0:

| Step | Result |
|---|---|
| 01 `py_compile` of the 8 changed or new files | pass |
| 02/03 `--help` for both entrypoints under `-W error` (CPU entrypoint test: no torch, Ray, or engine imported) | pass |
| 04 `tests.test_compare_scheduler_gpu` | 34 passed (24 earlier with the two-request and memory-pressure rows unchanged, plus 10 new) |
| 05 `tests.test_exact_integer_plan` | 8 passed (shared model and integer domains; 154 states against enumeration; statuses 1–4, exceptions, malformed vectors; near-integral acceptance; validator) |
| 06 `tests.test_lpserve_state_mapping` | 18 passed (3 new) |
| 07 `tests.test_lp_relaxation_scheduler` | 38 passed |
| 08 `tests.test_lpserve_plan_execution` | 16 passed |
| 09 `tests.test_lp_scheduler` | 24 passed |
| 10 `tests.test_check_lp_scheduler_ordering` | 5 passed |
| 11 trailing whitespace, 12 `git diff --check` | pass |
| 13 `compare_coefficient_policies_cpu.py --output-dir <run>/cpu_comparison` | `RESULT: PASS` |

Not run: the adapted-reference unit suite, the other `check_*` script
suites, and `test_lp_engine_args`/`test_run_lp_inference`. Their code is
unchanged and not imported differently. No GPU representative workload, GPU
ILP, or earlier-policy GPU rerun was run (none was authorized).

GPU: `bash validation_output/scheduler_comparison/20261024T182347Z/run_gpu.sh <run dir>`, each run as

```text
PYTHONPATH="$PWD" timeout 600s python -B scripts/compare_scheduler_gpu.py \
  --case <two_request|memory_pressure> --policy <lp_coordinated|adapted_vllm> \
  --resident-limit <1|4> --model-path <model dir> --output-dir <run dir>/gpu_<case>_<policy>_limit<n>
```

**Not established:** cache contents, attention correctness, sampler
generality, quality, physical-ID equality, recovery, starvation, fairness,
latency, throughput, and eventual completion. Inherited behavior (final-
prefill sample discarded, counter reset with overgeneration after restart:
reference +2 and coordinated +4 in memory_pressure) is recorded, not
repaired.

## Correction 2026-10-24: mixed-pressure diagnostic with one-block decode demand

**What was wrong (source and retained evidence).** In the 2026-10-24 batch,
`DIAGNOSTIC_STATES["mixed_pressure"]` built B as `(16, 16, 0)`: a complete
16-token prompt with no generated token. At block size 16, B's logical and
physical tables then both hold one block, so its decode demand is 0, while
the fixture note claimed one block. The retained plan confirms this
(`decode_demand` 0, `H` 1 in
`20261024T182347Z/cpu_comparison/diagnostic_states.json`). C's eviction
therefore funded only the LP decode prerequisite (at least one free block
before any decode), not an actual allocation. The relaxed value 1.1042
follows from that. Without an actual-memory charge for B, the relaxation can
satisfy `y_B + y_C ≤ 3H` with `z_C = 1/3`: 7/8 + 1/4 − 1/48 = 1.1042. The
old run's numbers stay as recorded for what was actually tested. The fixture
description in that run directory is not rewritten.

**Correction.** Only B's tuple changed, to `(16, 16, 1)`: one generated
token appended after allocation, through native sequence methods. All other
inputs are unchanged: A `(16, 8, 0)`, C `(8, 8, 0)`, pool 3 blocks with 0
free, resident limit 3, context 32, budget/`b_max`/`c_max` 8, `s_max` 3, and
the same coordinated coefficients and numerical policy. The diagnostic
record now also keeps the mapped decode demands and every resident's
physical block count before and after each execution.

**Analytical expectation.** B now carries actual-memory demand 1, so
`y_B ≤ H = z_C`. Each unit of B's decode then requires a full unit of C's
eviction, and the relaxation can no longer fund the decode with a fraction
of C. The LP optimum should equal the integer optimum: A:7 + B decode,
preempt C, 7/8 + 1/4 − 1/16 = 1.0625. A fractional decode of C cannot help,
since its 1/16 per token is below A's 1/8.

**Executed (CPU only)** —
[`validation_output/scheduler_comparison/20261024T193028Z_mixed_pressure_one_block_demand/`](../../validation_output/scheduler_comparison/20261024T193028Z_mixed_pressure_one_block_demand/)
(`run_checks.sh`, step logs 01–07, `mixed_pressure_diagnostic.json`,
`working_tree.diff`, `sha256sums.txt`, `environment.txt`, `git_state.txt`).
Base `43aa8378bb466a80ee0a09ac55772fb5747964ba` plus uncommitted changes to
`scripts/compare_coefficient_policies_cpu.py` and
`tests/test_compare_scheduler_gpu.py`. Host `gpu049`, Slurm job `66023208`,
`CUDA_VISIBLE_DEVICES=""`, `env/bin/python` 3.10.8, numpy 2.2.6, scipy
1.15.3.

- Mapped state: B has output length 1, 2 logical blocks, 1 physical block,
  and decode demand 1. C has demand 0. A has 8 tokens left and charge 0.
  Free blocks: 0.
- LP: relaxed optimum 1.0625 (the old fixture gave 1.1042), plan A:7 + B
  decode + preempt C = 1.0625. The integer optimum gives the same actions and
  1.0625. Relaxation gap and extraction loss are both 0. Bounded enumeration
  (90 plans) gives a best of 1.0625.
- Native execution (separate equivalent builds for LP and integer; the
  executor validated each complete plan before mutation): emitted
  `[[0, 7], [1, 0]]`, victims `[2]`. Physical blocks went from A 1, B 1, C 1
  (free 0) to A 1, B 2 (free 0). C's block was released and taken by B's
  decode.
- The reference proposes A:8 with no victim (1.0), which is LP-feasible: a
  strict objective preference for the integer optimum.

| Step | Command | Result |
|---|---|---|
| 01 | `python -B -m py_compile` (2 changed files) | exit 0 |
| 02 | import both modules under `-W error` | exit 0 |
| 03 | `python -B -m unittest tests.test_compare_scheduler_gpu.CoordinatedPolicyTest.test_mixed_pressure_uses_one_block_decode_demand -v` | 1 passed |
| 04 | `python -B -m unittest tests.test_compare_scheduler_gpu -v` | 35 passed (the 34 earlier expectations unchanged, plus the new test) |
| 05 | standalone `run_diagnostic("mixed_pressure", …)` → JSON | exit 0 |
| 06 | bounded enumeration vs integer optimum | agrees |
| 07 | `git diff --check` on the changed files and this handoff | exit 0 |

Not rerun: the 35-workload CPU batch, the other diagnostics, the other
suites, and all GPU runs. Their recorded results remain evidence for the
code and inputs they tested. This correction does not rerun or extend GPU
validation; the mixed-pressure diagnostic was never a GPU case.

**GPU console logs.** The four original logs
(`20261024T182347Z/gpu_{two_request,memory_pressure}_{lp_coordinated,adapted_vllm}_limit{1,4}/console.log`)
are present and tracked, added by `43aa837`. Their preservation is
complete.

## Addendum 2026-10-24: reciprocal-square policy as queues grow

**Scope.** One provisional coordinated policy with slower rank decay,
`reciprocal_square_rank_utility` (LP method `lp_reciprocal_square`), and
its CPU-only integer-driven method (`ilp_reciprocal_square`), compared with
the adapted reference on six queue-growth workloads. The six-request anchor
also runs the uniform, rank-based, and geometric coordinated LPs. Every
reached decision of the new runs has a same-state integer comparison. The
two authorized GPU runs of the constrained 48-request case were attempted
and **failed at native engine initialization**, before any decision,
because 12 blocks are fewer than one 256-token context needs (see the GPU
section). GPU confirmation is therefore not established. The geometric
policy is unchanged and was not run beyond the anchor. The formulation,
extraction, numerical policy, resource conventions, executor, live
scheduler, registry, allocator, `EngineArgs`, and dependencies are
unchanged. Q11, D-01–D-04, κ, and F02 remain OPEN. The exponent two and
all workload values are demonstration inputs and were not tuned after
results were seen.

**Evidence:** [`validation_output/scheduler_comparison/20261024T210104Z_queue_growth/`](../../validation_output/scheduler_comparison/20261024T210104Z_queue_growth/)
(`cpu_checks/` with `run_checks.sh`, step logs 01–13, `working_tree.diff`,
`git_state.txt`, `environment.txt`, `sha256sums.txt`; `cpu_comparison/` with
`comparison.txt`, `divergences.json`, `run_record.json`, and
`rollouts/<case>/<method>/{decision_trace.json.gz,summary.json,comparisons.json,console.log}`;
`gpu_queue48_burst_pool12_<policy>/` and `gpu_offline_comparison/` (see the
GPU section); `run_gpu.sh`, `run_gpu_stdout.txt`, `shell_environment.txt`,
`git_state_gpu.txt`, `sha256sums_gpu.txt`). Tested at HEAD
`fce4fd1b57527f3d198aecb39dadc807799359d9` plus uncommitted changes to the
six files below: `git diff HEAD` sha256 `e6dc3fd2…`, the same at the CPU gate
and the GPU start. Every file hash in `cpu_checks/sha256sums.txt` equals
`sha256sums_gpu.txt`. There were no untracked source files. Host `gpu051`,
Slurm job `66024401`, NVIDIA A16
`GPU-3b4275a6-87bc-8602-343d-c27ed356bfdd`, driver 595.91.07; modules
`uri/main`, `Python/3.10.8-GCCcore-12.2.0` (plus `CUDA/12.1.1` for GPU);
`env/bin/python` 3.10.8, numpy 2.2.6, scipy 1.15.3 (HiGHS), torch
2.3.0+cu121, ray 2.58.0. Same TinyLlama snapshot `fe8a4ea1…`, offline.
Decision traces are gzip-compressed (`gzip -n` or `mtime=0`); the GPU
traces' uncompressed sha256 is kept beside them.

### Files

| File | Change |
|---|---|
| `lpserve_state_mapping.py` | `reciprocal_square_rank_utility` (new callable for the existing hook). Earlier policies are unchanged |
| `scripts/compare_scheduler_gpu.py` | `lp_reciprocal_square` (CLI and CPU) and `ilp_reciprocal_square` (CPU only); `QUEUE_GROWTH_CASES` (only `queue48_burst_pool12` on the command line, resident limit 8); `capture_reference_problems`; the pre-request watermark check now expects the native `int(0.01 * pool)` (1 at 128 blocks) instead of 0 |
| `scripts/exact_integer_plan.py` | `problem_from_record`, `lp_summary`, `integer_summary`, `compare_summaries`; `plan_actions` accepts JSON records. `compare_same_state` keeps its outputs (adds an `integer` summary) |
| `scripts/compare_coefficient_policies_cpu.py` | `--batch queue_growth` and `--batch retained_runs --runs … [--cpu-batch …]`; the default batch and its CLI are unchanged; `compare_reference` delegates to `classify_reference` with unchanged outputs |
| `tests/test_compare_scheduler_gpu.py` | 9 new tests; three help-text assertions updated for the added choices |
| `tests/test_lpserve_state_mapping.py` | 3 new policy tests |

### Policy and comparison inputs (provisional)

With `r` the zero-based position in running order followed by arrived
waiting order: `w = 1 / (1 + r)^2`. For a prefill-eligible request,
`q = min(remaining, c_max, b_max)`. Decode reward is `w`, prefill token
reward `w / q`, and penalty `w`, each for eligible requests only. Capture,
ranks, and execution are as for the geometric policy.

Workloads: request k has prompt length `[8,16,17,32,64,128][k mod 6]`,
output limit `[2,4,8][k mod 3]`, and prompt IDs from `1000 + 256k`. Block
size 16, `max_model_len` 256, resident limit 8, budget/`b_max`/`c_max` 128,
`s_max` 8, bound `8n` nonempty decisions. Burst arrivals are all at
virtual time 0; staggered request k arrives at `2k`. Cases:
`queue6_burst`, `queue12_burst`, `queue24_burst`, `queue48_burst`, and
`queue48_staggered` use 128 blocks (native watermark 1);
`queue48_burst_pool12` uses 12 blocks (watermark 0).

Integer comparisons reuse `solve_integer_plan` unchanged: `milp`,
`mip_rel_gap` 0, presolve, 10 s per solve, conversion tolerance 1e-6,
comparison tolerance 1e-7. Each LP decision's own mapped problem is solved
once offline after the rollout. The integer-driven method's own result is
reused, never solved again. For the reference, the quiescent pre-decision
state is mapped once, read-only, with reciprocal-square coefficients and the
case's LP caps (`capture_reference_problems`, which skips idle steps). The
proposal is then classified against that problem. The comparisons read the
retained JSON: each problem is rebuilt, validated, and checked to serialize
back to the identical record. CPU and GPU runs take the same path.

### Critical review (analytical unless marked observed)

1. **The 128-block cases cannot create memory pressure.** At most eight
   residents of at most 136 tokens (9 blocks) hold 72 blocks. No decode can
   lack a block, and the LP prerequisite always has H ≥ 1. A voluntary
   replacement never pays: a waiting request's weight is below every
   resident's, so its value is less than the penalty. Observed: zero
   preemptions by every method in all five 128-block cases. Victim
   positions, discarded progress, and contract differences come only from
   `queue48_burst_pool12`.
2. **The staggered case never builds a queue** (observed). With arrivals
   every two decisions, at most four requests are present (maximum rank 3).
   All three methods select identically at every decision. The LP's
   emitted order differs from the reference's at decisions 13, 25, …, 85
   (its demand-group order), with the same selection. This case does not
   test long-queue rank effects.
3. **Prompt and output lengths are paired.** The periods 6 and 3 give only
   six request types, for example every 128-token prompt has 8 outputs.
4. **With per-token normalization and slow decay, a binding budget favors
   short prompts over the waiting order.** One prefill token is worth
   `w/q`, and adjacent weights differ by only `((r+2)/(r+1))^2` (1.31 at
   r = 6, 1.04 at r = 46). A later short request therefore outbids an
   earlier long one per token. Under the geometric policy the factor is 4
   per rank, so rank order dominates after about two positions. Observed:
   at decision 0 of every burst case with at least 12 requests, the LP
   admits R06, R07, and R08 (8, 16, 17 tokens) and gives R04 (64) at most
   14 tokens or none. The reference admits R00–R04 in order (R04 chunked to
   55). This is the specified policy, not a defect. It is not a fairness or
   starvation result.
5. **Weak rank separation lets block counts decide victim choice** (observed
   and integer-optimal). At pool-12 LP decision 2, three one-block decodes
   need 3 blocks with 1 free. Preempting R08 (rank 6, 2 blocks, cost 2/49)
   beats preempting tail R12 together with R06 (cost about 0.111). At ILP
   decision 2, a forced victim R04 (4 blocks) also funds two admissions.
   The reference always evicts the tail.
6. **The relaxed decode prerequisite causes non-tail victims through
   extraction** (observed, pool-12 LP decisions 20 and 34). All residents
   are decode-ready with 0 free blocks. The row `Σy ≤ S_max·H` holds with a
   small fraction of a 4-block resident (z = 0.121 and 0.4). Extraction
   orders victims by z, preempts that resident whole, and fills the freed
   blocks with admissions. The integer optimum preempts the 1-block tail
   instead. Losses: 0.084 and 0.091. This is the inherited convention
   (design §8.3) together with the relaxation and extraction, not the
   coefficients. The convention and extraction are unchanged; changing
   either needs a separate decision.
7. **The action cap splits admissions fractionally** (observed, decision 0
   of `queue24_burst`, `queue48_burst`, and `queue48_burst_pool12`). The
   eighth action is shared between R04 (I = 0.107, 6.9 tokens) and R12
   (I = 0.893). Extraction keeps R12, while the integer optimum takes R04's
   leftover tokens. Losses are 0.0028 and 0.0011. This is a known
   fractional-structure limitation (design §10.6). Pool-12 decision 13 shows
   a 6.2e-5 knapsack loss: extraction admits a 2-block request, and the
   optimum fits two 1-block ones.
8. **A coefficient above the solver tolerance does not by itself separate
   priorities.** Observed: the smallest positive coefficient is 3.4e-6 at
   48 requests (rank 47, q = 128), 34 times HiGHS's default 1e-7 dual
   feasibility tolerance. Geometric weights would give about 4e-31. But the
   smallest positive difference between two requests' per-token prefill
   rewards is 1.9e-7 at 24 requests and 5.0e-8 at 48. Pairs that close are
   not reliably distinguished by the solver. Exact ties also occur, from
   coincidences of `w/q`. In the observed runs, every LP/ILP and
   reference/ILP disagreement had an objective gap of at least 6.2e-5, so
   none is attributable to such near ties. Nothing was rescaled, floored,
   or clipped.
9. **Watermark.** The driver had required watermark 0 before submitting
   requests. At 128 blocks the native value is 1. The check now expects the
   shared manager's native `int(0.01 * pool)`, which is unchanged (0) for
   every earlier case. LP problems carry h = 1, and the reference uses the
   same manager value.

Review scope: the new policy and its hook; mapper capture and construction;
`_build_lp`, the extraction loop, and plan validation (read, unchanged);
the integer helper; the driver checks and capture wrapper; the reference
proposal path; and the comparison and coverage code. No defect was found
in the formulation, extraction, or executor paths. Kernels, cache contents,
and multi-worker paths were not reviewed.

### CPU results (executed; `cpu_comparison/comparison.txt`)

All 21 rollouts passed validation and drained within the bound. Columns:
decisions; preemptions (after output); discarded prompt tokens (equal to
replayed tokens here); overgeneration; admission-order and completion-order
inversions (pairs served out of request order).

| Case | Reference | Reciprocal-square LP | Reciprocal-square ILP |
|---|---|---|---|
| queue6_burst | 11; 0; 0; 0; 0 / 3 | 11; 0; 0; 0; 0 / 3 | 11; 0; 0; 0; 0 / 3 |
| queue12_burst | 15; 0; 0; 0; 0 / 14 | 15; 0; 0; 0; 3 / 17 | 15; 0; 0; 0; 3 / 17 |
| queue24_burst | 24; 0; 0; 0; 0 / 34 | 23; 0; 0; 0; 32 / 67 | 24; 0; 0; 0; 8 / 44 |
| queue48_burst | 42; 0; 0; 0; 0 / 80 | 40; 0; 0; 0; 33 / 110 | 40; 0; 0; 0; 9 / 89 |
| queue48_staggered | 104; 0; 0; 0; 0 / 22 | 104; 0; 0; 0; 0 / 22 | 104; 0; 0; 0; 0 / 22 |
| queue48_burst_pool12 | 117; 6 (3); 102; 3; 0 / 27 | 118; 5 (3); 226; 3; 208 / 246 | 117; 8 (1); 193; 1; 202 / 224 |

**Resident continuation.** In every run, by every method, each resident
received an action at every decision: zero resident omissions. All
requests completed in every run. Generated-token events exceeded the output
limits only by the overgeneration after restarts (pool 12: 3, 3, and 1 for
the reference, LP, and ILP; inherited counter reset). In the 128-block
cases nothing was
preempted. In the 12-block case, the reciprocal-square LP discarded 226
prompt tokens against the reference's 102. Its victims were at running
positions 6/8, 4/5, 2/4, 4/5, and 2/4; the reference's were always the
tail (3/4). Three LP decisions combined a victim with admissions
(findings 5 and 6); the reference never did.

**Service order.** In `queue48_burst_pool12`, the mean first-prefill /
completion decision index of the eight 128-token requests was 77.5 / 85.5
for the LP, against 57 / 65.5 for the reference. The 8-token requests had
9.4 / 11.4 against 44.4 / 46.4. These are decision indices, not latency.
The total drain was 118 against 117 decisions.

**Anchor (six requests).** The geometric and reciprocal-square LPs, the
rank-based LP, and the ILP make identical selections at every decision.
They differ from the reference only in emitted order at decision 2 (the LP
demand-group order). The uniform LP differs (four admission inversions).
At six requests, the decay rate makes no observed difference.

**Coverage** (`comparison.txt`, "Comparison coverage"). There were 967
reached decisions in 21 runs, each with exactly one record. All had a valid,
round-trip-identical problem, with zero mapping failures, missing problems,
missing records, or comparison errors. 656 offline integer solves were
attempted (every LP and reference decision), and 311 integer-driven
results were reused. All 967 were HiGHS-optimal (raw status 0), with
zero time limits and zero other failures. The maximum integrality
deviation was 7.1e-14, and the longest offline solve took about 0.01 s.

**Same state, reciprocal-square LP vs integer optimum** (identical
problems). The actions were identical except in 4 of 23 (`queue24_burst`),
4 of 40 (`queue48_burst`), and 6 of 118 (`queue48_burst_pool12`) decisions.
Each exception is an extraction loss (findings 6 and 7). Relaxation gaps
were present in 6, 13, and 92 decisions of those cases (maximum 0.219),
without changing the actions elsewhere. The bound LP ≥ integer ≥ LP plan
held everywhere, and there was no near tie. The anchor's uniform LP had
two exact ties with different actions; the other anchor LPs matched the
optimum.

**Same state, reference vs reciprocal-square integer optimum.** Only the
integer optimum of the same diagnostic problem is compared, never
objective values across policies. The integer optimum selects the
reference's actions in 11/11, 13/15, 21/24, 35/42, 104/104, and 64/117
states. The rest are strict objective preferences (2, 3, 7, and 45; the
smallest gap is 0.0053). In pool 12 there are also 8 contract differences,
all `decode_prerequisite` (the reference decodes zero-demand requests at 0
free blocks). No state showed a tie.

**Closed loop vs same state.** The LP and ILP rollouts first select
differently at decision 0 in `queue24_burst`, `queue48_burst`, and
`queue48_burst_pool12`: the extraction loss of finding 7 changes the
trajectory from the first step. They never diverge in the other three
cases. Against the reference, both differ at decision 0 in every burst case
with at least 12 requests (finding 4). In the anchor and staggered cases
they differ only in emitted order. Later decisions are not corresponding
states, so only the same-state records attribute gaps.

**Does integer optimization help?** It changes trajectories but not
uniformly. It reduces admission inversions (8–9 against 32–33 in the
24/48-request bursts). In pool 12 it preempts more (8 against 5) but after
output less often (1 against 3), and discards 193 against 226 tokens.

**Checks** (`CUDA_VISIBLE_DEVICES=""`,
`bash <run>/cpu_checks/run_checks.sh <run>`; every step exited 0):

| Step | Result |
|---|---|
| 01 `py_compile` (6 changed files) | pass |
| 02/03 `--help` for both entrypoints under `-W error` | pass |
| 04 `tests.test_compare_scheduler_gpu` | 44 passed (35 earlier, three help assertions updated for added choices, plus 9 new) |
| 05 `tests.test_exact_integer_plan` | 8 passed (unchanged) |
| 06 `tests.test_lpserve_state_mapping` | 21 passed (3 new) |
| 07 `tests.test_lp_scheduler`, 08 `tests.test_lpserve_plan_execution` | 24, 16 passed (mapper module changed) |
| 09/10 earlier coordinated batch regenerated outside the repository | `RESULT: PASS`; its report differs from the retained one only in the `mixed_pressure` line corrected at `fce4fd1` |
| 11 trailing whitespace, 12 `git diff --check` | pass |
| 13 `compare_coefficient_policies_cpu.py --batch queue_growth` | `RESULT: PASS` |

The 9 new driver tests cover: the six cases' inputs and the single GPU
case; hook selection and the native watermark (1 at 128 blocks, 0 at 12);
detection of geometric coefficients by the reciprocal-square check; one
complete, round-trip-identical record per decision for the reference, LP,
and ILP (no integer re-solve for the ILP; the capture does not change the
reference's decisions and maps nothing at idle); missing problems and
corrupt records are counted and fail completeness, while a mapping failure
is an explicit record; problem reload solves to the recorded plan; an
injected offline time limit keeps status, incumbent, bound, gap, and node
count, while the other decisions continue; an injected ILP-rollout time
limit stops the rollout before mutation with no LP fallback, and its record
is reused; and command-line limits.

Not run: `test_lp_relaxation_scheduler`, `test_check_lp_scheduler_ordering`,
the adapted-reference and other `check_*` suites, `test_lp_engine_args`,
and `test_run_lp_inference`. Their code is unchanged and not imported
differently. Earlier GPU runs were not rerun.

### GPU confirmation (executed; failed before any decision)

`bash <run>/run_gpu.sh <run>` ran two fresh processes sequentially, once
each, for `--case queue48_burst_pool12` with `--policy lp_reciprocal_square`
and then `--policy adapted_vllm` (`timeout 600s`, default resident limit
8). Both printed `CHECK FAILED` and `RESULT: FAIL workload: failure`. The
timeout/Python exit status was 1 and tee's was 0. Each took about 45 s. The
console shows profiling, then `# GPU blocks: 12` (the driver's capacity
selection), then:

```text
ValueError: Not enough available memory to schedule a request will maximum
allowed length 256. Need 16, available 12 gpu blocks.
```

raised by the inherited check in `BaseLLMEngine._init_cache`
(`sarathi/engine/base_llm_engine.py`, before `init_cache_engine`). That
check requires at least `ceil(max_model_len / block_size)` = 256/16 = 16
blocks. No engine was constructed (`cleanup.json`:
`engine_constructed: false`), no decision ran, and each process shut down
its own Ray. The only warning was the bfloat16→float16 cast. The retained
traces are empty, with their uncompressed sha256 kept beside them. The
offline comparison (`gpu_offline_comparison/`) printed `RESULT: FAIL`: the
empty runs have zero decisions and do not match the CPU rollouts. It did
not pass silently. There was no retry.

**Cause (a defect in the approved inputs, found too late).** The CPU
fixture sets the pool directly and never runs `_init_cache`, so the CPU
gate accepted a configuration that the native engine rejects. The earlier
`memory_pressure` case met the rule exactly (4 blocks = 64/16). The
rule was not checked before the GPU step. The CPU results above remain valid as
CPU scheduler observations of that state. The state is not reachable
through native engine initialization with these inputs.

**Pending decision (not made here).** Possible changes:

- set `max_model_len` to 192 for `queue48_burst_pool12` (12 × 16);
- raise its pool to at least 16 blocks;
- bypass the native check (not recommended: it weakens an engine
  validation).

In the scheduling path, `max_model_len` appears only in guards: context
≤ `max_model_len` in the mapper, LP checks, and executor, and decode needs
length < `max_model_len` in the reference. Every request has at most 136
tokens. An in-memory scratch what-if, not retained and not evidence, gave
identical CPU decision sequences at 192 for the LP and the reference, 118
and 117 decisions. The smallest change is therefore `max_model_len` 192,
with the CPU case and its comparisons rerun and the two GPU runs repeated.
That changes an approved input, so it awaits approval.

**Not established:** GPU execution of any queue-growth decision, cache
contents, attention correctness, sampler generality, quality,
physical-ID equality, recovery, starvation, fairness, latency, throughput,
and eventual completion beyond these bounded CPU runs. Inherited behavior
(final-prefill sample discarded, counter reset with overgeneration after
restart) is recorded, not repaired.

## Correction 2026-10-24: constrained case at context 192 and GPU confirmation

**Approved change.** The user explicitly approved `max_model_len` 192 for
`queue48_burst_pool12` only. This satisfies native cache initialization
(`ceil(192/16)` = 12 blocks) while keeping the 12-block pool, and with it
the memory pressure. Every other input is unchanged: the 48 requests,
lengths, arrivals, resident limit 8, budgets 128, action cap 8, bound 384,
coefficients, numerical policy, and model settings. The other five
queue-growth cases keep 256. The native engine check was neither bypassed
nor weakened. The failed GPU attempts and the CPU results at 256 above
remain historical evidence for their inputs.

**Evidence:** [`validation_output/scheduler_comparison/20261024T213405Z_queue_growth_context192/`](../../validation_output/scheduler_comparison/20261024T213405Z_queue_growth_context192/)
(`cpu_checks/` with `run_checks.sh`, step logs 01–10, `working_tree.diff`,
`sha256sums.txt`, `git_state.txt`, `environment.txt`, and
`runner_propagation/`; `cpu_comparison/` holds the corrected CPU rerun of
this case only; `gpu_queue48_burst_pool12_{lp_reciprocal_square,adapted_vllm}/`
with `command.txt`, `console.log`, `exit_status.txt`, `summary.json`,
`decision_trace.json.gz` plus the uncompressed sha256, and `cleanup.json`;
`gpu_offline_comparison/`; `run_gpu.sh`, `run_gpu_stdout.txt`,
`runner_failures.txt`, `runner_exit_status.txt`, `shell_environment.txt`,
`git_state_gpu.txt`, `sha256sums_gpu.txt`). Tested at HEAD
`7cdf3868565ad608be361009b8e8d35860b46795` plus uncommitted changes to
`scripts/compare_scheduler_gpu.py`, `scripts/compare_coefficient_policies_cpu.py`,
and `tests/test_compare_scheduler_gpu.py`. The `git diff HEAD` sha256
`d7076e5d…` is the same at the CPU gate and the GPU start. File hashes are
equal at the gate, the GPU start, and report time. Host `gpu052`, Slurm job
`66025629`, NVIDIA A16 `GPU-e400b1c6-777d-5eac-faef-7253f7d4d5f8`, driver
595.91.07. Modules, interpreter, packages, and the TinyLlama snapshot
`fe8a4ea1…` are as before (`shell_environment.txt`).

### Changes

- `compare_scheduler_gpu.py`: `_queue_growth` takes a per-case
  `max_model_len` (default 256), and the constrained case passes 192.
  `check_native_capacity` mirrors the native rule `num_gpu_blocks >=
  ceil(max_model_len / block_size)`. `main` runs it first inside its
  recorded `try`, so an incompatible case fails, with `summary.json`
  written, before engine arguments, Ray, workers, or CUDA. The native check
  still runs afterwards. Passing the preflight proves only that this one
  rule holds.
- `compare_coefficient_policies_cpu.py`: `--batch queue_growth --cases …`
  limits the batch to the named cases, recorded as `selected_cases`. The
  default is all six, and the other batches are unchanged.
- Tests: 4 new tests, and `test_queue_growth_inputs` now expects 192 for the
  constrained case. The new tests check that only that case's context
  changed (the longest request stays 136 tokens). They check that the
  preflight agrees with native `BaseLLMEngine._init_cache`, called on CPU
  with a stub: 12/192 is accepted, 12/256 is rejected by both, and every
  GPU case is accepted. They check that a 12/256 case fails in `main`
  before `engine_args` or `build_engine_class` is called, and they test case
  selection.
- New runner `run_gpu.sh`, in the new evidence directory; the historical
  script is unchanged. It records each required step's Python/timeout,
  tee, artifact, and offline-comparison status in `runner_failures.txt` and
  ends with an explicit `exit` derived from them. A trailing
  `echo`/`tee`/`tail` can no longer mask a failure. That had happened in
  the 20261024T210104Z run: both GPU processes and the offline comparison
  failed, yet `run_gpu_sh_exit=0`.

### Critical review

1. **The context change cannot alter decisions here** (analytical, then
   executed). In the scheduling path, `max_model_len` appears only in
   guards: context ≤ limit in the mapper, LP checks, and executor, and
   decode length < limit in the reference. No request exceeds 136 tokens.
   This holds only for these inputs. Any request longer than 192 tokens
   would now be rejected as a visible failure.
2. **The preflight copies a native rule and could drift from it.** The
   stub test calls the native function itself, so a change to the rule
   would fail the test. The CPU runtime still sets the pool directly
   without native initialization. The preflight guards the GPU driver
   only, which is the path where the failure occurred.
3. **The earlier runner masked failures.** This is fixed for new runs and
   shown by injection. The historical artifacts were left unchanged.

No other issue was found. Review scope: the case definitions, the native
`_init_cache` rule, `max_model_len` uses in the mapper, executor, LP
scheduler, and reference, the driver `main` order, and both runners.

### Results (executed)

**CPU rerun at 192** (`cpu_comparison/`, selected cases
`["queue48_burst_pool12"]`). All three methods passed validation and
drained. Their emitted decisions, decision for decision, and their
same-state records, apart from problem IDs, are identical to the 256-token
CPU runs (step 07):

| Method | Decisions | Preemptions | Replayed tokens | Events / overgeneration |
|---|---|---|---|---|
| reference | 117 | 6 | 102 | 227 / 3 |
| reciprocal-square LP | 118 | 5 | 226 | 227 / 3 |
| reciprocal-square ILP | 117 | 8 | 193 | 225 / 1 |

Coverage was complete for 352 decisions: 235 offline solves and 117 reused
integer results. All were optimal, with no time limit, failure, or missing
record.

**GPU confirmation** (`bash <run>/run_gpu.sh <run>`, two fresh processes,
one attempt each, `timeout 600s`). For each run: profiling reported 15,597
blocks, exactly 12 were initialized natively (`# GPU blocks: 12`; the
preflight required 12), the engine was constructed, and its own Ray shut
down. Both printed `RESULT: PASS workload: drained`, with Python and tee
statuses 0. The only warning was the bfloat16→float16 cast. Engine
initialization took about 76 s and 53 s.

| GPU run | validation_passed | workload_outcome | Decisions | Preemptions | Replayed | Emitted = corrected CPU run |
|---|---|---|---|---|---|---|
| reciprocal-square LP | true | drained | 118 | 5 | 226 | yes, every decision |
| reference | true | drained | 117 | 6 | 102 | yes, every decision |

Offline CPU comparison (`gpu_offline_comparison/`, `RESULT: PASS`, exit
0): every reached GPU decision had a valid, round-trip-identical problem
and one offline solve (118 and 117), all HiGHS-optimal (raw status 0).

- LP: identical actions in 112 decisions; extraction losses at decisions
  0, 8, 13, 20, 24, and 34. These are the same decisions and causes as on
  CPU.
- Reference against the reciprocal-square optimum: 64 identical actions,
  45 strict objective preferences (the smallest gap is 0.0156), and 8
  contract differences (`decode_prerequisite`), all as on CPU.

Central and worker physical block IDs differed after 107 (LP) and 106
(reference) decisions. This is recorded, not required: the inherited
set-order free. Table lengths, free counts, and conservation were checked
equal at every decision.

**Runner.** `runner_failures=0` and `run_gpu_sh_exit=0`, consistent with
the per-step statuses above. With injected fake children, on CPU, the same
script exited 1 when the first GPU process failed and when the offline
comparison failed, and 0 when all succeeded (step 08).

**Checks** (`CUDA_VISIBLE_DEVICES=""`,
`bash <run>/cpu_checks/run_checks.sh <run>`; every step exited 0):

| Step | Result |
|---|---|
| 01 `py_compile` (3 changed files), 02/03 `--help` under `-W error` | pass |
| 04 new and updated tests | 5 passed |
| 05 `tests.test_compare_scheduler_gpu` | 48 passed (44 earlier, one expectation updated, plus 4 new) |
| 06 corrected CPU rerun | `RESULT: PASS` |
| 07 comparison with the 256-token CPU runs | identical |
| 08 runner failure propagation | as expected in all three scenarios |
| 09 whitespace, 10 `git diff --check` | pass |

Not run: the other suites (their code is unchanged and not imported
differently), the other five queue-growth cases, any GPU ILP, and any
broader GPU matrix.

**Established:** for this one bounded workload and its two methods, native
GPU execution produced the CPU-predicted decisions and passed every
per-decision driver check, with complete same-state ILP coverage.
**Not established:** cache contents, attention correctness, quality,
physical-ID equality, recovery, fairness, latency, throughput, or
completion of other workloads.
