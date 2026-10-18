# Adapted vLLM reference scheduler — CPU handoff

**Scope.** CPU-only implementation and evidence for the approved adapted
upstream vLLM v0.31.0 reference (`AdaptedVLLMScheduler`, type
`adapted_vllm`), its registry/EngineArgs wiring, focused native-replay and
packing/sampler checks, and one bounded comparison with the uniform and
rank-based LP policies. No GPU, model, or performance experiment was run. The
specification and source provenance are in
[research context §16.1](../LP%20Scheduler%20Research%20Context.md#161-adapted-vllm-reference-specification).

**Evidence:** [`validation_output/adapted_vllm_reference/20261018T051258Z/`](../../validation_output/adapted_vllm_reference/20261018T051258Z/)
(`summary.json`, `run_checks.sh`, numbered command outputs, `trace.json`,
`comparison.txt`, `working_tree.diff`, `sha256sums.txt`). Tested base
`40652511bf4c7687eebaa4276f1751dc13820c6f` plus the working-tree changes in
that diff; host `gpu052`, Slurm job `65820785`, Python 3.10.8 at
`/home/atjoshi_umass_edu/LPServe/env`, numpy 2.2.6, scipy 1.15.3, torch
2.3.0+cu121 (CPU tensors only). Upstream files were read at commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6` (hashes in `summary.json`); vLLM
was not installed.

## Files

| File | Change |
|---|---|
| `sarathi/core/scheduler/adapted_vllm_scheduler.py` | New reference scheduler: proposal, layout validation, native commit |
| `sarathi/config.py` | `SchedulerType.ADAPTED_VLLM = 9`; `AdaptedVLLMSchedulerConfig` |
| `sarathi/core/scheduler/scheduler_registry.py` | Registration |
| `sarathi/core/block_space_manager/block_space_manager_registry.py` | Registration to `VLLMBlockSpaceManager` |
| `sarathi/engine/arg_utils.py` | `scheduler_type="adapted_vllm"` branch; requires `max_num_batched_tokens`, one stage |
| `tests/test_adapted_vllm_scheduler.py` | 19 focused tests (reuses `tests/test_lp_scheduler.py` fixtures unchanged) |
| `docs/LP Scheduler Research Context.md` | §16.1 specification, provenance, boundaries |

`tests/test_lp_scheduler.py` needed no change. `tests/` matches `.gitignore`
pattern `test*`, so the new test file needs `git add -f` when a commit is
authorized.

## Critical review findings

1. **Upstream watermark differs (adaptation, not a bug).** Upstream's default
   watermark is 0.0 and applies only when `running` is non-empty
   (`kv_cache_manager.py` L505–513). The approved rule always applies LPServe's
   `int(0.01 * pool)`. With a pool of at least 100 blocks and nothing running,
   the reference refuses admissions upstream would accept. This follows R05's
   shared-resource rule and is recorded in §16.1.
2. **Two required behaviors are unobservable in reachable states (analytical).**
   A prefill takes `min(remaining, budget)`, so a partial prefill exhausts the
   budget, and B >= `max_num_seqs` is required. Together these mean a
   reachable state has at most one partial prefill, and it sits at the running
   tail. Every resident is served unless the pass stops at self-eviction.
   As a result, the "scan running order, not decodes first" rule and "count
   omitted residents" never change a decision from a fresh start. The tests
   use a hand-built state, made with native operations, to exercise them. A
   consequence: on reachable states, the reference serves requests in the same
   order as a decodes-first scheduler. Differences from local Sarathi come
   from other mechanisms: its chunk size, arrival-time sorting, and append
   gate.
3. **Preempt-only proposals cannot happen in a real engine (analytical).** The
   first running request can only evict itself if its context does not fit
   in the whole pool. Engine initialization rejects any pool smaller than
   `ceil(max_model_len / block_size)` blocks. The reference also fails
   visibly on a decode at `max_model_len`. So the control-only report is
   reached only with the test harness's undersized pools.
4. **`max_model_len` handling is a provisional choice.** Upstream caps tokens at
   `max_model_len` (L685–690), and its stopping rules end requests there.
   LPServe has no such stop, so the reference raises before mutation for a
   decode at `max_model_len` or an arrived request above it. Workloads need
   prompt + `max_tokens` <= `max_model_len`. After a restart, the inherited
   output-counter reset lets a request generate past that sum and can trigger
   this failure. The upstream prefill cap of `max_model_len - computed - 1`
   is not reproduced. No exercised workload reaches it.
5. **Commit order is safe (analytical, checked at runtime; reasoning
   corrected 2026-10-18).** The original text claimed that every free count
   in commit order is at least the policy-order value. That is false:
   `test_zero_demand_decode_with_admission_of_last_free_block` checks the
   zero-demand decode with one block free in policy order, but commit order
   puts the admission first and leaves zero before the decode. The
   conclusion still holds, by this bounded argument:
   - Admissions occur only in decisions without victims. There, policy order
     is the running pass (decodes and resident prefills), then admissions.
     Moving admissions ahead of decodes means each admission's gate
     (`free - charge >= watermark`) sees at least as many free blocks as in
     policy order, so it still passes.
   - The total actual consumption (admission charges plus decode gaps) is
     the same in both orders and fit the free pool in policy order, so after
     the admissions every decode still finds its demand.
   - Each decode needs only its actual gap, 0 or 1; a zero-demand decode
     allocates nothing, even with zero blocks free.
   - Decisions with victims have no admissions. Commit frees every selected
     victim first, so each retained decode sees at least the free count it
     saw in policy order, where some victims were freed only after it.
   - Resident partial prefills allocate nothing.

   Assumptions: one synchronous step with nothing in flight; the worker
   manager starts the decision with the same free count, table lengths, and
   logical blocks as the central one (checked by the replay tests, not
   proven in general); the worker replays preemptions first, then metadata
   in emitted order; and its `can_append_slot()` result is not asserted.
   Layout validation also re-simulates commit order before any mutation, so
   a violation would fail before mutation rather than corrupt state. This is
   a correction to reasoning, not a new runtime result; the evidence below
   is unchanged. Unequal central and worker physical block IDs alone show
   neither corruption nor correct GPU cache contents.

No issue was found in the remaining approved rules. Review scope: the four
pinned upstream files, the listed LPServe callers, and the paths exercised
below. GPU attention kernels, multi-worker execution, and the engine's Ray
path were not exercised.

## Coverage

| Obligation | Location | Check | Result | Limitation |
|---|---|---|---|---|
| **Procedural policy** | | | | |
| Running-order scan; partial ahead of decode | `_propose` | `test_running_order_scan_with_partial_prefill_ahead_of_decode` | pass: P:8, D omitted, order kept | hand-built state (finding 2) |
| One budget, no chunk/action cap | `_propose` | `test_budget_chunking_without_chunk_or_action_cap` | pass: 5–6 actions, chunk 12 | — |
| Tail eviction, multiple victims, no admission after | `_propose`, `_commit` | `test_tail_eviction_multiple_victims_and_no_admission_after` | pass: victims (3,2), waiting [2,3,9] | — |
| Self-eviction keeps accepted work | `_propose` | `test_self_eviction_preserves_accepted_work` | pass | — |
| Resident limit, retained order, tail admission | `_propose` | `test_resident_limit_retained_order_and_tail_admission`, scan test | pass | omitted-resident case only hand-built |
| Arrived work behind future entry | `_validate_state` | `test_arrived_work_behind_future_dated_entry` | pass | — |
| Config, registries, EngineArgs (mocked model) | config, registries, `arg_utils` | `ConfigurationTest`, `EngineArgsTest` | pass | real engine init unexecuted |
| **Native central/worker CPU integration** | | | | |
| Zero-demand decode + last-block admission | `_commit` | `test_zero_demand_decode_with_admission_of_last_free_block` | pass: allocate then append, 0 free both managers | the LP rejects this plan (§21) |
| Zero-demand decode with 0 free; control-only report | `_schedule` | `test_standalone_zero_demand_decode_then_control_only_report` | pass; no mutation | undersized pool (finding 3) |
| Mixed 0/1-demand decodes in service order | `_commit` | `test_mixed_decode_demands_in_service_order` | pass | — |
| Full-context charge, 0 resident-prefill charge, watermark 1, contention | `_propose` | `test_full_context_charge_resident_prefill_and_positive_watermark`, `..._exact_exhaustion` | pass | — |
| Preemption, recomputation, readmission | native `_preempt` and replay | `test_preemption_recomputation_and_readmission` | pass; 4 token events vs `max_tokens` 3 | inherited counter reset |
| Unsupported/malformed state, inadmissible head, idle | `_validate_state`, `_schedule` | `PreMutationFailureTest` | pass; fingerprint unchanged, no native calls | — |
| Per-manager conservation, equal table lengths and free counts | tests | `_assert_conserved` in replay tests | pass | physical IDs recorded, not required equal |
| **Packing and sampler helpers** | | | | |
| Prompt-first packing, positions, flash-attention metadata, pruning, greedy sampling, completion | native `_prepare_inputs`, `FlashAttentionWrapper.begin_forward`, `Sampler` | `test_mixed_batch_rows_and_samples_follow_requests` | pass: each request's own row and token | TP gather stubbed with single-rank identity; greedy only |
| **GPU/model execution** | — | — | **unexecuted** | attention kernels, generation quality, Ray workers, profiling path |

## Commands and results

All run through `bash validation_output/adapted_vllm_reference/20261018T051258Z/run_checks.sh <dir>`
on `gpu052` (CPU only):

| Step | Command | Result |
|---|---|---|
| 01 | `python -B -m py_compile` on changed/new files | exit 0 |
| 02 | `python -B -W error -c` registry import | exit 0 |
| 03 | `python -B -m unittest tests.test_adapted_vllm_scheduler -v` | 19 passed |
| 04 | `compare_policies.py` | exit 0; `trace.json`, `comparison.txt` |
| 05 | `unittest discover -p test_lp_scheduler.py` | 24 passed |
| 06 | `unittest discover -p test_lp_engine_args.py` | 11 passed |
| 07 | `unittest discover -p test_check_lp_scheduler_preemption_gpu.py` | 7 passed |
| 08 | `git diff --check` | exit 0 |

No failures or skips. The other LP suites (mathematical layer, mapper, and
executor) do not import any changed module and were not rerun.

## Bounded comparison

Configuration: A = 5 prompt tokens (arrives at 1.0), B = 8 (arrives at 2.0),
`max_tokens` 2, 8-block pool, block size 4, `max_model_len` 16, native
watermark 0, resident limits 2 and 1, at most 8 decisions, decision time
10 + index, synthetic sampler. Reference: B = 8 with no chunk or action cap.
LP: `b_max` 8, `c_max` 8, `s_max` 1, numerical policy `lp_relaxation_mvp_v1`;
uniform 1/1/1, and rank-based with D = P = G = 1.

| Run | Decisions | First prefill | First decode | Completion | Preemptions | Unfinished |
|---|---|---|---|---|---|---|
| adapted, limit 2 | 4 | A0 B0 | A1 B2 | A2 B3 | 0 | — |
| LP uniform, limit 2 | 6 | A1 B0 | A4 B2 | A5 B3 | 0 | — |
| LP rank-based, limit 2 | 6 | A0 B1 | A2 B4 | A3 B5 | 0 | — |
| adapted, limit 1 | 6 | A0 B3 | A1 B4 | A2 B5 | 0 | — |
| LP uniform, limit 1 | 8 (bound) | A1 B0 | none | none | 7 | A, B |
| LP rank-based, limit 1 | 8 (bound) | A0 B1 | none | none | 7 | A, B |

Generated-token events equal the native output counters in every run. The
LP restarts happened before any output was produced. Every relaxed LP
solution was integral (`fractional_request_count` 0), so integer extraction
never changed a proposal.

First material differences, starting from identical states:

- **Limit 2, decision 0** (nothing running, waiting A, B, 8 free). The
  reference emits A:5 then B:3: FCFS order, with B chunked to the remaining
  budget. LP uniform emits B:8, and LP rank-based emits A:5. Two causes:
  (a) the LP-specific `s_max = 1` excludes any two-action batch; (b) the
  coefficient reward decides which single action wins. Uniform per-token
  reward prefers B's 8 tokens over A's 5. Rank-based gives A 1.0 per token
  and B 0.5, so A wins (5 against 4). An LP run with a larger `s_max` was
  not executed.
- **Limit 1, decision 1** (reference and rank-based LP in the same state:
  A resident with its prompt complete and gap 0, B waiting, 6 free). The
  reference decodes A, because no allocation failure occurs and it never
  preempts otherwise. The rank-based LP preempts A (penalty 1) and admits
  B:8, scoring 0.5 × 8 − 1 = 3 against 1 for A's decode. This is the LP
  proposal itself: its objective allows voluntary preemption whenever the
  net reward is higher. The two runs then diverge completely: the LPs
  alternate restarts with no output within the bound, while the reference
  drains. This is bounded non-completion, not proof of starvation.
  The uniform LP already differs at decision 0, for cause (b).
- **Physical block IDs:** in the rank-based limit-1 run, central and worker
  IDs differed after decisions 1–6, with equal table lengths and free
  counts. This is the inherited set-order free behavior, not corruption.

These are observations on one tiny workload. They do not show policy
equivalence or support a performance or fairness conclusion.

## Inherited qualifications and open items

- Inherited and not repaired: the final-prefill sample is discarded; the
  output counter resets on recomputation (overgeneration); stopping
  semantics; mixed-sampling association (only greedy is in scope); central
  and worker physical IDs can differ.
- Unexecuted: real engine initialization through `EngineArgs`, the GPU
  profiling path (`ModelRunner.profile_num_available_blocks` uses its
  generic branch for this type), GPU attention kernels, generation quality,
  and Ray workers.
- Q11 remains OPEN. No recovery or overhead criterion was selected. No LP
  mathematics, normative design, or execution contract changed. The
  `max_model_len` failure rule is a provisional choice recorded in §16.1.

## Correction and GPU-driver preparation (2026-10-18)

**Scope.** Two narrow production corrections, the corrected finding 5
reasoning above, and a prepared but **unexecuted** GPU validation driver.
Reviewed base `c6c89bc3c4f95114aecfd27bdbbc53e5e3374ec5`; tested as that base
plus uncommitted changes. No model, CUDA, or GPU path was run.

**Evidence:** [`validation_output/adapted_vllm_reference/20261018T055527Z/`](../../validation_output/adapted_vllm_reference/20261018T055527Z/)
(`summary.json`, `run_checks.sh`, numbered outputs, `working_tree.diff`,
`sha256sums.txt`, `scripts/empty_prompt_at_base.py`). Host `gpu049`, Slurm
job `65821002`, Python 3.10.8 at `/home/atjoshi_umass_edu/LPServe/env`,
numpy 2.2.6, scipy 1.15.3, torch 2.3.0+cu121 (CPU only). The earlier
evidence above does not validate these changes.

| File | Change |
|---|---|
| `sarathi/core/scheduler/adapted_vllm_scheduler.py` | Arrived empty prompt is `unsupported_state` before selection; layout check requires each prefill to have 1..remaining tokens and each decode 1 |
| `sarathi/engine/arg_utils.py` | `adapted_vllm` branch rejects `tensor_parallel_size != 1` before model metadata or `config.yml` |
| `tests/test_adapted_vllm_scheduler.py` | 4 new tests (23 total) |
| `scripts/check_adapted_vllm_gpu.py` | New two-case driver (`--case mixed|preemption`) |
| `tests/test_check_adapted_vllm_gpu.py` | New, 13 CPU tests; needs `git add -f` (`.gitignore` `test*`) |
| `docs/LP Scheduler Research Context.md` | §16.1 input and worker boundaries |

The driver reuses `CacheCapacitySelection`, `check_pool`, `select_prompt_ids`,
and provenance/serialization helpers from the two LP drivers unchanged. It
does not reuse their LP ordering checks or voluntary-preemption workload. It
builds configuration through `EngineArgs(scheduler_type="adapted_vllm")`.

**Defect reproduction (observed).** At the base, an arrived request with
`prompt_token_ids=[]` became `admission` with 0 tokens; `_allocate` ran, the
request moved to `running`, the batch count became 1, and the output was
`(0, 0)`, which downstream code reads as a decode. The new regression fails
against the base scheduler (step 05) and passes with the correction (step 03).

### Coverage

| Area | Check | Result |
|---|---|---|
| **Corrected production behavior** | | |
| Empty arrived prompt rejected; future empty prompt stays ordinary idle | `test_empty_prompt_rejected_before_native_mutation` | pass: no proposal, native call, queue, status, progress, block, worker, or batch change |
| Zero-token prefill rejected by layout check | `test_zero_token_prefill_fails_layout_validation` | pass: `layout_validation`, no mutation |
| TP 2 rejected before model, engine, `config.yml` | `test_missing_budget_and_multi_stage_fail_before_model`, `test_tensor_parallel_rejected_before_engine_startup` | pass |
| One worker accepted | `test_one_tensor_worker_is_accepted`, existing selection test | pass |
| Existing last-free-block, zero/mixed demand, ordering, preemption, packing/sampler tests | 19 existing tests | pass, unchanged |
| **Driver plumbing and planned workloads (CPU)** | | |
| Import and `--help` load no torch, Ray, engine, worker, or `arg_utils` | `DriverImportAndHelpTest` | pass |
| Capacity: one profile, chosen pool through native `_init_cache` (max-context check included), shortfall fails before init, second profile fails, other calls delegate | `CacheCapacityTest` | pass for pools 8 and 4 |
| Native `EngineArgs` configuration per case | `EngineArgsTest` | pass (mocked `ModelConfig`) |
| Mixed: A:5/B:3, then selection A decode, B:5 emitted `[[1,5],[0,0]]`; completion, release, idle | `test_mixed_case_through_native_cpu_replay` | pass: 4 decisions |
| Preemption: fill pool, zero-demand decodes, victims 3 then 2, readmission of 17-token contexts, 4 token events vs `max_tokens` 3 for 2 and 3, idle | `test_preemption_case_through_native_cpu_replay` | pass: 8 decisions |
| Checks discriminate (wrong layout, missing victims, decision bound) | `test_checks_reject_a_wrong_expectation`, `test_decision_bound_stops_an_unplanned_run` | pass |
| Evidence serialization; failure recorded without GPU; no overwrite | `EvidenceTest` | pass |
| **Prepared, unexecuted** | | |
| Real engine/Ray worker construction, profiling, cache initialization, flash-attention kernels, dummy-weight forward and greedy sampling, the worker class's by-value serialization into Ray | driver GPU path | **unexecuted** |

The CPU replay uses the planned block size 16, pools, budgets, model lengths,
arrivals (`time.perf_counter()`, as `add_request`), and output limits, and the
real scheduler, central manager, engine sequence manager, and a separate
worker sequence manager. Sampled tokens are synthetic (constant); no token
ID is prescribed. Neither planned workload needed a correction.

### Commands and results

`bash validation_output/adapted_vllm_reference/20261018T055527Z/run_checks.sh <dir>`
on `gpu049` (CPU only):

| Step | Command | Result |
|---|---|---|
| 01 | `python -B -m py_compile` on changed/new files | exit 0 |
| 02 | `python -B -W error -c` driver and registry import | exit 0 |
| 03 | `python -B -m unittest tests.test_adapted_vllm_scheduler -v` | 23 passed |
| 04 | `python -B -m unittest tests.test_check_adapted_vllm_gpu -v` | 13 passed |
| 05 | `empty_prompt_at_base.py c6c89bc` | new regression fails at base, as expected |
| 06 | `unittest discover -p test_lp_engine_args.py` | 11 passed |
| 07 | `unittest discover -p test_check_lp_scheduler_preemption_gpu.py` | 7 passed |
| 08 | `git diff --check` | exit 0 |

No failures or skips. `test_lp_scheduler.py` (fixtures used, file unchanged)
and the LP mathematical, mapper, and executor suites import no changed LP
code and were not rerun.

### Future GPU commands (UNEXECUTED)

On an allocated GPU node, after CPU review and separate authorization, one
case per process:

```bash
module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1
source env/bin/activate
MODEL_DIR=/home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
RUN=validation_output/adapted_vllm_gpu/$(date -u +%Y%m%dT%H%M%SZ)
for CASE in mixed preemption; do
  mkdir -p "$RUN/$CASE"
  PYTHONPATH="$PWD" timeout 600s python -B scripts/check_adapted_vllm_gpu.py \
    --case "$CASE" --model-path "$MODEL_DIR" --output-dir "$RUN/$CASE" \
    2>&1 | tee "$RUN/$CASE/console.log"
done
```

Expected for each case: `RESULT: PASS`; profiled capacity recorded and at
least the pool (8 or 4); exactly that pool in the central manager and worker
cache; and every decision matching `CASES[...]["decisions"]`: selection,
victims, stop reasons, emitted order, central and worker block operations
(table lengths), and request state. Each decision must also pass
per-manager conservation, equal central/worker table lengths, free counts,
and sequence state, and sampler outputs in emitted order. Finished outputs
must have `max_tokens` tokens and released blocks, followed by one ordinary
idle call. The planned decision counts are 4 (mixed) and 8 (preemption),
bound 12. Generated-token events must be `{0: 2, 1: 2}` and
`{0: 3, 1: 3, 2: 4, 3: 4}`. The second set shows the inherited counter
reset; it is not cumulative generation. A pass would not establish
physical-ID equality, cache-content correctness, sampler generality, or
generation quality.
