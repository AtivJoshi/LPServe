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
5. **Commit order is safe (analytical, checked at runtime).** Admissions happen
   only when there are no victims. All frees therefore come before
   allocations, and every free count in commit order is at least the
   policy-order value. As a result, no allocation and no worker `can_allocate`
   assertion can fail because of the reordering. Admissions are still checked
   in policy order. Layout validation re-simulates commit order before any
   mutation.

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
