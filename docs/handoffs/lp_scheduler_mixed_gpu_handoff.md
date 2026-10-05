# LPServe `LPScheduler` Two-Request Mixed-Batch GPU Check Handoff

Evidence record for a tiny mixed-batch GPU correctness check of the live
`LPScheduler`. Two greedy requests, A then B, run through a real
`BaseLLMEngine`, the registered `LPScheduler`, the real mapper, LP
solve/extraction, and native executor, then one Ray GPU worker
(FlashAttention, KV cache, sampler), and native completion. Weights are dummy
TinyLlama weights. B is admitted only after A's prompt completes, so two
steps carry one prefill chunk (B) and one decode (A) in the same forward pass.
This adds evidence only for that path. It is **not** full native-execution
acceptance. Inherited limitations recorded in
`docs/handoffs/lp_scheduler_gpu_handoff.md` ("Issues and limits") still apply
and are not repeated here.

## Provenance

| Item | Value |
|---|---|
| Repository | `/home/atjoshi_umass_edu/LPServe`, branch `main` |
| Tested revision | Base `f8eec6daa841d3d00099ce698d85a851f7048198` (the latest Mac-inspected revision, which is HEAD; tree was clean at task start) plus one uncommitted new file, `scripts/check_lp_scheduler_mixed_gpu.py`, sha256 `842d3f7266b786f3b667e599daef63bc9424a62611b2e99670270506c24c6394`. The script recorded `git status --short` as only that file. This handoff was written after the run. |
| Reused helper | `scripts/check_lp_scheduler_gpu.py` (committed, unchanged), sha256 `8d1423df0ad3052b0474679bac3c09014693331b44a53dca160d509613cf8591`. Imported read-only for `check`, `to_jsonable`, `outputs_record`, `request_output_record`, `environment_record`, and `Observer`. Its `main` and constants are not used. |
| Host / allocation | `gpu048`, SLURM job `65277251`, partition `gpu-preempt`, `CUDA_VISIBLE_DEVICES=0` |
| GPU | NVIDIA A16, UUID `GPU-44005302-0ffe-f46a-bb29-8b368a365328`, 15356 MiB, compute capability 8.6, driver 595.91.07 |
| Toolchain | Modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1` (nvcc 12.1.105), then `source env/bin/activate`, in the same shell as each command |
| Interpreter | `/home/atjoshi_umass_edu/LPServe/env/bin/python`, Python 3.10.8 |
| Dependencies | torch 2.3.0+cu121 (CUDA 12.1), transformers 4.57.6, ray 2.58.0, numpy 2.2.6, scipy 1.15.3, flashinfer 0.2.0.post1+cu121torch2.3, vllm-flash-attn 2.5.9, sarathi 0.1.7, nvidia-ml-py 13.595.45, huggingface-hub 0.36.2, tokenizers 0.22.2 |
| Model assets | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` config and tokenizer from the existing HF cache, snapshot `fe8a4ea1ffedaf415f4da2f062534de366a451e6` (`refs/main`, equal to the loaded config's `_commit_hash`). No download or access failure. Weights not loaded (`load_format=dummy`). |
| Ray | Before launch: no Ray processes for this user, no `/tmp/ray`. The script started its own local instance and called `ray.shutdown()`. Afterward: no Ray processes remained. The run left a `/tmp/ray` session directory; it was not deleted. |

## Changed paths

- `scripts/check_lp_scheduler_mixed_gpu.py` (new)
- `docs/handoffs/lp_scheduler_mixed_gpu_handoff.md` (new, this file)
- `validation_output/lp_scheduler_mixed_gpu/20261005T175528Z/` (new).
  `console.log` matches the repository's `*.log` ignore rule, so `git status`
  lists only the two JSON files.

No production code, the single-request script, historical evidence, design,
guide, or status file was changed.

## Pre-run source confirmations

- **Profiling.** `ModelRunner.profile_num_available_blocks` has no LP branch;
  `SchedulerType.LP` takes the generic `else` branch, which builds
  `max_num_seqs` prompts sharing `max_num_batched_tokens`
  (`LPSchedulerConfig.max_num_batched_tokens` returns `b_max`). With
  `b_max=64`, `max_num_seqs=2`, this gives two 32-token profiling prompts.
  The script asserts these inputs on the driver; the worker-side profiling
  batch was confirmed from source only. `b_max=64` is a scoped test input,
  not a policy change. No claim of general memory-profile adequacy.
- **Mixed order.** `lpserve_plan_execution.execute_plan` builds its action list
  sorted by `(0 if prefill else 1, order_key)` (D-21). The mapper's
  `order_key` is `(raw_seq_id,)`.
- **Sampler association.** `Sampler._sample` groups rows by sampling type, and
  `on_step_completed` pairs metadata with sampler outputs by position
  (design §16.4). With both requests greedy there is one contiguous group.
  `SamplerOutput.seq_id` is copied from worker-side metadata, so the recorded
  sampler IDs show the worker's metadata order, not which physical row
  produced each token.

## Resolved configuration (provisional test inputs)

- Model: TinyLlama-1.1B-Chat-v1.0, tokenizer mode `auto`, `load_format=dummy`,
  `dtype=float16` (checkpoint bfloat16 cast to float16, logged), `flash_attention`,
  TP 1, PP 1, `max_model_len=32`, KV block size 16, `gpu_memory_utilization=0.5`,
  seed 42, `trust_remote_code=True`, `download_dir=None`, `revision=None`.
- Request A prompt IDs `[1000, 1001, …, 1015]`; request B prompt IDs
  `[2000, 2001, …, 2015]` (16 consecutive IDs each, disjoint). Each ID is
  below the vocabulary size 32000, is not a special ID, and is not `unk`.
  Tokens are recorded in `summary.json`. EOS ID 2.
- Sampling, identical for both: temperature 0 (`GREEDY`), `max_tokens=2`,
  `ignore_eos=True`, no stop strings.
- LP scheduler: `b_max=64`, `c_max=8`, `s_max=2`, `max_num_seqs=2`, reserve 1
  block, `conservative_one_block_v1`, utilities 1/1/1, numerical policy
  `lp_relaxation_mvp_v1` with tolerances `1e-7`/`1e-6`/`1e-9`/`1e-9`.
- Metrics: enabled mode, all optional outputs off, never plotted;
  `metrics_store_unused/` was not created.
- Profiled pool: `# GPU blocks: 15606`, so **N = 15606**; watermark 0.01 =
  156 blocks. The script asserted `N − 3 ≥ 156 + 1` before adding A.
  Engine construction took 44.5 s (informational).

## Observation method

As in the single-request check, local wrappers call the original exactly
once, return the result unchanged, and record JSON copies of immutable
records or primitive state. Wrapped: `map_scheduler_state`,
`solve_and_extract` (also records the input `problem_id`), `execute_plan`
(also records the input `snapshot_id` and result `problem_id`),
`scheduler.schedule` (captures state after native mutation, before
replay), and `engine._run_workers` (method name, plus
`(seq_id, output_token)` for `execute_model`). State is also captured before
and after each `engine.step()` and around each `engine.add_request()`.
Requests advance only through these two engine calls.

## Commands and observed results

All on `gpu048`, repository root, after the module/environment setup above.

```text
# CPU suites at base f8eec6d (before the script existed)
python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v   Ran 5   OK  exit 0
python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v     Ran 7   OK  exit 0
python -B -m unittest discover -s tests -p 'test_lpserve_plan_execution.py' -v    Ran 9   OK  exit 0
python -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v              Ran 13  OK  exit 0

# Static checks on the final script
python -m py_compile scripts/check_lp_scheduler_mixed_gpu.py   exit 0
git diff --check                                               exit 0, no output
rg -n -i 'phase' scripts/check_lp_scheduler_mixed_gpu.py       exit 1, no matches

# GPU run (single attempt)
OUT=/home/atjoshi_umass_edu/LPServe/validation_output/lp_scheduler_mixed_gpu/20261005T175528Z
mkdir -p "$OUT"
PYTHONPATH="/home/atjoshi_umass_edu/LPServe:/modules/uri_apps/software/Python/3.10.8-GCCcore-12.2.0/easybuild/python" \
  timeout 300s /home/atjoshi_umass_edu/LPServe/env/bin/python -B scripts/check_lp_scheduler_mixed_gpu.py \
  --output-dir "$OUT" > >(tee "$OUT/console.log") 2>&1
# $? (the timeout/python status, not tee's):   0
# printed: RESULT: PASS
```

The `PYTHONPATH` value is the expansion of `"$PWD${PYTHONPATH:+:$PYTHONPATH}"`.
One GPU run; no retries, no relaxed assertions.

## Observed decisions and asserted state

Assigned IDs: A = seq 0, B = seq 1 (A has the smaller order key, `(0,)` vs
`(1,)`). Each `add_request` made exactly one `add_seq` worker call, placed the
request in `waiting` with no blocks, and did not change the iteration or free
blocks.

Every nonempty step passed these assertions:

- Call order exactly mapper → solver → executor → `schedule` returns → one
  `execute_model` worker call.
- `StateSnapshot` with configured limits, reserve, resident limit, policy ID,
  numerical policy, free blocks, uniform utilities, and the expected request
  set in order-key order.
- `SchedulingSuccess` (category `optimal_candidate`, raw status 0, zero
  fractional requests, no dominant/safety preemption IDs). Snapshot problem
  ID = solver input = result = plan = executor input; executor snapshot ID =
  mapped snapshot ID.
- Exact per-request plan `(prefill, decode, preempt)`; token ≤ 64, chunk ≤ 8,
  actions ≤ 2, residents ≤ 2.
- Executor output identical to `schedule()` output; exact emitted list; no
  ignored/preempted IDs; output ID = snapshot iteration = previous + 1.
- Sampler output seq IDs in the emitted order.
- Running batches 0 before scheduling, 1 after, 0 after completion.
- Prompt progress and generated counts of both requests unchanged by
  scheduling.
- Exact ownership, central block tables, sequence-manager IDs, unfinished
  count, free blocks, per-request status/progress/logical/physical counts, and
  returned request outputs (order, IDs, token counts, finished flags).

| Step | Plan A / B | Emitted `(seq, chunk)` | After scheduling | After `step()` | Returned outputs |
|---|---|---|---|---|---|
| 1 | A (8,0,0) | `[(0,8)]` | A 1 block; free N−1 = 15605 | A PAUSED, prompt 8/16, 0 gen; free 15605 | A `[]` |
| 2 | A (8,0,0) | `[(0,8)]` | A 1 block; free 15605 | A prompt 16/16, 0 gen; free 15605 | A `[]` |
| — | add B | — | — | waiting `[1]`, running `[0]` | — |
| 3 | A (0,1,0) / B (8,0,0) | `[(1,8),(0,0)]` | B 1, A 1; free N−2 = 15604 | B prompt 8/16; A 1 gen, 2 logical / 1 physical; free 15604 | B `[]`, A `[10935]` |
| 4 | A (0,1,0) / B (8,0,0) | `[(1,8),(0,0)]` | A 2, B 1; free N−3 = 15603 | A FINISHED_LENGTH_CAPPED, freed; B prompt 16/16, 1 block; free N−1 = 15605; running `[1]` | B `[]`, A `[10935, 15928]` finished `length` |
| 5 | B (0,1,0) | `[(1,0)]` | B 1; free 15605 | B 1 gen, 2 logical / 1 physical; free 15605 | B `[3591]` |
| 6 | B (0,1,0) | `[(1,0)]` | B 2; free N−2 = 15604 | B FINISHED_LENGTH_CAPPED; free N = 15606 | B `[3591, 7027]` finished `length` |

Both mixed steps (3, 4) emit B's positive prefill chunk before A's decode,
although A has the smaller ID and order key: prompt-first grouping, not
ID order. Sampler outputs were `[(1,15619),(0,10935)]` and
`[(1,3591),(0,15928)]`; the prefill-completing samples (step 2 for A, step 4
for B) were discarded natively, as expected. Plan objectives: 8, 8, 9, 9, 1,
1. Mapper: `w = 1`; admission fixed charge 1; decode charge 1.

**Drained state after step 6:** exactly two finished outputs, one each for
seq 0 and seq 1, each with two tokens and finish reason `length`; no
unfinished requests; empty `waiting`, `running`, central block tables, and
sequence manager; free blocks N = 15606.

**Idle call:** `engine.step()` returned `[]`; only `schedule` was called (no
mapper, solver, executor, or worker call); output ID 6 with nothing
scheduled; iteration 5 → 6; all other captured state unchanged.

Dummy-weight text (`'osten invoke'`, `'Result Great'`) is not evidence of
generation quality. A's tokens equal the earlier single-request run's, but no
reference comparison was required or is claimed.

## Artifacts

`validation_output/lp_scheduler_mixed_gpu/20261005T175528Z/`:

| File | Content | sha256 |
|---|---|---|
| `console.log` | Combined stdout/stderr plus the appended exit-status line | `74171d2f6ce07e1ec2f40401be0b77b0be19dcb20e535238e779e9947657f771` |
| `decision_trace.json` | Admission records, per-step before/events/after/request outputs, idle record | `f3a2275dc76e4fedd6d3c3bdfa36163210b483fdd05ecba12edb8e7c14876b4f` |
| `summary.json` | Script/helper hashes, environment, configuration, N, IDs, finished outputs, decisions, pass flag | `ec3224f9304826fbc38bfc22393e2983e2e4b0d6e55c5f9628cec25ed87868ba` |

## Check status

- **Passed:** four CPU suites; `py_compile`; `git diff --check`; phase-term
  scan; GPU run (all assertions, exit 0).
- **Failed:** none. **Skipped:** none.
- **Not run:** `ray stop` (nothing remained); `/tmp/ray` removal; independent
  GPU-worker block-table comparison; reference numerical comparison;
  benchmarks.

## Limits

This run does not establish: semantic generation quality; numerical agreement
with any reference; correctness with different sampling methods in one batch
(the §16.4 grouping issue is avoided, not repaired); arbitrary mixed
workloads; central/GPU-worker block-table equality; runtime preemption,
pipeline execution, or performance; or full native-execution acceptance. The
recorded sampler IDs come from worker metadata and do not independently prove
which hidden-state row produced each token.

OPEN decisions and contracts: none affected. All values are scoped
provisional test inputs. Not committed; left for review.
