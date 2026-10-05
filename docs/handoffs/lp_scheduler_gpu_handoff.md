# LPServe `LPScheduler` Single-Request GPU Check Handoff

Evidence record for the smallest GPU correctness check of the live
`LPScheduler`. One request runs through a real `BaseLLMEngine`, the registered
`LPScheduler`, the real mapper, LP solve/extraction, and native executor, then
one Ray GPU worker (FlashAttention, KV cache, sampler), and native completion.
Weights are dummy TinyLlama weights. This is **not** full native-execution
acceptance. See "Limits" below.

## Provenance

| Item | Value |
|---|---|
| Repository | `/home/atjoshi_umass_edu/LPServe`, branch `main` |
| Tested revision | Base `a3be58a4e632da79edcc4b3b5d70b0471ed013d9` (the latest Mac-inspected revision; tree was clean at task start) plus one uncommitted new file, `scripts/check_lp_scheduler_gpu.py`, with sha256 `8d1423df0ad3052b0474679bac3c09014693331b44a53dca160d509613cf8591`. The script records `git status --short`, which shows only that file. This handoff was written after the run. |
| Included accepted work | Executor `0afba0d`, failure-interface correction `f053afd`, live integration `276e5f4`, docs through `a3be58a`. All are ancestors of HEAD. No newer commits exist. |
| Host / allocation | `gpu048`, SLURM job `65250285`, partition `gpu-preempt`, `CUDA_VISIBLE_DEVICES=0` |
| GPU | NVIDIA A16, UUID `GPU-f9f58869-c10d-bf00-4312-5c55680c7447`, 15356 MiB, compute capability 8.6, driver 595.91.07 |
| Toolchain | Modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1` (nvcc 12.1.105), loaded one at a time, then `source env/bin/activate`, all in the same shell as each command |
| Interpreter | `/home/atjoshi_umass_edu/LPServe/env/bin/python`, Python 3.10.8 |
| Dependencies | torch 2.3.0+cu121 (CUDA 12.1), transformers 4.57.6, ray 2.58.0, numpy 2.2.6, scipy 1.15.3, flashinfer 0.2.0.post1+cu121torch2.3, vllm-flash-attn 2.5.9, sarathi 0.1.7, nvidia-ml-py 13.595.45, huggingface-hub 0.36.2, tokenizers 0.22.2 |
| Model assets | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. The config and tokenizer came from the existing HF cache `~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0`, snapshot `fe8a4ea1ffedaf415f4da2f062534de366a451e6` (`refs/main`; the loaded config also reports this `_commit_hash`). No download or access failure occurred. Weights were not loaded (`load_format=dummy`). |
| Ray before launch | No Ray processes for this user and no `/tmp/ray` directory. The script started its own local instance and called `ray.shutdown()` at exit. No Ray processes remained afterward, so `ray stop` was not needed. |

## Changed paths

- `scripts/check_lp_scheduler_gpu.py` (new): the validation script.
- `docs/handoffs/lp_scheduler_gpu_handoff.md` (new): this file.
- `validation_output/lp_scheduler_gpu/20261005T084934Z/` (new run artifacts).
  `console.log` is matched by the repository's `*.log` ignore rule, so
  `git status` lists only the two JSON files.

No scheduler, mapper, mathematical layer, executor, engine, worker, model
runner, sampler, configuration, registry, or design file was changed.

## Pre-run source confirmations

- **Profiling.** `ModelRunner.profile_num_available_blocks` has no LP branch,
  so `SchedulerType.LP` takes the generic `else` branch. That branch builds
  `max_num_seqs` sequences totaling `max_num_batched_tokens`.
  `LPSchedulerConfig.max_num_batched_tokens` returns `b_max`. With `b_max=32`
  and `max_num_seqs=1`, this gives one 32-token profiling prompt. The script
  asserts `max_num_batched_tokens == 32` on the driver. The worker-side
  profiling batch itself was confirmed from source only, not instrumented.
- **Metrics (issue found).** The existing disabled `MetricsStore` mode
  (`write_metrics=False` or `None`) cannot run `engine.step()`. The engine's
  `_on_step_completed` always calls `record_block_util` and
  `record_active_gpu_seqs`. Neither method has the `@check_enabled` guard, and
  both read `batch_metrics_count_distribution` and
  `completion_metrics_time_series`, which only enabled mode creates in
  `reset()`. The step would raise `AttributeError` after GPU execution and
  sequence replay. As the task allowed, the script uses the existing enabled
  mode instead, as the earlier baseline smoke test did:
  - `write_metrics=True`;
  - wandb, op-level, CPU-op, chrome-trace, and request-output metrics off;
  - `keep_individual_batch_metrics=True`.

  `plot()` is never called, and the configured `metrics_store_unused/`
  directory was never created. This is an inherited framework defect and was
  not repaired.
- **Dtype.** The script uses the benchmark runner's existing convention,
  `dtype="float16"`. The checkpoint config is bfloat16; LPServe logged
  `Casting torch.bfloat16 to torch.float16`. The resolved model dtype is
  `torch.float16`. The recorded `hf_config_torch_dtype` also reads float16,
  because `ModelConfig` overwrites `hf_config.dtype`.
- **Watermark.** The executor's admission gate uses the block manager's
  default watermark: 0.01, so `int(0.01·N)` blocks. The script asserts
  `N − 2 ≥ watermark_blocks + reserve` before adding the request.

## Resolved configuration (provisional test inputs)

The script defines these as visible constants, and `summary.json` records
them.

- **Model:**
  - model/tokenizer `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, tokenizer mode
    `auto`;
  - `load_format=dummy`;
  - `flash_attention`;
  - TP 1, PP 1;
  - `max_model_len=32`;
  - KV block size 16;
  - `gpu_memory_utilization=0.5`;
  - seed 42;
  - `trust_remote_code=True` (benchmark-runner convention);
  - `download_dir=None`, `revision=None`.
- **Request:**
  - one request with prompt token IDs
    `[1000, 1001, …, 1015]` (16 consecutive IDs);
  - each ID is below the vocabulary size of 32000, is not in
    `all_special_ids`, and is not `unk`;
  - tokens: `ied, ER, ▁stat, fig, me, ▁von, ▁inter, roid, ater, ▁their, ▁bet,
    ▁ein, }\, ">, ▁sub, ▁op`;
  - EOS ID 2;
  - sampling: temperature 0 (greedy), `max_tokens=2`, `ignore_eos=True`, no
    stop strings.
- **LP scheduler:**
  - `b_max=32`, `c_max=8`, `s_max=1`, `max_num_seqs=1`;
  - reserve 1 block;
  - `conservative_one_block_v1`;
  - utilities 1/1/1;
  - numerical policy `lp_relaxation_mvp_v1`, tolerances
    `1e-7`/`1e-6`/`1e-9`/`1e-9`.
- **Profiled pool:**
  - `# GPU blocks: 15612`, so N = 15612 free blocks before the request;
  - watermark 0.01, which is 156 blocks;
  - engine construction took 32.3 s (informational only).

## Observation method

The wrappers live only in the script. Each one calls the original exactly
once, returns its result unchanged, and stores a JSON copy of the immutable
record or of selected primitive state. The script wraps:

- `lpserve_state_mapping.map_scheduler_state`;
- `lp_relaxation_scheduler.solve_and_extract`;
- `lpserve_plan_execution.execute_plan`;
- the engine's `scheduler.schedule`, which captures state after native
  mutation and before engine replay;
- `engine._run_workers`, which records the worker method name.

State is also captured before each `engine.step()` and after it returns.
Requests are advanced only through `engine.add_request` and `engine.step()`.

## Commands and observed results

All commands ran on `gpu048` from the repository root, after the module and
environment setup above.

```text
# CPU suites at base a3be58a (before the script existed)
python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v   Ran 5   OK  exit 0
python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v     Ran 7   OK  exit 0
python -B -m unittest discover -s tests -p 'test_lpserve_plan_execution.py' -v    Ran 9   OK  exit 0
python -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v              Ran 13  OK  exit 0

# Static checks on the final script
python -m py_compile scripts/check_lp_scheduler_gpu.py      exit 0
git diff --check                                            exit 0, no output
rg -n -i 'phase' scripts/check_lp_scheduler_gpu.py          exit 1, no matches

# GPU run (single attempt)
OUT=/home/atjoshi_umass_edu/LPServe/validation_output/lp_scheduler_gpu/20261005T084934Z
mkdir -p "$OUT"
PYTHONPATH="/home/atjoshi_umass_edu/LPServe:/modules/uri_apps/software/Python/3.10.8-GCCcore-12.2.0/easybuild/python" \
  timeout 300s /home/atjoshi_umass_edu/LPServe/env/bin/python -B scripts/check_lp_scheduler_gpu.py \
  --output-dir "$OUT" > >(tee "$OUT/console.log") 2>&1
# $? (the timeout/python status, not tee's):                  0
# printed: RESULT: PASS
```

The `PYTHONPATH` line above is the expanded value of
`"$PWD${PYTHONPATH:+:$PYTHONPATH}"`. The second entry comes from the Python
module. Only one GPU run was made. There were no retries and no relaxed
assertions.

## Observed decisions and asserted state

Every nonempty decision passed these assertions:

- The call order was exactly mapper → solver → executor → `schedule` returns
  → one `execute_model` worker call.
- The mapping succeeded (`StateSnapshot`), with `b_max`/`c_max`/`s_max`,
  reserve, resident limit, policy ID, numerical policy, free blocks, the
  single request, and uniform utilities as configured.
- `SchedulingSuccess` was returned, with solver category `optimal_candidate`,
  raw status 0, zero fractional requests, and no dominant or safety
  preemption IDs.
- The plan's decision matched the emitted `(seq_id, chunk)`.
- The executor output was identical to the `schedule()` output.
- No ignored or preempted IDs were emitted.
- The output ID equals the snapshot iteration, which equals the previous
  iteration + 1.
- Running batches were 0 before scheduling, 1 after scheduling, and 0 after
  completion.
- Prompt progress and generated count were unchanged between scheduling and
  completion.

| # | Decision | Plan (prefill, decode, preempt) | Emitted | After scheduling (physical / free) | After `step()` |
|---|---|---|---|---|---|
| 0 | admission (waiting → running) | (8, 0, 0), objective 8 | `[(0, 8)]` | 1 / N−1 = 15611 | PAUSED, prompt 8/16, 0 generated, 1 logical block, free 15611, output `[]` unfinished |
| 1 | resident prefill | (8, 0, 0), objective 8 | `[(0, 8)]` | 1 / 15611 (no allocation) | PAUSED, prompt 16/16 complete, 0 generated, free 15611, output `[]` unfinished |
| 2 | decode, gap 0 | (0, 1, 0), objective 1 | `[(0, 0)]` | 1 / 15611 (no allocation) | PAUSED, 1 generated, 2 logical / 1 physical, free 15611 |
| 3 | decode, gap 1 | (0, 1, 0), objective 1 | `[(0, 0)]` | 2 / N−2 = 15610 (one append) | FINISHED_LENGTH_CAPPED, 2 generated, finish reason `length` |

The emitted chunk 0 encodes a decode. The mapper reported:

- `m_free` = 15612, then 15611 for each later decision;
- `w` = 1;
- prefill fixed charge 1 at admission;
- decode charge 1 under `conservative_one_block_v1`.

As expected, the sample from the prompt-completing prefill (decision 1) was
discarded natively.

**Drained state** (asserted after decision 3):

- exactly one finished request output, with token IDs `[10935, 15928]`;
- no unfinished requests;
- empty `waiting`, `running`, and central block tables;
- free blocks restored to N = 15612;
- the request removed from the engine sequence manager.

The detokenized text `'osten invoke'` comes from dummy weights and is not
evidence of generation quality.

**Idle call:**

- `engine.step()` returned `[]`;
- the only observed call was `schedule`, which emitted output ID 4 with
  nothing scheduled;
- no mapper, solver, executor, or `_run_workers` call occurred, so no model
  forward ran;
- the iteration advanced from 3 to 4;
- every other captured field was unchanged.

## Artifacts

`validation_output/lp_scheduler_gpu/20261005T084934Z/`:

| File | Content | sha256 |
|---|---|---|
| `console.log` | Combined stdout/stderr plus the appended exit-status line | `2127b2bc4288e167e9e35c0811603d5e055bef116364247647758db864812211` |
| `decision_trace.json` | Per-step before, events (snapshot, LP result, plan, outputs, post-scheduling state), after, request outputs; idle record | `c4b2213438e31c96fcf48803280b00182586cfdbf2bfca3eb12c4bed4c9ed145` |
| `summary.json` | Environment, resolved configuration, N, final output, emitted decisions, pass flag | `ea779ffeeb26c942399848397931205aa80a3a664a6d61246bd8f1beaaa4c548` |

## Check status

- **Passed:**
  - the four CPU suites;
  - `py_compile`;
  - `git diff --check`;
  - the phase-term scan;
  - the GPU run: every schedule and state assertion, exit status 0.
- **Failed:** none.
- **Skipped:** none.
- **Not run:**
  - the `ray stop` cleanup (nothing remained);
  - independent GPU-worker block-table comparison (out of scope);
  - reference numerical comparison;
  - benchmarks;
  - `EngineArgs`/CLI construction.

## Issues and limits

- **Inherited defect, not repaired:** disabled `MetricsStore` mode cannot be
  used with `BaseLLMEngine.step()` (see above). If this matters later, the
  smallest correction would be to guard `record_block_util` and
  `record_active_gpu_seqs` with `@check_enabled`. That is a production change
  and is not made here.
- **Evidence boundaries:**
  - dummy weights only;
  - one request;
  - no mixed batch;
  - `s_max=1`;
  - no preemption.
- **What the run does not establish:**
  - equality of central and GPU-worker block tables (that remains limited to
    the existing CPU evidence);
  - numerical equivalence with any reference;
  - performance or timing;
  - semantic generation quality;
  - memory-profile adequacy for other workloads. The 32-token profile covers
    only this bounded run.
- The Ray worker runs in a separate process. It imports the root-level LP
  modules to unpickle `LPSchedulerConfig.numerical_policy`, so the repository
  root must be on `PYTHONPATH`, not only on the driver's `sys.path`. The
  script fails visibly when it is missing.
- **OPEN decisions and contracts:** none affected. All values above are
  scoped provisional test inputs, and no design or normative document was
  changed.
- **Not committed.** The script, artifacts, and this handoff are left
  uncommitted for review.
