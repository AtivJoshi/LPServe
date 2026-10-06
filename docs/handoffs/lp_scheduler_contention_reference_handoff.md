# LPServe `LPScheduler` Three-Request Real-Weight Contention Comparison Handoff

## Objective

Check one small workload with real TinyLlama weights. Requests A, B, and C
were each run alone through LPServe's registered `VLLMScheduler` to get
reference results. Then all three were submitted together to the registered
`LPScheduler`, which can schedule at most two actions per step
(`s_max=2`). Each request's four generated token IDs from the LP run were
compared with its reference.

**Result:** all four runs passed, all five contention witnesses occurred, and
all three token comparisons were exactly equal.

This is evidence for this workload only. It is **not** evidence of general
fairness, arbitrary-workload correctness, general numerical equivalence,
generation quality, performance, or full native-execution acceptance. Both
paths share the model and sampler code, so agreement cannot rule out a defect
in that shared code.

## Changed paths

- `scripts/check_lp_scheduler_contention_reference_gpu.py` (new). It imports
  read-only helpers from `check_lp_scheduler_gpu.py`,
  `check_lp_scheduler_reference_gpu.py` (snapshot inspection, single-request
  state capture and step checks, token comparison), and
  `check_lp_scheduler_contention_gpu.py` (multi-request state capture,
  per-step contention checks, remaining work). It calls no `main` and changes
  no helper constants. Before running, it checks that the helper constants it
  relies on equal its own values.
- `docs/handoffs/lp_scheduler_contention_reference_handoff.md` (this file).
- `docs/handoffs/lp_scheduler_reference_handoff.md`: one dated correction
  appended to its Environment section. Its CPU environment record lists job
  `65298930`, while both of its GPU summaries record `65301682`; both were
  verified before writing. The original wording is kept.
- `validation_output/lp_scheduler_contention_reference/20261006T054522Z/`
  (evidence).

No production code, existing test or validation script, configuration
interface, dependency, design document, guide, status record, or other
historical evidence was changed.

## Environment

- Host `gpu051`, SLURM job `65301682` (`gpu-preempt`), one NVIDIA A16
  (driver 595.91.07), with torch 2.3.0+cu121 (CUDA 12.1). The same values were
  recorded for the CPU gates and all four GPU runs.
- Modules: `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, and `CUDA/12.1.1`.
  Interpreter: `env/bin/python` 3.10.8. Each execution shell loaded the
  modules, ran `source env/bin/activate`, and exported
  `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`, `HF_HUB_OFFLINE=1`, and
  `TRANSFORMERS_OFFLINE=1`.
- Tested code: HEAD `e04ba4a535dc6f89b3cb418b3bbd3dc7f5ca1ba6`, with a clean
  tracked tree (the `git diff HEAD` hash is the empty-input SHA-256) plus the
  new untracked script. The script SHA-256 was `2fdfb0e6…7743`. The script,
  helper, and source hashes are identical in all four summaries under
  `provenance`.

## Model assets

The existing snapshot `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @
`fe8a4ea1ffedaf415f4da2f062534de366a451e6` (`refs/main` is the same) was used
as both model and tokenizer in all four runs, with `load_format="auto"`.
Before any engine was started, `asset_availability.log` confirmed the
following:

- `model.safetensors` is 2,200,119,864 bytes, and its full SHA-256 is
  `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`. Both
  match the earlier record.
- The format is safetensors with a header-consistent size, there is no index
  file, and there is no npcache `np/` directory.
- The config, tokenizer, and file hashes are listed in each summary's
  `asset`.

Nothing was downloaded. Each run logged `Casting torch.bfloat16 to
torch.float16` (the stored weights are BF16). After engine start, all four
runs recorded identical model and tokenizer facts: `torch.float16`, vocabulary
size 32000, `LlamaTokenizerFast`, EOS 2, and config commit hash equal to the
revision.

## Inputs (scoped provisional test inputs)

- Shared by all four runs: float16 with the `flash_attention` backend; seed
  42; tensor and pipeline parallelism 1; `max_model_len=32`; KV block size 16;
  GPU memory utilization 0.5; tokenizer mode `auto`; and
  `trust_remote_code=True`. Sampling was greedy (temperature 0),
  `max_tokens=4`, `ignore_eos=True`, no stop strings. Metrics mode was enabled
  with all optional outputs off and no plotting.
- Prompts: A = `1000..1015`, B = `2000..2015`, C = `3000..3015`. Each run
  checked that every ID is a valid ordinary vocabulary token and that the
  three lists are disjoint. The decoded tokens are in `shared_after_init`.
- **Reference:** `VLLMSchedulerConfig` with `max_num_seqs=1`,
  `max_num_batched_tokens=32`, and one pipeline stage.
- **LP:** `max_num_seqs=3`, `b_max=96`, `c_max=8`, `s_max=2`, reserve 1,
  `conservative_one_block_v1`, utilities 1/1/1, and `lp_relaxation_mvp_v1`
  (tolerances 1e-7, 1e-6, 1e-9, 1e-9).
- Profiling (source-confirmed, not observed on the worker): both schedulers
  take the generic branch of `ModelRunner.profile_num_available_blocks`. For
  LP this gives `96 // 3` = three 32-token prompts; for the reference, one
  32-token prompt.

## Commands and results

| Check | Observed result |
|---|---|
| `python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v` | Ran 5, OK, exit 0 |
| `… -p 'test_lpserve_state_mapping.py' -v` | Ran 7, OK, exit 0 |
| `… -p 'test_lpserve_plan_execution.py' -v` | Ran 9, OK, exit 0 |
| `… -p 'test_lp_scheduler.py' -v` | Ran 14, OK, exit 0 |
| `python -m py_compile scripts/check_lp_scheduler_contention_reference_gpu.py` | exit 0 |
| `git diff --check` | exit 0, no output |
| `rg -n -i 'phase' scripts/check_lp_scheduler_contention_reference_gpu.py` | exit 1, no matches |
| `--help`; `--scheduler lp` without `--references-dir` | exit 0; usage error, exit 2 |
| Reference A: `timeout 300s python -B scripts/check_lp_scheduler_contention_reference_gpu.py --scheduler vllm --request A --model-path "$LP_REAL_SNAPSHOT" --output-dir "$LP_MULTI_REFERENCE_OUT/reference/A"` | timeout/Python exit **0**, `RESULT: PASS` |
| Reference B (same, `--request B`, `reference/B`) | exit **0**, `RESULT: PASS` |
| Reference C (same, `--request C`, `reference/C`) | exit **0**, `RESULT: PASS` |
| LP: `timeout 300s python -B scripts/check_lp_scheduler_contention_reference_gpu.py --scheduler lp --model-path "$LP_REAL_SNAPSHOT" --references-dir "$LP_MULTI_REFERENCE_OUT/reference" --output-dir "$LP_MULTI_REFERENCE_OUT/lp"` | exit **0**, `EXECUTION: PASS`, `COMPARISON: PASS`, `RESULT: PASS` |

The fully resolved commands are in each case's `command.txt`. Each case ran
exactly once, in its own Python process, with no retries.

Ray was checked before and after each case (`ray_state_*.txt`). Before each
case there were no Ray processes and GPU memory was 0 MiB. Each run shut down
the Ray runtime it had started (`ray_initialized_after_shutdown: false`).
Afterward there were again no Ray processes and GPU memory was 0 MiB. A
`/tmp/ray` session directory that existed before this task was not touched.

## Reference runs

Each reference profiled N = 15612 blocks (watermark 156), and its request got
`seq_id` 0. Each run matched the expected schedule exactly:

1. A 16-token admission prefill (+1 block). Its sample is discarded.
2. A decode with no allocation.
3. A decode that appends one block.
4. A decode.
5. A final decode, after which the request finishes and its blocks are freed
   back to N.

Each step passed the reference script's checks: call order, ownership,
allocation/append/free deltas, block-ID preservation, the native gates,
progress, and request outputs. Each run produced one finished output with
four tokens and finish reason `length`. Afterward the queues, sequence map,
and block tables were empty and N blocks were free. One idle call ran
(iteration 4 → 5, only `schedule`, no state change).

## LP run

The LP run first loaded the three reference summaries. For each one it
checked the label, the exact prompt IDs, that the run had passed with no
failure, the HEAD, diff, script, helper, and source hashes, the assets, and
the shared settings. After engine start it checked that the model and
tokenizer facts matched as well. The SHA-256 of each summary it used:

- A: `c3d35f28601d8a0c48589dbbf7f8dfeb30691fe6aa4dff0bdb46f758806dd8eb`
- B: `71a53d629ef1f3c02c3df5b28141a4542b00c53ce5afcc7bc426ee8c45084ab8`
- C: `56764555a1dc57adef4c0686b58e71240ef491e3b488398096735f24b931215d`

The LP engine profiled N = 15601 blocks (watermark 156). Before the
requests were added, the run checked that `N − 6 − 2 ≥ 156 + 1`. The requests
were added in the order A, B, C and got IDs 0, 1, 2.

| Step | Emitted `(seq, chunk)` | Omitted | Note |
|---|---|---|---|
| 1 | `(1,8)`, `(2,8)` | A (waiting) | B and C admitted |
| 2 | `(1,8)`, `(2,8)` | A (waiting) | B and C prompts complete |
| 3 | `(0,8)`, `(2,0)` | B (resident) | A admitted; C decodes |
| 4 | `(0,8)`, `(2,0)` | B (resident) | A prompt complete; C appends a block |
| 5 | `(1,0)`, `(2,0)` | **A** (resident) | all three prompt-complete and allocated |
| 6 | `(1,0)`, `(2,0)` | **A** (resident) | C finishes and its blocks are freed |
| 7 | `(0,0)`, `(1,0)` | — | **A selected again** |
| 8 | `(0,0)`, `(1,0)` | — | B finishes |
| 9 | `(0,0)` | — | |
| 10 | `(0,0)` | — | A finishes; free blocks back to N |

Remaining work went 60 → 44 → 28 → 19 → 10 → 8 → 6 → 4 → 2 → 1 → 0, within
the 60-step guard. This is the same decision sequence as the earlier
dummy-weight contention run. The schedule was recorded, not prescribed.

**Witnesses** (from `lp/summary.json` → `witnesses`):

1. A was an eligible waiting request omitted at steps 1–2. It stayed waiting
   with no blocks.
2. At steps 5–6, all three requests were prompt-complete, allocated, and
   unfinished, and two of them were selected.
3. At those steps A was the omitted resident. Its status, progress, token
   IDs, logical blocks, and central block IDs did not change.
4. A was selected at step 7 and in every later step.
5. All three requests finished.

Every nonempty step passed all the per-step checks in the imported
contention helper (`check_step`):

- The boundary was quiescent: one pipeline stage and zero running batches.
- Ownership was exact: no request was lost or duplicated.
- The call order was mapper → solver → executor → `schedule` → one
  `execute_model` worker call.
- Mapping and solving succeeded, and the snapshot, problem, and plan
  identities matched.
- The mapped limits, policy, utilities, and free blocks matched the
  configuration and the observed state.
- The complete plan equalled the emitted actions: prefills first, then
  decodes, each group in `order_key` order.
- Selected request IDs were unique; chunks were within their bounds; decodes
  were only for eligible requests; and the action (≤ 2), token, and resident
  limits held.
- No preemption or ignore actions appeared.
- The native output counts matched the emitted metadata.
- Running batches went 0 → 1 → 0 across the step, and scheduling did not
  advance any request's progress.
- Admission and append block counts, and the frees at completion, matched
  the observed actions and block gaps.
- Each prefill advanced exactly its chunk and added no token, and each decode
  appended exactly one token.
- Requests that were not selected kept their ownership, status, progress,
  tokens, and central block IDs.
- Finished requests released their blocks and left all queues and the
  sequence map.
- The returned request outputs matched the sequence state.

**Completion:** each request produced exactly one finished output with four
tokens and finish reason `length`. Afterward the queues, the central sequence
map, and the block tables were empty, and all N blocks were free. One idle
call ran (iteration 9 → 10, only `schedule`, with no mapper, solver,
executor, or worker call and no state change).

## Token comparison

| Request | Reference | LP | Equal | First difference (zero-based) |
|---|---|---|---|---|
| A | `[601, 333, 29899, 6707]` | `[601, 333, 29899, 6707]` | true | none |
| B | `[344, 29892, 322, 278]` | `[344, 29892, 322, 278]` | true | none |
| C | `[29889, 13, 13, 1576]` | `[29889, 13, 13, 1576]` | true | none |

Execution passed and the comparison passed. They are recorded separately in
`lp/summary.json` (`execution_passed`, `comparison_passed`, `comparisons`).
Reference A also equals the earlier one-request result, but that is
historical context only, not a check.

## Evidence

Location: `validation_output/lp_scheduler_contention_reference/20261006T054522Z/`

- `cpu_checks/`: `environment.txt` (host, job, interpreter, modules, HEAD,
  status, hashes) and logs 1–6, each with its exit status.
- `asset_availability.log`: snapshot listing, full weight SHA-256, and the
  `inspect_snapshot` record.
- `reference/{A,B,C}/` and `lp/`: `command.txt`, `console.log` (with the
  appended `exit_status`), `decision_trace.json`, `summary.json`, and
  `ray_state_before.txt` / `ray_state_after.txt`.

## Limitations

- One workload: three prompts and four greedy tokens each. The equality
  shown is for mixed prefill/decode batches and chunked prefills on this
  input only.
- Sampler IDs come from metadata and do not independently prove which
  physical hidden-state row produced each token (§16.4). GPU-worker block
  tables were not instrumented. No independent model library was used as a
  reference.
- Not exercised: preemption, ignore controls, pipeline execution, memory
  pressure, and performance. The inherited limitations in the earlier GPU and
  contention handoffs still apply.

OPEN decisions: none resolved. All values are scoped provisional test inputs.
No documented contract changed.
