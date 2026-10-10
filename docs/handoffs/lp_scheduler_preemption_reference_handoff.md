# LPServe Native Preemption Real-Weight Reference Handoff

## Objective

Compare the bounded partial-prefill preemption workload under real
TinyLlama weights with per-request references. Three cases ran in
sequence, each with its own engine:

1. request A alone through the registered `VLLMScheduler`;
2. request B alone through the same scheduler;
3. A and B through the registered `LPScheduler`, using the validated
   preemption workload from
   `docs/handoffs/lp_scheduler_preemption_gpu_handoff.md`.

Each LP request's generated token IDs must equal its own reference. The LP
preemption, replay, recomputation, completion, release, and idle
assertions are checked separately. Execution success and token agreement
are recorded as separate results.

## Added paths

- `scripts/check_lp_scheduler_preemption_reference_gpu.py`: the comparison
  driver, with modes `--scheduler vllm --request A|B` and
  `--scheduler lp --references-dir`.
- `tests/test_check_lp_scheduler_preemption_reference_gpu.py`: 6 CPU tests
  of reference validation and token comparison. `tests/` matches an ignore
  rule, so `git status` shows this file only with `--ignored`.
- `docs/handoffs/lp_scheduler_preemption_reference_handoff.md` (this file).
- `validation_output/lp_scheduler_preemption_reference/20261010T065408Z/`
  (evidence).

No production code, existing script, historical evidence, design document,
audit, or `docs/math/` file was changed.

## Driver design

**Reuse without modification.** The driver imports existing helpers
read-only and calls no imported `main`:

- From `check_lp_scheduler_preemption_gpu`: the four-block
  `CacheCapacitySelection` and engine/worker factories, `central_state`,
  `check_quiescent` (which includes the per-manager pool-conservation
  check), `check_decision`, `check_boundary`, and the expected LP schedule.
- From `check_lp_scheduler_reference_gpu`: `inspect_snapshot`,
  `compare_tokens`, and the streaming `sha256`.
- From `check_lp_scheduler_gpu`: `Observer`, `check`, the environment
  record, and the output records.

The reused functions read some helper constants. `check_helper_constants`
requires each of those constants to equal this case's value; the values
found are recorded in every summary.

The older reference helpers that embed the earlier 16-token prompt, four
output tokens, or 32-token model length were not reused. These are
`shared_inputs`, `validate_prompt`, `load_reference`, and `capture_state`.

**Real-weight evidence.** The engine subclass keeps the four-block cache
sizing. Its worker adds one read-only method that returns loaded parameter
values:

- `model.embed_tokens.weight` and `lm_head.weight`, rows 1000, 1016, 2000,
  and 2047;
- all of `model.norm.weight`.

The driver reads the same values on the CPU from the local
`model.safetensors` and casts them to the engine dtype. Exact equality is
required. Random dummy initialization cannot satisfy this check.

**Reference validation.** Before the LP engine is built, the driver
validates both reference summaries:

- each passed, with the expected scheduler and label and the exact prompt;
- real-weight loading was verified;
- assets, shared inputs, and the four-block capacity are identical;
- the reference scheduler identity is correct;
- the tested-code hashes and environment match this run;
- each has exactly one in-vocabulary integer token and a native `length`
  finish.

After the LP engine starts, the driver also requires the references'
model/tokenizer facts and weight fingerprint to match the LP run.
Comparisons are associated by A/B label and exact prompt, not by raw
sequence ID.

**Import safety.** Importing the driver loads no torch, Ray, safetensors,
engine, or worker module; a test checks this.

## Inputs

- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0, local snapshot
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6` (equal to `refs/main` and to
  the config commit hash), used for both model and tokenizer, offline.
  Weights come from `model.safetensors`: 2,200,119,864 bytes, 201 tensors,
  bf16, header-complete. The native `load_format="auto"` path loaded them.
- **Execution:** float16, FlashAttention, seed 42, tensor parallel 1, one
  pipeline stage.
- **Cache:** maximum model length 64, block size 16, profiling utilization
  0.5. Every case had 15,606 profiled available blocks and 4 initialized
  blocks.
- **Metrics:** enabled, with every optional output off.
- **Prompts and sampling:** A uses token IDs 1000–1016 (17 tokens) and B
  uses 2000–2047 (48 tokens); both were validated as ordinary vocabulary
  tokens. Sampling is greedy (temperature 0), with `max_tokens=1`,
  `ignore_eos=True`, and no stop strings.
- **Reference scheduler:** `VLLMScheduler` with `max_num_seqs=1`,
  `max_model_len=64`, one stage, and `max_num_batched_tokens=64`.
- **LP scheduler:** resident limit 1, `b_max=64`, `c_max=16`, `s_max=1`,
  reserve 0, `conservative_one_block_v1`; utilities prefill-token 1,
  decode 40, penalty 1; numerical policy `lp_relaxation_mvp_v1`. These are
  provisional scoped inputs, and D-03 remains OPEN.

## Environment

- **Host:** `gpu051.unity.rc.umass.edu`, SLURM job `65495160`
  (`gpu-preempt`), `CUDA_VISIBLE_DEVICES=0`.
- **GPU:** NVIDIA A16 (`GPU-b7192761-b009-1409-dfe0-541ef40add86`), driver
  595.91.07, torch CUDA 12.1.
- **Modules:** `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`.
- **Interpreter:** `env/bin/python` 3.10.8.
- **Packages:** torch 2.3.0+cu121, transformers 4.57.6, ray 2.58.0,
  safetensors 0.8.0, scipy 1.15.3, vllm-flash-attn 2.5.9.
- **Revision:** HEAD `a177a3e767e817937a34e4b28bf7838712fecb0f` plus the
  untracked new files, with no tracked modifications.

## Commands and results

Each shell ran `module load Python/3.10.8-GCCcore-12.2.0`,
`module load CUDA/12.1.1`, `source env/bin/activate`, and
`export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`.

**CPU checks** (`cpu_checks/`):

| Suite | Result |
|---|---|
| `test_lp_relaxation_scheduler.py` | 5 OK |
| `test_lpserve_state_mapping.py` | 7 OK |
| `test_lpserve_plan_execution.py` | 13 OK |
| `test_lp_scheduler.py` | 15 OK |
| `test_check_lp_scheduler_preemption_gpu.py` | 7 OK |
| `test_check_lp_scheduler_preemption_reference_gpu.py` | 6 OK |

Every exit status was 0, and nothing was skipped. `7-static.log` records
`py_compile` of the new driver and test (exit 0) and direct scans of both
new files for trailing whitespace and phase terms (no match).

**GPU cases**, run in order with one engine at a time; each `command.txt`
holds the fully resolved command. `<snapshot>` stands for the snapshot
directory above, and `<root>` for the evidence root.

| Case | Command | Result |
|---|---|---|
| Reference A | `timeout 300s python -B scripts/check_lp_scheduler_preemption_reference_gpu.py --scheduler vllm --request A --model-path <snapshot> --output-dir <root>/reference/A` | PASS, exit 0 |
| Reference B | the same with `--request B` and `<root>/reference/B` | PASS, exit 0 |
| LP | `… --scheduler lp --model-path <snapshot> --references-dir <root>/reference --output-dir <root>/lp` | EXECUTION PASS, COMPARISON PASS, exit 0 |

Each case passed on its first attempt; there were no failed or repeated
attempts. The only console warning in each case was
`Casting torch.bfloat16 to torch.float16`, which comes from the requested
dtype.

## Observed results

**Real weights.** In all three cases the fingerprint matched the file
exactly: 8,192 values each for `embed_tokens` and `lm_head`, and 2,048 for
`model.norm`. The loaded dtype was float16 and the file dtype bf16. The
fingerprint hashes were identical across cases.

**Capacity.** In every case:

- central: `CacheConfig`, manager, and allocator each had 4 blocks, all
  free, with watermark 0;
- worker: `CacheConfig` and cache engine had 4 blocks, the 22 layer cache
  tensors had shape `[4, 16, 4, 64]`, and the manager and allocator had
  4 blocks, all free.

**References.**

- *A:* emitted `[0, 17]` (2 blocks; 2 free in each manager), then `[0, 0]`.
  It generated `[403]` with finish reason `length`, then was released
  (4 free in each manager), followed by idle.
- *B:* emitted `[0, 48]` (3 blocks; 1 free in each manager), then `[0, 0]`.
  It generated `[29871]` with finish reason `length`, then was released,
  followed by idle.
- *Both:* each finish step showed one central and one worker `free`, and
  ownership and allocations agreed between the managers. Each final token
  equaled the decode step's sampler output.

**LP execution.** The required schedule was emitted:

| Step | Preempted; scheduled | Central / worker free |
|---|---|---|
| 0 | `[]`; `(0,16)` | 2 / 2 |
| 1 | `[0]`; `(1,16)` | 1 / 1 |
| 2 | `[]`; `(1,16)` | 1 / 1 |
| 3 | `[]`; `(1,16)` | 1 / 1 |
| 4 | `[]`; `(1,0)` B finishes | 4 / 4 |
| 5 | `[]`; `(0,16)` A readmitted | 2 / 2 |
| 6 | `[]`; `(0,1)` | 2 / 2 |
| 7 | `[]`; `(0,0)` A finishes | 4 / 4 |
| idle | iteration 7 → 8; only `schedule` called | 4 / 4 |

At the preemption boundary (step 1):

- *Mapped state:* zero batches in flight at mapping, `m_free=2`, `w=0`,
  legal victim set `{"0"}`. A was running with 16 of 17 tokens processed,
  2 physical blocks, and recovery 2. B was waiting with fixed charge 3.
- *Solution:* relaxed A was x=0, y=0, I=0, z=0.5 and B was x=16, y=0, I=1,
  z=0, with relaxed objective 15.5. The integer plan preempted A and
  prefilled B for 16 tokens (objective 15, A dominant).
- *Central execution:* `free(0, [3, 2])`, then `allocate(1, [2, 3, 1])`. A
  stayed PAUSED with 16 processed until replay.
- *Worker replay:* `free(0, [3, 2])`, then `allocate(1, [2, 3, 1])`.
- *After replay:* both central and worker A were WAITING, with 0 processed,
  prompt incomplete, and no output.

At readmission (step 5), the mapper saw A waiting and unallocated, with a
full-context charge of 2. Central and worker each made one fresh 2-block
allocation, and A recomputed from token zero (16 tokens processed in both
managers). The physical IDs then differed (central `[2, 3]`, worker
`[2, 1]`). This is the inherited set-order free behavior; counts, free
totals, and each manager's own pool conservation were checked instead.

B finished before A.

**Token comparison.**

| Request | Reference | LP | Equal |
|---|---|---|---|
| A | `[403]` | `[403]` | yes |
| B | `[29871]` | `[29871]` | yes |

The validated reference summaries are recorded in `lp/summary.json`, with
their paths and SHA-256 hashes (A `e39f685e4005…`, B `eabdf31b62ea…`).
Discarded prefill samples were not compared. Native text (`"ate"`, `" "`)
is recorded but is not used as evidence.

**Cleanup.** Every case ran `ray.shutdown()`; Ray was initialized before it
and not after. No Ray processes for this user remained, and GPU memory
used was 0 MiB after each case. No metrics output directory was created.

## Evidence

The root is
[`validation_output/lp_scheduler_preemption_reference/20261010T065408Z/`](../../validation_output/lp_scheduler_preemption_reference/20261010T065408Z/):

| Path | Content |
|---|---|
| [`cpu_checks/environment.txt`](../../validation_output/lp_scheduler_preemption_reference/20261010T065408Z/cpu_checks/environment.txt) | Host, job, modules, GPU, HEAD, status (including ignored tests), versions, asset listing, file hashes, and Ray processes |
| `cpu_checks/1-…7-*.log` | Each command, its verbose output, and exit status |
| `reference/A/`, `reference/B/`, `lp/` | Per case: `command.txt` (resolved command, environment, and Ray/GPU state before and after), `console.log` (with exit status), `summary.json`, and `decision_trace.json` |
| [`lp/summary.json`](../../validation_output/lp_scheduler_preemption_reference/20261010T065408Z/lp/summary.json) | Provenance and file hashes, helper constants, environment, assets, shared inputs, scheduler identity, capacity, weight fingerprint, validated references, final outputs, comparisons, separate execution/comparison/overall flags, and cleanup |

## Limitations

This supports only this bounded real-weight partial-prefill preemption case
on one GPU. Exact agreement cannot rule out a defect shared by both
scheduler paths, which use the same model, sampler, and completion code.

It does not establish:

- interruption after generated tokens (the §§16.1–16.2 limitations
  remain);
- corrected generation or output semantics, or useful text;
- correct token-to-text conversion (§16.7);
- arbitrary-workload completion, fairness, or pipeline support;
- independent framework correctness or performance.

The fingerprint compares sampled parameter rows, not every weight.
