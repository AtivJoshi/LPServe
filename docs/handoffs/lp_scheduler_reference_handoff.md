# LPServe `LPScheduler` One-Request Real-Weight Reference Comparison Handoff

## Objective

Compare the four generated token IDs from one request under two execution
paths. Both paths use real TinyLlama weights:

- **Reference:** LPServe's registered `VLLMScheduler` processes the full
  16-token prompt in one step.
- **LP:** the registered `LPScheduler` processes the prompt in two 8-token
  chunks.

Each path runs in a separate process. Each run also checks its own schedule,
completion, and memory release.

**Result:** both runs passed all of their checks. The generated tokens were
identical: `[601, 333, 29899, 6707]` from both.

This establishes agreement for this one request only. Both paths share the
model and sampler code, so agreement cannot rule out a defect in that shared
code. It is not evidence of general numerical equivalence, sampler
correctness, or generation quality.

## Changed paths

- `scripts/check_lp_scheduler_reference_gpu.py` (new). It imports read-only
  helpers from `scripts/check_lp_scheduler_gpu.py` and does not call that
  script's `main`.
- `docs/handoffs/lp_scheduler_reference_handoff.md` (this file).
- `validation_output/lp_scheduler_reference/20261006T051432Z/` (evidence).

No production code, existing test or GPU script, design document, guide,
status record, historical handoff, or artifact was changed.

## Model assets

The first inspection found that the cached snapshot
`fe8a4ea1ffedaf415f4da2f062534de366a451e6` held only the config and tokenizer
files. It had no weight files. This is recorded in `asset_availability.log`;
the trailing `AttributeError` there comes from a wrong exception name in an
ad hoc probe command, not from the script. Work stopped before engine
initialization.

With user authorization, `model.safetensors` was downloaded into the existing
cache at that same pinned revision with `hf_hub_download`
(`weight_download.log`). Nothing else was downloaded, and no dependencies or
environment settings were changed.

| Item | Value |
|---|---|
| Repository and revision | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea1ffedaf415f4da2f062534de366a451e6` (`refs/main` is the same) |
| Snapshot | `~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1…`, passed as both `ModelConfig.model` and `ModelConfig.tokenizer` |
| Weights | `model.safetensors`: 2,200,119,864 bytes, SHA-256 `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` (equals the cache blob name), 201 BF16 tensors, header-consistent size; no index file |
| Selected format | safetensors via `load_format="auto"` (`prepare_hf_model_weights` glob of a local directory, no download path); no `np/` npcache directory |
| Config SHA-256 | `486bedda…8fe9`; `config.json` `torch_dtype` is bfloat16 |
| Tokenizer | `LlamaTokenizerFast`, length 32000, EOS 2; file hashes are in `summary.json` → `asset.small_files` |

Engine output confirmed `load_format=auto` and `dtype=torch.float16`. It also
logged `Casting torch.bfloat16 to torch.float16`: the stored BF16 weights are
cast to the specified float16 in both runs. Both runs ran with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Shared configuration (scoped provisional inputs)

- float16 with the `flash_attention` backend; seed 42; tensor and pipeline
  parallelism 1.
- `max_model_len=32`, KV block size 16, GPU memory utilization 0.5,
  tokenizer mode `auto`, and `trust_remote_code=True`.
- Prompt IDs `1000..1015`. Each ID was validated as an ordinary vocabulary
  token: not special and not unknown. The decoded tokens are recorded in
  `summary.json`.
- Greedy sampling (temperature 0), `max_tokens=4`, `ignore_eos=True`, no stop
  strings.
- Enabled metrics mode with all optional outputs off and no plotting.
- **Reference scheduler:** `max_num_seqs=1`, `max_num_batched_tokens=32`, one
  pipeline stage.
- **LP scheduler:** `max_num_seqs=1`, `b_max=32`, `c_max=8`, `s_max=1`,
  reserve 1, `conservative_one_block_v1`, utilities 1/1/1, and numerical
  policy `lp_relaxation_mvp_v1` (tolerances 1e-7, 1e-6, 1e-9, 1e-9).

Before starting its engine, the LP run checked that the reference summary had
passed. It also checked that the reference matched its own HEAD, diff hash,
script, helper, and source hashes, assets, and shared inputs. After engine
start, it checked that the tokenizer and model facts matched too. It recorded
the reference summary path and SHA-256
`9dd786f7389d76ddae4831a79e868fbec1dc81162b5f5b5b933014887917e7da`.

## Environment

- Host `gpu051`, SLURM job `65298930`, one NVIDIA A16. The exact GPU, driver,
  and CUDA details are in each `summary.json` under `environment`.
- Modules: `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, and `CUDA/12.1.1`.
- Interpreter: `env/bin/python` (3.10.8), with `PYTHONPATH` set to the
  repository root.
- Tested base: HEAD `e3e5647800cbe54d22bc1bccd73c9e8e7dd21546` with a clean
  tracked tree plus the new untracked script. The script and source hashes are
  in `cpu_checks/environment.txt` and in each summary's `provenance`.
- The hashes were rechecked after the download and matched the gated code
  (`asset_availability_after_download.log`).

## Commands and results

Each execution shell loaded the modules, ran `source env/bin/activate`, and
exported `PYTHONPATH="$PWD"`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

| Command | Result |
|---|---|
| `python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v` | Ran 5, OK, exit 0 |
| `… -p 'test_lpserve_state_mapping.py' -v` | Ran 7, OK, exit 0 |
| `… -p 'test_lpserve_plan_execution.py' -v` | Ran 9, OK, exit 0 |
| `… -p 'test_lp_scheduler.py' -v` | Ran 14, OK, exit 0 |
| `python -m py_compile scripts/check_lp_scheduler_reference_gpu.py` | exit 0 |
| `git diff --check` | exit 0, no output |
| `rg -n -i 'phase' scripts/check_lp_scheduler_reference_gpu.py` | exit 1, no matches |
| `timeout 300s python -B scripts/check_lp_scheduler_reference_gpu.py --scheduler vllm --model-path "$LP_REAL_SNAPSHOT" --output-dir "$LP_REFERENCE_OUT/reference"` | timeout/Python exit **0**, `RESULT: PASS` |
| `timeout 300s python -B scripts/check_lp_scheduler_reference_gpu.py --scheduler lp --model-path "$LP_REAL_SNAPSHOT" --reference-summary "$LP_REFERENCE_OUT/reference/summary.json" --output-dir "$LP_REFERENCE_OUT/lp"` | timeout/Python exit **0**, `EXECUTION: PASS`, `COMPARISON: PASS`, `RESULT: PASS` |

The fully resolved commands are in `reference/command.txt` and
`lp/command.txt`. Each case ran exactly once, with no retries.

Ray was checked before and after each run (`ray_state_*.txt`):

- No Ray processes were running before either run.
- Each run shut down the Ray runtime it had started
  (`ray_initialized_after_shutdown: false`).
- No Ray processes remained afterward, and GPU memory returned to 0 MiB.
- The `/tmp/ray` session directory left by the reference run was not deleted.

## Observed schedules and memory

Both runs profiled the same pool: N = 15612 blocks, with a watermark of 156
blocks. The pool sizes did not need to match, but they did. The assigned
request ID was 0 in both runs. The emitted actions are `(seq, chunk)` pairs,
where chunk 0 means a decode.

| Step | Reference emitted | Blocks after scheduling, free after completion | LP emitted | Blocks after scheduling, free after completion |
|---|---|---|---|---|
| 1 | `(0,16)` admission | +1, free N−1 | `(0,8)` admission | +1, free N−1 |
| 2 | `(0,0)` decode, gap 0 | +0, N−1 | `(0,8)` resident prefill | +0, N−1 |
| 3 | `(0,0)` decode | +1 append, N−2 | `(0,0)` decode, gap 0 | +0, N−1 |
| 4 | `(0,0)` decode | +0, N−2 | `(0,0)` decode | +1 append, N−2 |
| 5 | `(0,0)` final decode | +0; finished, freed to N | `(0,0)` decode | +0, N−2 |
| 6 | — | — | `(0,0)` final decode | +0; finished, freed to N |

Both runs matched their expected schedules exactly.

Every nonempty step passed these checks:

- It started from a single-stage boundary with zero running batches.
- The call order was exactly `schedule` → one `execute_model` for the
  reference, and mapper → solver → executor → `schedule` → one
  `execute_model` for LP.
- No ignored or preempted IDs were emitted.
- There was one running batch after scheduling and zero after completion.
- Scheduling changed only the allocation; prompt progress, tokens, and
  logical blocks were unchanged.
- Existing central block IDs were preserved, and the number of appended
  blocks equalled the logical/physical gap.
- The native admission and append gates held.
- Each prefill advanced by exactly its chunk and added no token.
- Each decode appended exactly one token.
- Ownership of the request (queues, block tables, sequence map) stayed
  consistent.
- The returned `RequestOutput` token IDs matched the sequence state.

For LP, each step also showed:

- successful mapping and solving;
- mapped limits, policy, utilities, and free blocks matching the
  configuration and the observed state;
- matching snapshot, problem, and plan identities;
- a plan decision equal to the emitted action;
- no preemption selected;
- action, chunk, token, and resident limits respected.

At completion, each run produced exactly one finished `RequestOutput` with
four tokens and finish reason `length`. The queues, the central sequence map,
and the block tables were empty, and all N blocks were free again.

The idle call returned `[]` and invoked only `schedule`. For LP, it made no
mapper, solver, executor, or worker call. The iteration advanced once (4 → 5
for the reference, 5 → 6 for LP), and no other observed state changed.

## Token comparison

| | Token IDs (generated positions 0–3) |
|---|---|
| Reference | `[601, 333, 29899, 6707]` |
| LP | `[601, 333, 29899, 6707]` |
| Equal | **true**; first difference: none (zero-based indexing) |

## Evidence

All evidence is in `validation_output/lp_scheduler_reference/20261006T051432Z/`:

- `cpu_checks/`: the environment record and the gate logs with their exit
  statuses.
- `asset_availability.log`, `weight_download.log`, and
  `asset_availability_after_download.log`: the asset inspection before and
  after the download, the download itself, and the hash recheck.
- `reference/` and `lp/`: `command.txt`, `console.log` (with the appended
  `exit_status`), `decision_trace.json`, `summary.json`, and
  `ray_state_before.txt` / `ray_state_after.txt`.
- The token comparison is in `lp/summary.json` under `comparison`.

`*.log` files match the repository's ignore rule.

## Limitations

- This is one request with one prompt and four greedy tokens. It is not
  evidence of general reference agreement, sampler or physical-row
  correctness, generation quality, or behavior under memory pressure,
  preemption, ignore controls, or pipeline execution.
- GPU-worker block tables were not instrumented, and no independent model
  library was used as a reference.
- The weights are stored as BF16 and run as float16 in both paths, as
  specified.
- The inherited limitations in the earlier GPU handoffs still apply.

OPEN decisions: none resolved. All values are scoped provisional comparison
inputs. No documented contract changed.
