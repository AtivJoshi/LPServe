# LPServe LP Inference Program Handoff

## Objective

Add a small reusable program that runs one to three text prompts through the
registered `LPScheduler`. It reuses LPServe's engine setup, request
submission, step loop, native completion, and `RequestOutput` fields. It does
not change the scheduler's mathematics, supported actions, or any production
code.

## Changed paths

- `scripts/run_lp_inference.py` (new): the program and its reusable
  functions.
- `tests/test_run_lp_inference.py` (new): focused CPU tests that use a small
  fake engine. `tests/` matches an ignore rule, so add this file with
  `git add -f` if it is committed.
- `docs/handoffs/lp_inference_handoff.md` (this file).
- `validation_output/lp_inference/20261006T064204Z/` (evidence, including the
  validation driver `validate_run.py`).

No other file was changed.

## Behavior

```bash
python -B scripts/run_lp_inference.py \
    --model-path <existing local snapshot> \
    --prompts-file <JSON file> \
    --output-dir <new results directory>
```

The repository root must be on `PYTHONPATH` so the Ray worker can import the
LP modules. If it is not, the program fails visibly before it starts the
engine.

- **Input.** A JSON array of 1–3 strings. Each string must contain at least
  one non-whitespace character and is used exactly as written: no chat
  template and no trimming. The program rejects a malformed file, a non-array,
  the wrong number of prompts, non-string elements, and empty or
  whitespace-only prompts with exit 2 before it starts the engine. It also
  exits 2 if the output directory already holds `results.json` or
  `summary.json`. Files are opened in exclusive-create mode, so an earlier run
  is never overwritten.
- **Tokenization.** After the engine starts, the program encodes every prompt
  with `engine.tokenizer.encode`, the same call that `add_request` makes. It
  checks that each prompt has at least one token and that prompt tokens + 4
  fit within 32, before the first request is added. Nothing is truncated or
  adjusted. Special tokens such as BOS (the beginning-of-sequence marker) are
  allowed. A failed check exits 1 and writes a failure summary.
- **Run.** The program adds every request before the first `step()`. It
  identifies each request's sequence ID from the new entry in the engine's
  sequence map, and checks that the engine's stored prompt token IDs equal
  the validated IDs. It then steps the engine until no requests remain.
  Each finished output is collected once. An unknown or duplicate finished
  ID, a prompt-text mismatch, or a missing output raises an error. Results are
  returned in input order, whatever the completion order. The step guard is
  total prompt tokens + 4 × number of prompts (39 for the GPU workload).
- **Failure.** Any exception, including `LPSchedulingError` and unsupported
  selected preemption, ends the run. There is no retry, fallback, or engine
  reuse. `summary.json` records the failure type, message, traceback, and LP
  stage, category, and reason when present. It lists the requests that
  finished before the failure as diagnostics only. `results.json` is not
  written. Exit status is 1.
- **Cleanup.** The program shuts down Ray only if Ray was not initialized
  before this invocation.
- **Outputs.** `results.json` has one entry per input: `index`, `seq_id`,
  `prompt`, `prompt_token_ids`, `generated_text`, `generated_token_ids`, and
  `finish_reason`. `summary.json` records success, the command, the resolved
  settings, the input-file SHA-256, the git HEAD, status, and diff hash, code
  hashes, the environment, the engine model facts, the sequence-ID map, the
  step count and guard, Ray cleanup, and any failure.
- **Import safety.** Importing the module only defines constants and
  functions. Reusable functions: `read_prompts`, `encode_prompts`,
  `create_engine`, `make_sampling_params`, `run_requests`, `write_json`, and
  `main(argv=None)`.

**Fixed provisional configuration.** These are scoped inputs, not research
policy. They are listed in `--help` and saved under `settings`. Real
TinyLlama-1.1B-Chat weights (`load_format="auto"`), float16, flash_attention,
TP=PP=1, seed 42, model length 32, block size 16, GPU memory utilization 0.5,
tokenizer mode `auto`, and `trust_remote_code=True`, with model and tokenizer
from the same snapshot. LP settings: resident limit 3, `b_max=96`, `c_max=8`,
`s_max=2`, reserve 1, `conservative_one_block_v1`, utilities 1/1/1, and
`lp_relaxation_mvp_v1` (1e-7, 1e-6, 1e-9, 1e-9). Sampling: greedy,
`max_tokens=4`, `ignore_eos=True`, no stop strings. Metrics: enabled mode
with every optional output off, never plotted.

## Environment

- Host `gpu049`, SLURM job `65305994` (`gpu-preempt`), one NVIDIA A16
  (driver 595.91.07), with no other process using the GPU (0 MiB before the
  run).
- Modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, and `CUDA/12.1.1`
  (full `LOADEDMODULES` in `cpu_checks/environment.txt`).
  `env/bin/python` 3.10.8. torch 2.3.0+cu121, transformers 4.57.6,
  tokenizers 0.22.2, and ray 2.58.0.
- Every execution shell loaded the modules, ran `source env/bin/activate`,
  and exported `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`,
  `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
- Tested code: HEAD `82c019a626dad5faade1f6c8f50071678792a4eb` (verified
  equal to the expected baseline) with a clean tracked tree (diff hash = the
  empty-input SHA-256), plus the new untracked files. SHA-256 hashes:
  `scripts/run_lp_inference.py` `3a175845…7080` and
  `tests/test_run_lp_inference.py` `b3ffea07…bf6819`. Full hashes are in
  `cpu_checks/8-recheck.log` and `run/summary.json`.

## Model assets

Snapshot `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @
`fe8a4ea1ffedaf415f4da2f062534de366a451e6` (`refs/main` is the same),
resolved from the current `HF_HUB_CACHE`. `asset_availability.log` records
the following:

- `model.safetensors` is 2,200,119,864 bytes, and its full SHA-256 is
  `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`, which
  matches the record.
- The existing `inspect_snapshot` check passed: safetensors format, no
  index file, no `np/` directory.

Nothing was downloaded. The engine logged `Casting torch.bfloat16 to
torch.float16`.

## Commands and results

The evidence directory is `validation_output/lp_inference/20261006T064204Z/`
(`$EV` below).

| Check | Observed result |
|---|---|
| `python -B -m unittest discover -s tests -p 'test_run_lp_inference.py' -v` | Ran 17, OK, exit 0 (rerun after a docstring-only edit: Ran 17, OK) |
| `… -p 'test_lp_relaxation_scheduler.py' -v` | Ran 5, OK, exit 0 |
| `… -p 'test_lpserve_state_mapping.py' -v` | Ran 7, OK, exit 0 |
| `… -p 'test_lpserve_plan_execution.py' -v` | Ran 9, OK, exit 0 |
| `… -p 'test_lp_scheduler.py' -v` | Ran 14, OK, exit 0 |
| `python -B -m py_compile` on the script, the test, and the driver | exit 0 |
| `git diff --check` | exit 0, no output |
| `rg -n -i 'phase'` on the new Python files | exit 1, no matches |
| `--help`; no arguments | exit 0; usage error, exit 2 |
| Prompt tokenization (snapshot tokenizer, same arguments as the engine) | each prompt is 9 tokens (BOS + 8); 9 + 4 ≤ 32 |
| GPU: `timeout 300s python -B $EV/validate_run.py --model-path <snapshot> --prompts-file $EV/prompts.json --output-dir $EV/run --driver-dir $EV` | timeout/Python exit **0**; program `SUCCESS`; `DRIVER RESULT: PASS` |

The CPU tests cover input acceptance and rejection, the token bounds,
out-of-order completion with input-order results, all requests being added
before the first step, mid-run failure propagation with no further steps,
duplicate and unknown finished IDs, the step guard, an encoding mismatch, the
complete `main` path (results, summary, and refusal of a second run into the
same directory), a failure summary without results, input rejection before
the engine starts, and import safety in a subprocess. The fake engine checks
program behavior only and says nothing about scheduler correctness.

**GPU invocation.** The validation driver calls
`run_lp_inference.main(["--model-path", …, "--prompts-file", …,
"--output-dir", "$EV/run"])` unchanged. It wraps `create_engine` to install
read-only observers using the existing `Observer`, `capture_state`,
`outputs_with_counts`, and `worker_record` helpers. The direct
`scripts/run_lp_inference.py` invocation was not run separately, by design.
The exact command is in `command.txt`.

**Shell-wrapper error, then one real attempt.** The first wrapper never
invoked Python. A preceding `ls -d /tmp/ray` in an `&&` chain exited 2, so
the command was skipped. No Python output, run directory, engine, or Ray was
created. Its files are kept unchanged in `not_invoked_attempt/` with a
`NOTE.txt`. The GPU check then ran exactly once.

## Observed GPU run

The engine profiled N = 15601 blocks (watermark 156). Inputs 0, 1, and 2 got
sequence IDs 0, 1, and 2. The decisions were recorded, not prescribed:

| Step | Emitted `(seq, chunk)` | Omitted | Free blocks after | Finished |
|---|---|---|---|---|
| 1 | `(1,8)`, `(2,8)` | 0 (waiting) | 15599 | |
| 2 | `(0,8)`, `(2,1)` | 1 (resident) | 15598 | |
| 3 | `(0,1)`, `(1,1)` | 2 (resident) | 15598 | |
| 4–6 | `(1,0)`, `(2,0)` | 0 (resident) | 15598 | |
| 7 | `(1,0)`, `(2,0)` | 0 (resident) | 15600 | 1, 2 |
| 8–10 | `(0,0)` | — | 15600 | |
| 11 | `(0,0)` | — | 15601 | 0 |

Remaining work went 39 → 23 → 14 → 12 → 10 → 8 → 6 → 4 → 3 → 2 → 1 → 0 over
11 steps (guard 39). Requests 1 and 2 finished before request 0, so the run
exercised input-order collection with a different completion order.

The driver's per-step checks passed for every step:

- The boundary was quiescent with exact ownership.
- The call order was mapper → solver → executor → `schedule` → one
  `execute_model`.
- Mapping and solving succeeded, and the snapshot, problem, and plan
  identities matched.
- The mapped limits, policy, utilities, and free blocks matched.
- The emitted actions equalled the plan, prefills first and in `order_key`
  order.
- At most two actions per step, with unique IDs.
- Chunks stayed within min(remainder, 8).
- Decodes were only for prompt-complete residents.
- No preemption or ignore IDs appeared.
- Sampler output IDs matched the emitted order.
- Admission allocated exactly the logical blocks, decodes needed no
  appends, and completion freed the finished requests' blocks.
- Each prefill advanced exactly its chunk, and each decode appended one
  token.
- Unselected requests were unchanged.
- The returned outputs matched the sequence state.

After the run:

- The registered scheduler was `LPScheduler`.
- The engine facts matched every configured value, including
  `load_format="auto"`, float16, and config commit hash = revision.
- Each request's `SamplingParams` was GREEDY, temperature 0, `max_tokens=4`,
  `ignore_eos=True`, and `stop=[]`.
- `summary.settings` equalled the program's settings.
- Each request finished exactly once with four tokens and `length`.
- Every `results.json` entry matched its native finished output and its
  add-time sequence: sequence ID, prompt text, prompt IDs, generated text,
  and token IDs.
- Waiting and running were empty, as were the central block tables and the
  sequence map; the unfinished count was 0, and the free blocks were back to
  N.
- The summary records `ray.started_by_this_run=true` and
  `initialized_after_shutdown=false`. No Ray processes ran before or after
  the run, and GPU memory was 0 MiB both times. Ray created `/tmp/ray`
  during the run, and it was left in place.

**Generated output (observed; quality not assessed).** All three prompts
produced token IDs `[2, 29871, 13, 29966]`, with saved text `". \n<"`. The
first generated token is EOS (ID 2), which continues only because
`ignore_eos=True`. One plausible explanation, not verified: without a chat
template, the chat-tuned model ends its turn and then starts the next
template marker (`<`). This run therefore shows correct mechanics and
association, not useful answers. The leading `.` in the saved text comes
from the framework's native detokenization and was saved as returned.

## Artifacts

In `validation_output/lp_inference/20261006T064204Z/`:

- `prompts.json` (SHA-256 `19135ce0…5079`) and `prompt_tokenization.log`.
- `asset_availability.log`.
- `cpu_checks/`: `environment.txt` and logs 1–8, each with its exit status.
- `command.txt`, `console.log` (with `exit_status`), and
  `ray_state_before.txt` / `ray_state_after.txt`.
- `run/results.json` and `run/summary.json`: the program's own output.
- `validate_run.py`, `driver_summary.json`, and `decision_trace.json`:
  validation machinery and its per-step observations.
- `not_invoked_attempt/`: the skipped wrapper, described above.

## Limitations

- One workload: three 9-token prompts and four greedy tokens each. This does
  not establish general scheduler correctness, fairness, arbitrary-workload
  completion, sampler correctness (§16.4), generation quality, or
  performance. GPU-worker block tables were not instrumented.
- Not exercised: preemption (unsupported, and it would fail visibly), ignore
  controls, pipeline execution, memory pressure, and prompts that cross a
  block boundary during decode.
- The configuration is fixed and provisional. `EngineArgs`, benchmark
  integration, chat templates, and configurable settings are deferred.
- Token validation runs after engine start (about 50 s in this run), so an over-long
  prompt costs one engine start before failing.

OPEN decisions: none resolved; all values are scoped provisional inputs. No
documented contract changed.
