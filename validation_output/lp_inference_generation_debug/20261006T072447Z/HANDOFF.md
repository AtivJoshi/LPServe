# LP inference generation diagnosis (EOS-first output)

Diagnosis only. No implementation, test, configuration, design, or
historical file was changed. Everything below is in this directory.

## Question

Does LPServe's registered `VLLMScheduler`, serving the recorded request
`"Explain why plants need sunlight."` alone, produce the same generated IDs
as the recorded three-request LP run
(`validation_output/lp_inference/20261006T064204Z`), namely
`[2, 29871, 13, 29966]`?

## Answers

1. **Yes, the reference reproduces the recorded IDs exactly.** It generated
   `[2, 29871, 13, 29966]`, which equals the recorded LP result. There is no
   first differing index. The text output also matches: `". \n<"`.
2. **The LP-alone case was not run**, by design, because the reference IDs
   match.
3. **Yes, the saved text is explained by observed decoding behavior.** The
   leading `.` in `". \n<"` is the prompt's last token, not generated output.
   Replaying the native `detokenize_incrementally` (with
   `skip_special_tokens=True`, as `EngineSequenceManager._decode_seq` calls it)
   over the same IDs reproduces the native text exactly. This was checked on
   CPU before the GPU run and again inside the run (`replay_equals_native:
   true`). On the first generated token, the function converts the last six
   IDs. Five are prompt tokens (`▁plants ▁need ▁sun light .`) and the sixth is
   EOS, which is dropped because it is a special token. The offsets assume
   that the last remaining token is new, so `.` is emitted as new text. The
   remaining tokens are `▁` (shown as a space), `<0x0A>` (a newline), and `<`.
   For comparison:
   - `tokenizer.decode(ids)` = `"</s> \n<"`.
   - `tokenizer.decode(ids, skip_special_tokens=True)` = `"\n<"`.
   This is inherited framework detokenization behavior. It is a difference in
   the text, not in the generated tokens.
4. See "Verified versus hypothesis" below.

## Reference case (one GPU execution)

- **Command** (fully resolved in `reference/command.txt`):
  `timeout 300s python -B $LP_GENERATION_DEBUG/debug_run.py --scheduler vllm --model-path <snapshot> --recorded-results validation_output/lp_inference/20261006T064204Z/run/results.json --output-dir $LP_GENERATION_DEBUG/reference`.
  Exit status 0. Driver result `COMPLETED`.
- **Engine.** `VLLMScheduler` with `max_num_seqs=1`,
  `max_num_batched_tokens=32`, and one pipeline stage. TinyLlama real weights
  (`load_format=auto`), float16 (the log shows `Casting torch.bfloat16 to
  torch.float16`), flash_attention, seed 42, TP=PP=1, model length 32, block
  size 16, GPU memory utilization 0.5, tokenizer mode `auto`,
  `trust_remote_code=True`, and config commit hash = revision. Metrics were
  in enabled mode with optional outputs off. The pool had 15612 blocks
  (watermark 156).
- **Submission.** Native `add_request(text, params,
  prompt_token_ids=<recorded IDs>)`. Native `encode` of the text equals the
  recorded nine IDs. Sampling: GREEDY, temperature 0, `max_tokens=4`,
  `ignore_eos=True`, `stop=[]`.
- **Observed trace** (`reference/trace.json`). "Sampler" is the token sampled
  for that step's row.

  | Step | Emitted | Sampler | Output after the step | Free blocks |
  |---|---|---|---|---|
  | 1 | `(0, 9)` | 2 | `[]` | 15612 → 15611 |
  | 2 | `(0, 0)` | 2 | `[2]` | 15611 |
  | 3 | `(0, 0)` | 29871 | `[2, 29871]` | 15611 |
  | 4 | `(0, 0)` | 13 | `[2, 29871, 13]` | 15611 |
  | 5 | `(0, 0)` | 29966 | `[2, 29871, 13, 29966]` | 15611 → 15612 |

  This was the expected shape: one nine-token prefill, then four decodes. No
  ignored or preempted IDs appeared. Each step called `schedule`, then
  `execute_model`. The request finished once with `finish_reason=length`. At
  the end, waiting, running, central block tables, and the engine sequence
  map were empty, the unfinished count was 0, and free blocks were back to
  15612. Ray was started by this run and shut down afterward
  (`initialized_after_shutdown=false`). No `raylet` or `gcs_server` processes
  existed before or after, and GPU memory showed 0 MiB before and after.
- **EOS.** `eos_token_id=2` (`</s>`). The first generated token is EOS
  (position 0). Normal stopping was deliberately disabled by
  `ignore_eos=True`.

Note on `summary.json` field names: `comparisons.recorded_lp_three_request`
reuses the existing `compare_tokens` helper. In that record,
`reference_token_ids` holds the **recorded LP** IDs and `lp_token_ids` holds
**this reference run's** IDs.

## Verified versus hypothesis

**Verified (observed in this task):**

- For this request alone, the reference scheduler and the recorded
  three-request LP run produce the same four generated IDs and the same
  native text.
- The prefill step's sampled token (2, which is discarded because the
  framework appends no token at prompt completion) equals the first decode's
  token.
- The leading `.` in the saved text comes from the native incremental
  detokenizer. It was reproduced exactly by replay.
- The tokenizer's chat template (CPU inspection,
  `cpu_checks/tokenizer_inspection.txt`) formats a user turn as
  `<|user|>\n{content}</s>\n<|assistant|>\n`.

**Not established:**

- Generation correctness. A match with the reference does not prove the
  output is right. Both paths share the model runner, attention, sampler,
  and detokenizer, so a shared framework defect is not excluded.
- Why the model selects EOS first. **Hypothesis (inspection only):** the
  chat-tuned model, given raw text without the template, treats the input as
  a complete user turn. It predicts `</s>`, then a newline, and then begins
  the template marker `<` (as in `<|assistant|>`). The generated
  `</s> ▁ \n <` sequence matches the template's `</s>\n<|...` structure, but
  this run does not test that explanation.
- **Inspection only:** the first decode re-feeds the last prompt token at
  position 8 with 8 cached tokens (`model_runner._prepare_inputs`,
  `flash_attention_wrapper` `decode_cache_len = context_len - 1`). That
  appears to be equivalent to the discarded prefill prediction, which is
  consistent with both being token 2.

**Smallest justified next step:** if the EOS-first behavior matters,
separately authorize one reference-only run of the same request formatted
with the snapshot's chat template, keeping everything else fixed. If that
yields ordinary text, the template hypothesis is supported. Whether
`run_lp_inference.py` should apply a template, and whether the
detokenizer's prompt-tail leak should be documented as inherited behavior
(design §16), are project decisions and were not made here.

## Environment and provenance

- Host `gpu049`, SLURM job `65305994` (`gpu-preempt`, 8 CPUs, 16 GB). One
  NVIDIA A16 (UUID `GPU-07228a4a-…-ed30e030c387`, driver 595.91.07), 0 MiB
  used beforehand.
- Modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, and `CUDA/12.1.1`. Full
  `LOADEDMODULES` is in `cpu_checks/environment.txt`. Interpreter
  `env/bin/python` 3.10.8. torch 2.3.0+cu121, transformers 4.57.6,
  tokenizers 0.22.2, ray 2.58.0, huggingface-hub 0.36.2, vllm-flash-attn
  2.5.9.
- Every execution shell loaded the modules, activated `env`, and set
  `PYTHONPATH=$PWD:…`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
- Tested code: HEAD `c7ebebdac637f54e823bb955ef27f14ef8d0de21` (the expected
  baseline). The tracked tree was clean: `git diff HEAD` SHA-256 is the
  empty-input hash `e3b0c442…b855`. The only untracked files are in this
  directory. Driver `debug_run.py` SHA-256 is
  `786d0bfaf3317f84269b1791713c64aef242bf9fea75dd4ac855307538a27ca6`.
  Hashes of the relevant engine, scheduler, sequence-manager, detokenizer,
  model-runner, attention, sampler, and LP sources are in
  `reference/summary.json` under `provenance.file_sha256`.
- Asset: snapshot `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea1…a451e6`
  (`refs/main` is the same), from the local Hugging Face cache.
  `model.safetensors` is 2,200,119,864 bytes, and its SHA-256
  `6e6001da…14933` equals the record (`cpu_checks/asset_identity.txt`, and
  rechecked by the driver). Nothing was downloaded.
- Original inputs: `prompts.json` SHA-256 `19135ce0…5079`; `run/results.json`
  SHA-256 `8be14149…f4a2`.

## Checks run

| Check | Result |
|---|---|
| `python -B -m py_compile debug_run.py` | exit 0. It wrote `__pycache__/debug_run.cpython-310.pyc` here, because py_compile writes bytecode despite `-B`. |
| `--help`; no arguments; `--scheduler x` | exit 0; exit 2; exit 2 |
| Import in a fresh interpreter | no torch, Ray, or engine modules loaded |
| `load_recorded` on the original results; loading the recorded `validate_run.check_step` | passed |
| CPU detokenizer replay on recorded IDs | `'. \n<'`, equal to the saved text |
| `rg -n -i phase debug_run.py` | exit 1, no matches |
| GPU reference case (one attempt) | exit 0, completed, IDs equal |
| GPU LP-alone case | **not run**: the reference matched |

Unchanged CPU suites were not rerun because no scheduler code changed. The
LP-mode branch of the driver (which reuses the recorded run's
`validate_run.check_step`) was never executed. Only its loading was checked.

## Files

- `debug_run.py`: the driver.
- `cpu_checks/`: `environment.txt`, `asset_identity.txt`,
  `tokenizer_inspection.txt`, `driver_checks.txt`. In
  `environment.txt`, the `pgrep` lines match only the shell's own command
  line. The Ray checks that count are the exact-name checks in
  `reference/command.txt`.
- `reference/`: `command.txt` (with the exit status and the Ray and GPU
  state before and after), `console.log`, `summary.json`, `trace.json`.

OPEN decisions: none affected. No documented contract changed.
