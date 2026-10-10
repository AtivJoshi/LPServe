# LPServe Decode-Preemption Real-Weight Reference Handoff

## Objective

Validate one bounded case in which the LP scheduler interrupts a request
*after it has generated a token*:

1. A is prefilled and decodes once (one generated token).
2. B arrives; the real mapper, solver, and extraction preempt A to admit B.
3. B finishes.
4. A's expanded context (original prompt plus its first generated token) is
   readmitted, recomputed from token zero, decodes twice more, and finishes.

The case ran first as a live CPU replay test with synthetic tokens, then on
one GPU with real TinyLlama weights. On the GPU, A's cumulative token
history and B's token were compared with uninterrupted runs of each request
under the registered `VLLMScheduler`. Earlier preemption work
(`lp_scheduler_preemption_handoff.md`, `…_preemption_gpu_handoff.md`,
`…_preemption_reference_handoff.md`) interrupted A only during prefill,
before any token was generated.

## Changed paths

- `tests/test_lp_scheduler.py`: new `LiveDecodePreemptionTest`. The harness
  `replay`/`step` methods gained an optional `sample` callback so each
  scheduled entry can get a chosen synthetic token; the default
  (`SAMPLED_TOKEN` for every entry) is unchanged.
- `scripts/check_lp_scheduler_decode_preemption_reference_gpu.py` (new):
  the GPU comparison driver.
- `tests/test_check_lp_scheduler_decode_preemption_reference_gpu.py` (new):
  11 CPU tests of the driver.
- `docs/lp_scheduler_design.md`: §13 native-preemption evidence boundary and
  §15.4 traceability now record this bounded result.
- This handoff and the evidence directory below.

`tests/` matches an ignore rule, so the new test file appears only in
`git status --ignored`. No production code changed: scheduler, mapper,
executor, mathematics, engine, worker, sequence, and block-manager code are
untouched, as are existing drivers, historical evidence, and `docs/math/`.

## Scoped inputs (provisional, validation-only)

| Input | CPU | GPU |
|---|---:|---:|
| Block size | 4 | 16 |
| Blocks per manager | 4 | 4 |
| A / B prompt length | 5 / 12 | 17 / 48 |
| `max_model_len` | 16 | 64 |
| `b_max`, `c_max`, `s_max` | 4, 4, 1 | 64, 16, 1 |
| Resident limit, reserve | 1, 0 | 1, 0 |

Both use `conservative_one_block_v1` and `lp_relaxation_mvp_v1`
(feasibility 1e-7, integrality 1e-6, objective tolerances 1e-9).

**Utilities** (fixed per request): A decode 40, prefill-token 1, penalty 1;
B decode 40, prefill-token 20, penalty 1. `LPSchedulerConfig` requires a
uniform triple, so it holds A's values. A validation-only override of
`_build_utilities` calls the real method (which selects the arrived,
unfinished raw IDs) and gives each of those IDs its fixed triple. The mapper
still checks the exact key set and copies the values. The CPU test patches
the class method inside a context manager; the GPU driver sets an instance
attribute and deletes it afterwards (`utility_override_removed: true`).
D-03 (preemption penalty policy) remains OPEN.

**Sampling:** greedy (temperature 0), `ignore_eos=True`, no stop strings.
LP A `max_tokens=2`, reference A `max_tokens=3`, B `max_tokens=1` in both.
The A difference is deliberate: the inherited reset clears A's output list,
so LP A produces 1 + 2 = 3 tokens in total (§16.1). Reference A's three
uninterrupted tokens cover the same positions. Both drivers record this
explicitly; sampling fields other than `max_tokens` are identical.

**Prompts:** CPU uses the harness lists (A `0..4`, B `0..11`). GPU uses A
IDs 1000–1016 and B IDs 2000–2047, validated against the vocabulary and
special-token IDs. A is raw ID 0 and B is raw ID 1.

## Environment

- Host `gpu048.unity.rc.umass.edu`, SLURM job `65515174` (`gpu-preempt`),
  `CUDA_VISIBLE_DEVICES=0`.
- GPU: NVIDIA A16 (`GPU-8d45767b-01c8-5486-03af-138bbc1c6752`), driver
  595.91.07, 15356 MiB, torch CUDA 12.1.
- Modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1` (full
  list in the evidence). Interpreter `env/bin/python` 3.10.8.
- torch 2.3.0+cu121, transformers 4.57.6, ray 2.58.0, numpy 2.2.6,
  scipy 1.15.3, safetensors 0.8.0, vllm-flash-attn 2.5.9.
- Revision: HEAD `025a9042dba388b175ea8a7b57a720e9fa136d94` on `main` (the
  reviewed baseline; tree clean at start) plus the uncommitted changes
  above. `sha256(git diff HEAD)` at test time was `035ff9a2d972…`. The
  design-document edit and this handoff were written after all runs; the
  tested code-file hashes are unchanged.

## Commands and results

Every execution shell ran `module purge`, the three `module load`s,
`source env/bin/activate`, and `export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`
from the repository root.

**CPU gates** (`python -B -m unittest discover -s tests -p '<file>' -v`):

| Suite | Baseline (unmodified) | After change |
|---|---|---|
| `test_lp_relaxation_scheduler.py` | 5 OK | 5 OK |
| `test_lpserve_state_mapping.py` | 7 OK | 7 OK |
| `test_lpserve_plan_execution.py` | 13 OK | 13 OK |
| `test_lp_scheduler.py` | 15 OK | 16 OK |
| `test_check_lp_scheduler_preemption_gpu.py` | 7 OK | 7 OK |
| `test_check_lp_scheduler_preemption_reference_gpu.py` | 6 OK | 6 OK |
| `test_check_lp_scheduler_decode_preemption_reference_gpu.py` | — | 11 OK |

All exit statuses were 0; nothing was skipped. `cpu_checks/8-static.log`:
`py_compile` of the new and changed Python files (exit 0), `git diff --check`
of the test file (exit 0), trailing-whitespace and phase-term scans (no
match). The first static attempt failed only because my check passed
`cfile=/dev/null`, which `py_compile` refuses; it is kept as
`8-static_attempt1_py_compile_invocation_error.log`.

Before the final runs, a throwaway mutation check (not retained) confirmed
the live test fails if B's prefill utility is set to 1, or if A's synthetic
second token repeats its first.

**CPU trace:** `python -B <root>/trace_live_decode_preemption.py <root>`
runs the same test method once and writes its trace. Exit 0, `PASS`.

**GPU cases**, run in order with one engine each. `<snapshot>` is
`/home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6`
and `<root>` is the evidence root:

| Case | Command | Result |
|---|---|---|
| Reference A | `timeout 300s python -B scripts/check_lp_scheduler_decode_preemption_reference_gpu.py --scheduler vllm --request A --model-path <snapshot> --output-dir <root>/reference/A` | PASS, exit 0 |
| Reference B | the same with `--request B` and `<root>/reference/B` | PASS, exit 0 |
| LP | `… --scheduler lp --model-path <snapshot> --references-dir <root>/reference --output-dir <root>/lp` | EXECUTION PASS, COMPARISON PASS, exit 0 |

Each case passed on its first attempt. The only warning was
`Casting torch.bfloat16 to torch.float16`, from the requested dtype.

## Observed results

### Emitted schedule (both CPU and GPU)

| Step | Preempted; scheduled (CPU) | Scheduled (GPU) |
|---|---|---|
| 0 | `[]`; `(0,4)` | `(0,16)` |
| 1 | `[]`; `(0,1)` | `(0,1)` |
| 2 | `[]`; `(0,0)` A's first token | `(0,0)` |
| 3 | `[0]`; `(1,4)` | `[0]`; `(1,16)` |
| 4–5 | `[]`; `(1,4)` | `(1,16)` |
| 6 | `[]`; `(1,0)` B finishes | `(1,0)` |
| 7 | `[]`; `(0,4)` A readmitted | `(0,16)` |
| 8 | `[]`; `(0,2)` | `(0,2)` |
| 9 | `[]`; `(0,0)` | `(0,0)` |
| 10 | `[]`; `(0,0)` A finishes | `(0,0)` |
| idle | iteration 10 → 11; no mapping or solve | same; only `schedule` called |

Output IDs equaled the step number, and every step had zero batches in
flight at mapping and a pipeline call order of map → solve → execute.

### Before B arrived (after step 2)

A had one generated token, was unfinished and `PAUSED`, held 2 blocks, and
2 blocks were free in each manager (CPU and GPU).

### Preemption boundary (step 3)

- *Mapped state:* legal victims `{"0"}`, `m_free=2`, `w=0`. A was
  `running`/`PAUSED`, prompt complete (5/5 CPU, 17/17 GPU), 2 physical
  blocks, recovery 2, decode-eligible with conservative charge 1. B was
  `waiting`, unallocated, fixed charge 3. On the GPU the native gates read
  B `can_allocate=False` (needs 3, 2 free, watermark 0) and A
  `can_append_slot=True`.
- *Relaxed solution:* A x=0, y=0, I=0, z=0.5; B x=`c_max`, y=0, I=1, z=0.
  Objective 79.5 (CPU) and 319.5 (GPU), as derived by hand from the LP.
- *Integer plan:* preempt A (dominant; no safety victim), prefill B for 4
  (CPU) or 16 (GPU). Objective 79 / 319.
- *Central execution:* `free(A)` of 2 blocks, then `allocate(B)` of 3.
  Before replay, A was in `waiting`, still `PAUSED`, with its original
  prompt and its generated token unchanged. `preempted_seq_ids=[0]`.
- *Replay:* CPU call log was central reset, worker reset, worker `free(0)`,
  worker `allocate(1)`. On the GPU, the central reset changed A from
  prompt `1000…1016` + output `[403]` to prompt `1000…1016, 403` + output
  `[]`, and the worker ops were `free(0, [3, 2])` then
  `allocate(1, [2, 3, 1])`.
- *After replay:* both copies of A were `WAITING`, prompt progress 0,
  prompt incomplete, output empty, and the expanded prompt was the original
  prompt followed by the first token exactly once.

Central and worker physical IDs differed after the free (GPU: central
B `[3, 2, 1]`, worker `[2, 3, 1]`). This is the known inherited set-order
free behavior. Allocated sets, per-request counts, free counts, and each
manager's pool conservation were checked instead, at every step.

### Readmission and completion

At step 7 the mapper saw A `waiting`, unallocated, prompt length 6 (CPU) or
18 (GPU), progress 0, and charge 2; `can_allocate` was true. Central and
worker each allocated 2 blocks, and A recomputed from token zero. B
finished at step 6 and A at step 10, each releasing its blocks centrally
and on the worker. Afterwards queues, sequence maps, and block tables were
empty, with 4 free blocks in each manager.

### Token histories

A's history is taken from the sampler outputs of completed decode entries.
Prefill entries' samples are recorded separately as discarded and never
enter a history.

| | CPU (synthetic) | GPU LP | GPU reference | Equal |
|---|---|---|---|---|
| A, cumulative | `[101, 102, 103]` | `[403, 263, 1016]` | `[403, 263, 1016]` | yes |
| A, first token | `[101]` | `[403]` | `[403]` | yes |
| B | `[201]` | `[29871]` | `[29871]` | yes |

Native final state for A (GPU): prompt IDs `1000…1016, 403`; output IDs
`[263, 1016]`; finish reason `length`. The first token appears only in the
expanded prompt, and three tokens were generated for a 2-token cap. Both
are the inherited §16.1–16.2 behaviors. The CPU case showed the same
per-epoch split (`[101]` before the reset; `[102, 103]` after). Native text
(`"ate a don"`, `" "`) was recorded but is not evidence.

The first-token reference agrees with the earlier one-token reference A
(`[403]`) and B with the earlier reference B (`[29871]`) in
`lp_scheduler_preemption_reference_handoff.md`.

### Capacity, weights, and cleanup (GPU, every case)

- Profiled capacity 15,606 blocks; 4 initialized once. Central
  `CacheConfig`, manager, and allocator: 4 blocks, watermark 0. Worker:
  cache engine 4 blocks, 22 layer tensors of shape `[4, 16, 4, 64]`,
  manager and allocator 4.
- Real-weight fingerprint matched `model.safetensors` exactly (8,192 values
  each for `embed_tokens` and `lm_head`, 2,048 for `model.norm`; bf16 file,
  float16 loaded), with identical fingerprint hashes across cases.
- References were validated before the LP engine was built: label, prompt,
  sampling (A cap 3, B cap 1), assets, shared inputs, capacity, scheduler
  identity, code hashes, and environment.
- Cleanup: `ray.shutdown()` ran (initialized before, not after). No Ray
  processes for this user remained, and GPU memory used was 0 MiB after
  each case. `/tmp/ray` existed after the runs; its state before the first
  run was not recorded. No metrics output directory was created.

## Evidence

Root:
[`validation_output/lp_scheduler_decode_preemption_reference/20261010T175710Z/`](../../validation_output/lp_scheduler_decode_preemption_reference/20261010T175710Z/)

| Path | Content |
|---|---|
| `baseline_checks/` | The six existing suites at unmodified HEAD, plus an environment note |
| `cpu_checks/environment.txt` | Host, job, modules, packages, GPU, branch, HEAD, `git status` (including ignored tests), diff hash, file hashes, assets, Ray processes |
| `cpu_checks/tested_tracked_changes.diff` | The exact tracked diff that was tested (`tests/test_lp_scheduler.py`) |
| `cpu_checks/1-…8-*.log` | Each command, its verbose output, and exit status |
| `trace_live_decode_preemption.py`, `trace_console.log` | CPU trace script and its output |
| `decision_trace.json`, `summary.json` | CPU per-step snapshots, relaxed and integer results, block operations, replay log, states, and idle; summary with provenance, inputs, boundary, readmission, and decode events |
| `reference/A/`, `reference/B/`, `lp/` | Per GPU case: `command.txt` (resolved command, environment, Ray/GPU state before and after), `console.log`, `summary.json`, `decision_trace.json` |

`lp/summary.json` holds the validated references with their hashes
(A `1bce5f80d502…`, B `ffeec27de1e1…`), the assembled histories and
discarded prefill samples, native output semantics, the three comparisons,
separate execution and comparison flags, the per-step utility calls, the
boundary record, and cleanup. In `command.txt`, the "exit status" line
repeats its label; this is a cosmetic artifact of the wrapper.

The `*.log` and `*.diff` evidence files and
`tests/test_check_lp_scheduler_decode_preemption_reference_gpu.py` match
ignore rules and appear only in `git status --ignored`.

## Limitations

- Bounded to this one workload: one victim, one preemption, one interruption
  point, one GPU. It does not establish repeated preemption, multiple
  victims (or their waiting order), block growth during generation,
  arbitrary-workload completion, fairness, pipeline support, or performance.
- The generation-count defect is inherited, not fixed: LP A requested 2
  tokens and generated 3 (§16.1). Output and text semantics after a reset
  are inherited (§§16.2, 16.7) and not repaired or validated.
- CPU tokens are synthetic; they establish replay and bookkeeping, not model
  output. GPU agreement is a same-framework comparison; it cannot rule out a
  defect shared by both scheduler paths.
- The utilities are scoped test inputs supplied through a validation-only
  override. D-03 remains OPEN, and no production policy interface was added.
- The weight fingerprint checks sampled rows, not every weight.
