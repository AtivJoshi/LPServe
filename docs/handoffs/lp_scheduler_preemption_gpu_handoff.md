# LPServe Native Preemption Dummy-Weight GPU Handoff

## Objective

Check, on one bounded GPU workload, that the registered `LPScheduler` and
the native preemption path work together on a real engine:

1. partially prefill request A;
2. select A's preemption to admit request B under real cache pressure;
3. run B to completion;
4. readmit A, recompute its prompt from token zero, and complete it;
5. release all allocations and reach ordinary idle.

The run used one tensor-parallel worker, one pipeline stage, and dummy
TinyLlama weights. It continues the CPU preemption work accepted at
`04beb8789f17e11d55aec72d9f168353d486b2ba`
(`docs/handoffs/lp_scheduler_preemption_handoff.md`).

## Added paths

- `scripts/check_lp_scheduler_preemption_gpu.py`: the validation driver.
- `tests/test_check_lp_scheduler_preemption_gpu.py`: 7 CPU checks of the
  driver's cache-capacity selection. `tests/` matches an ignore rule, so
  `git status` shows this new file only with `--ignored`.
- `docs/handoffs/lp_scheduler_preemption_gpu_handoff.md` (this file).
- `validation_output/lp_scheduler_preemption_gpu/20261010T062705Z/`: the
  interrupted first CPU-gate run, kept as recorded.
- `validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/`: the
  complete CPU gates and the GPU run.

No production code changed: the scheduler, executor, mapper, mathematics,
engine, worker, cache, and configuration are untouched. Existing scripts,
historical evidence, `docs/math/`, the design document, and the pinned audit
are also unchanged. The engine, worker, cache-engine, and model-runner
sources have no diff from the audit baseline `c3e0143`.

## How the driver works

- **Four-block cache.** A driver-local `BaseLLMEngine` subclass intercepts
  one result of `_run_workers`: the one for `profile_num_available_blocks`.
  It records the actual profiled value. It requires exactly one non-boolean
  integer per worker, at least 4 blocks, and only one profiling call; it
  never substitutes a larger pool. It then returns `[4]`.
  - Native `_init_cache` runs unchanged. It validates the value (4 blocks
    are needed for a 64-token request) and calls `init_cache_engine`, which
    builds the worker GPU cache and the worker block manager.
  - The engine then builds the central scheduler from the same
    `CacheConfig`. The subclass also records the single `init_cache_engine`
    call and requires 4 blocks there.
  - Native profiling itself is unchanged. Its generic branch runs one
    64-token request (`b_max=64`, resident limit 1).
- **Validation worker.** A `BaseWorker` subclass is supplied through the
  engine's existing `_get_worker_impl` hook.
  - After native `init_cache_engine`, it records its own block manager's
    `free` and `allocate` calls. The wrappers delegate unchanged.
  - It adds one read-only method, `validation_state`, which returns
    primitive cache, manager, and sequence state and clears only its own
    call log.
  - The class is built inside a function, so importing the driver loads no
    engine, worker, Ray, or torch. Ray serializes the class by value.
- **Central observation.** The driver wraps mapping, solving, execution,
  `schedule`, `_run_workers`, and the central block manager's
  `free`/`allocate`. Each wrapper records and returns the native result
  unchanged.

## Configuration (provisional scoped inputs)

- **Model and execution:** the TinyLlama/TinyLlama-1.1B-Chat-v1.0
  architecture and tokenizer from local snapshot
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6` (offline), dummy weights,
  float16, FlashAttention, seed 42.
- **Parallelism and cache:** tensor parallel 1, pipeline stages 1, maximum
  model length 64, block size 16, profiling GPU-memory utilization 0.5, and
  4 initialized blocks.
- **Scheduler:** resident limit 1, `b_max=64`, `c_max=16`, `s_max=1`,
  reserve 0, `conservative_one_block_v1`.
- **Utilities:** prefill-token 1, decode 40 (user-approved for this case),
  penalty 1.
- **Numerical policy:** `lp_relaxation_mvp_v1` (feasibility 1e-7,
  integrality 1e-6, objective tolerances 1e-9).
- **Metrics:** enabled, with every optional output off.
- **Requests:** A is seq 0 with 17 prompt tokens, IDs 1000–1016. B is seq 1
  with 48 prompt tokens, IDs 2000–2047. Both are greedy, with
  `max_tokens=1`, `ignore_eos=True`, and no stop strings. B was submitted
  only after A's first step completed.

D-03 remains OPEN. All values above are scoped inputs, not policy decisions.

## Environment

- **Host:** `gpu051.unity.rc.umass.edu`, SLURM job `65495160` (`gpu-preempt`),
  `CUDA_VISIBLE_DEVICES=0`.
- **GPU:** NVIDIA A16 (UUID `GPU-b7192761-b009-1409-dfe0-541ef40add86`),
  driver 595.91.07, 15356 MiB, torch CUDA 12.1.
- **Modules:** `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, and
  `CUDA/12.1.1`, plus their dependencies.
- **Interpreter:** `env/bin/python` (3.10.8).
- **Packages:** torch 2.3.0+cu121, transformers 4.57.6, ray 2.58.0,
  numpy 2.2.6, scipy 1.15.3, vllm-flash-attn 2.5.9.
- **Revision:** tested at HEAD `04beb8789f17e11d55aec72d9f168353d486b2ba`
  plus the untracked files listed above. There were no tracked
  modifications.

## Commands and results

Each shell ran `module load Python/3.10.8-GCCcore-12.2.0`,
`module load CUDA/12.1.1`, `source env/bin/activate`, and
`export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`.

**Interrupted run, `20261010T062705Z/`** (gpu052, job 65493676). The command
was interrupted, and that allocation was later preempted. Only the
environment record and the mathematical suite (5 OK) completed. The mapper
log has only its command line. Nothing else ran, including any GPU attempt;
`INTERRUPTED.md` explains this. Its recorded file hashes equal those of the
complete run.

**Complete run, `20261010T063249Z/cpu_checks/`** (gpu051):

| Command | Result |
|---|---|
| `python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v` | 5 OK |
| `… -p 'test_lpserve_state_mapping.py' -v` | 7 OK |
| `… -p 'test_lpserve_plan_execution.py' -v` | 13 OK |
| `… -p 'test_lp_scheduler.py' -v` | 15 OK |
| `… -p 'test_check_lp_scheduler_preemption_gpu.py' -v` | 7 OK |

Every exit status was 0, and nothing was skipped. The driver tests check:

- the driver import loads no torch, Ray, engine, or worker module;
- the actual profile is retained and 4 is returned;
- insufficient capacity, an unexpected result shape, and a second profile
  each fail;
- through native `_init_cache`, 4 blocks reach `init_cache_engine`;
- insufficient capacity fails before cache initialization;
- other worker calls are delegated unchanged.

`6-static.log` records the static checks:

- `py_compile` of the driver and its test: exit 0.
- `git diff --check` on the driver and this handoff: exit 0. The check is
  vacuous here because both files are untracked.
- A direct trailing-whitespace scan: no match.
- A phase-terminology scan: no match.

**GPU attempt (one, first try):**

```
timeout 300s python -B scripts/check_lp_scheduler_preemption_gpu.py \
  --model-path /home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
  --output-dir validation_output/lp_scheduler_preemption_gpu/20261010T063249Z
```

The run printed `RESULT: PASS`, with exit status 0. The only warning was
`Casting torch.bfloat16 to torch.float16`, which comes from the requested
dtype.

## Observed results

**Cache capacity:**

- The actual profiled capacity was 15606 blocks; 4 were deliberately
  initialized.
- Profiling ran once, and initialization ran once with 4.
- Central: `CacheConfig`, manager, and allocator each had 4 blocks, all 4
  free, with watermark 0.
- Worker: `CacheConfig` and cache engine had 4 blocks. All 22 layer cache
  tensors had shape `[4, 16, 4, 64]`. The manager and allocator had 4
  blocks, all 4 free.

**Emitted schedule** (preempted; scheduled), all matching the required
sequence:

| Step | Output | Central / worker free after replay | Sampled token |
|---|---|---|---|
| 0 | `[]`; `(0,16)` | 2 / 2 | 10935 (prefill, discarded) |
| 1 | `[0]`; `(1,16)` | 1 / 1 | 3591 (prefill, discarded) |
| 2 | `[]`; `(1,16)` | 1 / 1 | 24476 (discarded) |
| 3 | `[]`; `(1,16)` | 1 / 1 | 13271 (discarded) |
| 4 | `[]`; `(1,0)` | 4 / 4 | 13271, B finishes |
| 5 | `[]`; `(0,16)` | 2 / 2 | 10935 (discarded) |
| 6 | `[]`; `(0,1)` | 2 / 2 | 11087 (discarded) |
| 7 | `[]`; `(0,0)` | 4 / 4 | 11087, A finishes |
| idle | none | 4 / 4 | iteration 7 → 8; only `schedule` called |

**Preemption boundary (step 1):**

- *Mapped state:* legal victim set `{"0"}`; `m_free=2`, `w=0`; A was
  running, PAUSED, 16/17 processed, with 2 physical blocks and recovery 2;
  B was waiting, with fixed charge 3.
- *Solution:* relaxed A was x=0, y=0, I=0, z=0.5; B was x=16, y=0, I=1, z=0.
  The relaxed objective was 15.5. The integer plan preempted A and
  prefilled B for 16 tokens (objective 15, A dominant).
- *Central execution:* `free(0, [3, 2])` and then `allocate(1, [2, 3, 1])`.
  After scheduling, `waiting=[0]` and `running=[1]`. A was still PAUSED
  with 16 processed; it changed only at replay. B's metadata was the only
  scheduled entry.
- *Worker replay:* `free(0, [3, 2])` and then `allocate(1, [2, 3, 1])`.
- *After replay:* both central and worker copies of A were WAITING, with 0
  processed, prompt incomplete, no output tokens, and the same 17 prompt
  IDs. B had 16 processed and 3 blocks in both managers.

**Readmission (step 5):**

- The mapper saw A waiting, unallocated, with 0 processed and a full-context
  charge of 2.
- Central and worker each made exactly one `allocate(0)` of 2 blocks, and
  both reached 16 processed.
- The physical IDs differed (central `[1, 2]`, worker `[3, 1]`). This is the
  inherited set-order free behavior; allocated sets, per-request counts,
  free counts, and each manager's pool integrity were checked instead.
- A's recomputed 16-token prefill sampled the same token (10935) as its
  first prefill. That fits recomputation from token zero with greedy
  dummy weights. The run checks prompt progress and fresh allocation; it
  does not inspect KV contents.

**Completion:**

- B finished before A. Each finished with one token and finish reason
  `length` (B `[13271]`, A `[11087]`).
- The finished request was freed by one central and one worker `free` at
  its finishing step.
- Afterwards, queues, the central and worker sequence maps, and the block
  tables were empty, with 4 free blocks in each manager.

**Checked at every step:** zero batches in flight at mapping, one pipeline
stage, the call order map → solve → execute → schedule → `execute_model`,
the plan matching the emitted controls and metadata, the token and action
limits, the sampler-output association, unique and disjoint ownership,
resident count ≤ 1, and central/worker agreement.

**Repeated token:** a prompt-completing prefill and the following decode
sampled the same token (13271 for B, 11087 for A). Native completion
discards the prefill's sample, and the decode recomputes from the same
context. This is recorded as inherited behavior and not analyzed further.

**Cleanup:**

- `ray.shutdown()` ran: Ray was initialized before it and not after.
- There were no Ray processes for this user before or after the run.
- `/tmp/ray` did not exist before the run and exists afterwards; it was not
  deleted, matching earlier GPU checks.
- No metrics output directory was created.

## Evidence

The complete run is in
[`validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/`](../../validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/):

| Path | Content |
|---|---|
| [`cpu_checks/environment.txt`](../../validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/cpu_checks/environment.txt) | Host, job, modules, GPU, HEAD, `git status` (including the ignored test), versions, asset listing, file hashes, and Ray processes |
| `cpu_checks/1-…6-*.log` | Each command, its verbose output, and exit status |
| [`gpu_command.txt`](../../validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/gpu_command.txt) | Resolved command, `PYTHONPATH`, interpreter, modules, and Ray state before and after |
| `console.log` | GPU run output and exit status |
| [`summary.json`](../../validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/summary.json) | Provenance (HEAD, status, diff hash, hashes of the driver, helper, tests, LP modules, and native paths), environment, configuration and prompt IDs, cache capacity, decisions, the boundary record, final outputs, and the pass flag |
| [`decision_trace.json`](../../validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/decision_trace.json) | Per step: central and worker state before; mapped snapshot, relaxed solution, plan, executor output, central state after execution and central block calls, and sampler outputs; central and worker state after; worker block calls; request outputs; plus the idle record |
| [`cleanup.json`](../../validation_output/lp_scheduler_preemption_gpu/20261010T063249Z/cleanup.json) | Ray state around shutdown |

The interrupted run is in
[`20261010T062705Z/INTERRUPTED.md`](../../validation_output/lp_scheduler_preemption_gpu/20261010T062705Z/INTERRUPTED.md).
The driver and test hashes recorded by the CPU gates equal those recorded
by the GPU run.

## Limitations

This run establishes only this bounded dummy-weight path on one GPU. It
does not establish:

- numerical agreement with real weights, or useful generation;
- interruption after generated tokens (the §§16.1–16.2 limitations still
  apply);
- arbitrary-workload completion, fairness, or pipeline support;
- corrected output semantics or performance;
- independent framework correctness.

Native text (`" updating"`, `"unkt"`) does not show that token-to-text
conversion is correct, and no LP/reference token comparison was made.
Equality of physical block IDs between the central and worker managers is
not claimed.
