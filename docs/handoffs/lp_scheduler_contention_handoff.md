# LPServe `LPScheduler` Three-Request Contention Handoff

## Objective

Show on one small workload that the live LP scheduler repeatedly selects a
subset of eligible requests, leaves omitted requests intact, serves them
later, and completes the whole workload. Three requests (A, B, C) are all
submitted before the first step and compete for two scheduled actions per
step (`s_max=2`) with room for three residents (`max_num_seqs=3`). The case
was checked once on CPU (synthetic sampler) and once on GPU (dummy TinyLlama
weights). The solver's tie choices were recorded, not prescribed.

This is evidence for this workload only. It is **not** evidence of general
fairness, arbitrary-workload correctness, sampler correctness, numerical or
reference agreement, performance, or full native-execution acceptance.
Inherited limitations in `docs/handoffs/lp_scheduler_gpu_handoff.md` and
`docs/handoffs/lp_scheduler_mixed_gpu_handoff.md` still apply.

## Changed paths

- `tests/test_lp_scheduler.py`: one new case,
  `LivePipelineTest.test_three_requests_contend_for_two_action_slots_until_completion`.
  The existing fixtures and tests are unchanged.
- `scripts/check_lp_scheduler_contention_gpu.py` (new). It imports read-only
  helpers from `scripts/check_lp_scheduler_gpu.py` and does not call that
  script's `main`.
- `docs/handoffs/lp_scheduler_contention_handoff.md` (this file).
- `validation_output/lp_scheduler_contention/20261006T043706Z/` (evidence).
- Documentation: `docs/lp_scheduler_design.md` (§13 clarification, §15.6
  evidence wording), `PROJECT_GUIDE.md`, `AGENTS.md`, `docs/project_status.md`,
  `docs/lpserve_scheduler_architecture_summary.md`.

No production code, existing GPU script, historical handoff or artifact, or
pinned audit was changed.

## Evidence

All machine evidence is in
`validation_output/lp_scheduler_contention/20261006T043706Z/`:

| Path | Content |
|---|---|
| `exact_id_characterization/` | Original exact-ID check failure and characterization: tested test-file copy, diff, scratch scripts, raw output, and `README.md` with commands and environment |
| `cpu_checks/environment.txt` | Host, allocation, interpreter, modules, HEAD, `git status`, and hashes of the tested test file and scripts |
| `cpu_checks/1-…4-*.log` | Each unittest command, its full verbose output, and exit status |
| `cpu_checks/5-static.log` | `py_compile`, `git diff --check`, and phase-term scans with exit statuses |
| `gpu_command.txt` | Resolved `PYTHONPATH`, output directory, interpreter, and command |
| `console.log` | GPU run stdout/stderr and the appended `exit_status` line |
| `decision_trace.json` | Per-request admission records; per-step before/events/after state, plans, outputs, and sampler records; the idle record |
| `summary.json` | Provenance (HEAD, `git status`, diff hash, hashes of the executed script, test file, helper, and LP modules, loaded modules), environment, full configuration and prompt IDs, profiled pool, witnesses, decisions, finished outputs, and pass flag |

`*.log` files match the repository ignore rule. `tests/` also matches an
ignore rule, but `tests/test_lp_scheduler.py` is tracked, so `git status`
shows it as modified. `summary.json` records this explicitly.

## Environment (summary; exact values are in the evidence)

The checks ran on `gpu051`, SLURM job `65298930` (`gpu-preempt`), on an NVIDIA
A16 GPU with driver 595.91.07 and CUDA 12.1. The modules were `uri/main`,
`Python/3.10.8-GCCcore-12.2.0`, and `CUDA/12.1.1`. The interpreter was
`env/bin/python` (Python 3.10.8), with the repository root on `PYTHONPATH`.
Dependencies were unchanged from the mixed check (torch 2.3.0+cu121, ray
2.58.0, scipy 1.15.3, and others). The TinyLlama config and tokenizer came
from the existing HF cache, snapshot
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, which equals the config
`_commit_hash`. Before launch there were no Ray processes and no `/tmp/ray`.
The script stopped its own Ray runtime, and no Ray processes remained
afterward. The `/tmp/ray` session directory it left was not deleted.

## Source confirmations

- **Profiling:** `SchedulerType.LP` takes the generic branch of
  `ModelRunner.profile_num_available_blocks`. With `b_max=96` and
  `max_num_seqs=3` this branch builds three 32-token prompts. This was
  confirmed from source and the driver-side inputs; the worker batch was not
  observed. No general profiling-adequacy claim is made.
- **Completion:** `BaseSequenceManager._process_seq_output` advances prompt
  progress for a prefill and discards its sample, including for the
  prompt-completing prefill. It appends exactly one token for a decode.

## Commands and results

Both checks ran from the repository root after module loading,
`source env/bin/activate`, and
`export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"` in the same shell.

**1. Original exact-ID CPU check: FAILED (preserved, not a gate pass).** As
first specified, the new case asserted that the complete central and worker
block tables were equal after every replay. The single-test run failed with
`{0: [7, 8]} != {0: [7, 6]}` (central first). In a scratch characterization
over 12 fresh processes, the IDs **diverged in 5 runs** (4 one way, 1 the
other) and matched in 7. Per-request block counts matched in all 12. The
tested code, commands, and outputs are in `exact_id_characterization/`.

**2. Revised checks (user-approved).** The exact-ID comparison was replaced
by the following checks after every replay:

- equal central/worker allocated-request sets, per-request block counts, and
  free-block counts;
- in each manager independently: allocated block numbers are valid and
  unique, free block numbers are unique, the allocated and free sets are
  disjoint, and together they conserve the original pool;
- unchanged block tables (and worker sequence state) for unselected requests,
  checked in each manager;
- both managers fully drained at the end.

All other assertions are unchanged.

**3. Gates on the revised test.** All passed:

| Command | Result |
|---|---|
| `python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v` | Ran 5, OK, exit 0 |
| `… -p 'test_lpserve_state_mapping.py' -v` | Ran 7, OK, exit 0 |
| `… -p 'test_lpserve_plan_execution.py' -v` | Ran 9, OK, exit 0 |
| `… -p 'test_lp_scheduler.py' -v` | Ran 14, OK, exit 0 (new case `ok`) |
| `python -m py_compile tests/test_lp_scheduler.py scripts/check_lp_scheduler_contention_gpu.py` | exit 0 |
| `git diff --check` | exit 0, no output |
| `rg -n -i 'phase' scripts/check_lp_scheduler_contention_gpu.py` | exit 1, no matches |
| Added test lines scanned with the same pattern | exit 1, no matches |

**4. GPU run (single attempt).** The command was:

```text
timeout 300s python -B scripts/check_lp_scheduler_contention_gpu.py \
  --output-dir "$LP_CONTENTION_OUT" > "$LP_CONTENTION_OUT/console.log" 2>&1
```

Here `LP_CONTENTION_OUT` was
`/home/atjoshi_umass_edu/LPServe/validation_output/lp_scheduler_contention/20261006T043706Z`.
The observed timeout/Python exit status was `0`, and the console printed
`RESULT: PASS`. There were no retries and no relaxed assertions.

## GPU configuration and observed decisions

**Configuration (scoped provisional test inputs):**

- Model: TinyLlama dummy weights, float16, `flash_attention`, TP 1 / PP 1,
  `max_model_len=32`, block size 16, GPU memory utilization 0.5, seed 42.
- Prompts: A = 1000–1015, B = 2000–2015, C = 3000–3015 (disjoint and valid;
  every ID and token is recorded).
- Sampling: greedy, `max_tokens=4`, `ignore_eos=True`.
- Scheduler: `b_max=96`, `c_max=8`, `s_max=2`, `max_num_seqs=3`, reserve 1,
  `conservative_one_block_v1`, utilities 1/1/1, `lp_relaxation_mvp_v1`.

**Pool:** profiled pool N = 15601 blocks, with a watermark of 156 blocks. The
script checked `N − 6 − 2 ≥ 156 + 1` before adding any request. Request IDs
were A = 0, B = 1, C = 2.

**Decisions** (plan per request is `(prefill, decode, preempt)`):

| Step | Emitted `(seq, chunk)` | Omitted | Notes |
|---|---|---|---|
| 1 | `[(1,8),(2,8)]` | A (waiting) | B, C admitted |
| 2 | `[(1,8),(2,8)]` | A (waiting) | B, C prompts complete |
| 3 | `[(0,8),(2,0)]` | B (resident) | A admitted; C decodes |
| 4 | `[(0,8),(2,0)]` | B (resident) | A prompt complete; C appends a block |
| 5 | `[(1,0),(2,0)]` | **A** (resident) | all three prompt-complete and allocated |
| 6 | `[(1,0),(2,0)]` | **A** (resident) | C finishes and is freed |
| 7 | `[(0,0),(1,0)]` | — | **A selected after omission** |
| 8 | `[(0,0),(1,0)]` | — | B finishes |
| 9 | `[(0,0)]` | — | |
| 10 | `[(0,0)]` | — | A finishes; free = N |

Remaining work went 60 → 44 → 28 → 19 → 10 → 8 → 6 → 4 → 2 → 1 → 0. Each
step's reduction equalled its emitted tokens.

**Witnesses** (recorded in `summary.json`):

1. A was an eligible waiting request omitted at steps 1–2, and it stayed
   waiting with no blocks.
2. Steps 5–6 were boundaries with all three requests prompt-complete,
   allocated, and unfinished, with two selected.
3. At those boundaries A was the omitted resident. Its status, prompt
   progress, generated token IDs, logical blocks, and central block IDs were
   unchanged through each step.
4. A was selected at step 7 and in every later step.
5. All three requests finished.

Each nonempty step also passed these assertions:

- The call order was exactly mapper → solver → executor → one
  `execute_model` worker call.
- The snapshot, problem, plan, and executor identities matched, and the
  mapped limits, policy, utilities, and free blocks matched the inputs.
- The finished request was excluded from the planning universe.
- The complete plan matched the emitted actions, prompt-first and then by
  `order_key`, with no preemption or ignore controls.
- Chunk, token, action, and resident limits were respected.
- Native counts matched the metadata.
- Running batches were 0 before scheduling, 1 after scheduling, and 0 after
  completion.
- Scheduling did not advance progress.
- Admission and append allocation deltas, including block-ID prefix
  preservation, matched the actions, and completion frees matched.
- Prefill advanced only by its chunk and appended no token.
- Each decode appended exactly one token.
- Returned request outputs matched in order, token count, and finished flag.

**Completion:** each request produced exactly one finished `RequestOutput`
with 4 tokens and finish reason `length`: A `[10935, 15928, 554, 21418]`, B
`[3591, 7027, 30481, 3049]`, and C `[10123, 30925, 4974, 20646]`. At the end,
`waiting`, `running`, the central block tables, and the sequence manager were
empty, and free blocks were back to N = 15601.

**Idle call:** returned `[]` and called only `schedule`; the iteration went
9 → 10 and no other captured state changed.

Sampler IDs come from worker-side metadata. They do not independently prove
which hidden-state row produced each token. GPU-worker block tables were not
instrumented. The dummy-weight text is not evidence of quality.

## CPU case observations

The CPU case passed every per-step assertion listed in the task, plus the
revised central/worker checks above. That includes the same witnesses, a
24-step termination guard, a remaining-work decrease of at least one unit
per step, native completion, both managers drained and conserved, and an
idle call. The CPU sampler token is synthetic, so this case tests scheduling,
replay, and bookkeeping, not generation.

## Issues and limits

- **Observed inherited limitation — set-order free.**
  `BaseBlockSpaceManager._free_block_table` iterates `set(block_table)`, and
  `PhysicalTokenBlock` uses identity hashing. As a result, the central and
  worker allocators can return a freed multi-block table's blocks in
  different orders, and later allocations can then get different physical
  IDs. Counts and ownership agree, and each pool stays internally consistent.
  This milestone therefore does **not** establish identical central/worker
  physical block IDs (design §13 clarification). Not repaired.
- The schedule observed here reflects the solver's tie handling for this
  input. It is not a claim about fairness or about other workloads.
- Not run or not established: independent GPU-worker block tables, reference
  numerical comparison, mixed sampling types, preemption, pipeline execution,
  and benchmarks.

OPEN decisions: none resolved. All values are scoped provisional test inputs.
Contract changes: the §13 test-scope clarification and the §15.6 evidence
wording, both as authorized.
