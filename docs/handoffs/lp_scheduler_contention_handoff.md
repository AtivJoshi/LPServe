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


## Corrected-framework GPU regression (2026-10-13; awaiting review)

The sections above are historical evidence for the superseded contracts
(planning reserve, conservative decode charge, `order_key`-only ordering).
This run repeats the same workload on the corrected framework. It validates
only this dummy-weight workload.

**Provenance.** Clean `main` at `82609eb290df754b841aecc1c1a379ea8a325ecb`,
with no tracked or untracked changes other than this run's own output
(`git diff HEAD` empty; `summary.json` diff hash is the empty-input SHA-256).
Since the tested code equals the accepted CPU baseline, no CPU suite was
rerun. No code was changed. Evidence:
`validation_output/lp_scheduler_contention/20261013T064744Z/`
(`environment.txt`, `gpu_command.txt`, `console.log`, `summary.json`,
`decision_trace.json`, `trace_analysis.py`/`.txt`). SHA-256 of the tested
sources is listed in `environment.txt`, including
`scripts/check_lp_scheduler_preemption_gpu.py` (`517fef8b…`). The driver
imports `check_declared_order` from that file but does not hash it in
`summary.json`.

**Environment.** `gpu051`, SLURM job `65621892` (`gpu-preempt`), NVIDIA A16
(`GPU-c1783933-ac1a-85f1-4c50-56eb59884cf1`), driver 595.91.07, torch
2.3.0+cu121, ray 2.58.0, scipy 1.15.3, numpy 2.2.6, transformers 4.57.6;
modules `uri/main`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`;
`env/bin/python` 3.10.8. The repository root was on `PYTHONPATH`, and
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` were exported to prevent
downloads; neither changes the driver. Assets: HF cache snapshot
`fe8a4ea1ffedaf415f4da2f062534de366a451e6` (`refs/main` and config
`_commit_hash`), dummy weights. Before the run there were no Ray processes
and no `/tmp/ray`. Afterwards `ray.shutdown()` had run, no Ray processes
remained, and the `/tmp/ray` it created was left in place.

**Command (single attempt):**

```text
timeout 300s python -B scripts/check_lp_scheduler_contention_gpu.py \
  --output-dir "$LP_CONTENTION_OUT" > "$LP_CONTENTION_OUT/console.log" 2>&1
```

Exit status `0`, `RESULT: PASS`. The log contains the warning
`Casting torch.bfloat16 to torch.float16` and the transformers notice
`` `torch_dtype` is deprecated! Use `dtype` instead! `` (`console.log` line 1;
corrected 2026-10-13, earlier text said the cast warning was the only one).
The post-hoc trace analysis
(`python3 -I trace_analysis.py <dir>`, which reads JSON only) exited 0 with
no failures.

**Inputs.** Unchanged from the driver: prompts A/B/C = 16 token IDs from
1000/2000/3000; greedy, `max_tokens=4`, `ignore_eos`; `b_max=96`,
`c_max=8`, `s_max=2`, `max_num_seqs=3`; uniform utilities 1/1/1;
`lp_relaxation_mvp_v1`; TP 1 / PP 1; `max_model_len=32`; block size 16. Profiled
pool N = 15601 with native watermark h = 156. Seq IDs were A = 0, B = 1, C = 2.

**Observed decisions** (D0 = zero-demand decode, D1 = one-block-demand decode):

| Step | Running before | Emitted `(seq, chunk)` in order | Groups | Omitted | Relaxed = integer objective |
|---|---|---|---|---|---|
| 1 | — (A, B, C waiting) | B 8, C 8 | admit B, C | **A waiting** | 16 |
| 2 | B, C | B 8, C 8 | resident prefills | **A waiting** | 16 |
| 3 | B, C | A 8, C 0 | admit A (tail); C D0 | B | 9 |
| 4 | B, C, A | A 8, C 0 | A prefill; C D1 | B | 9 |
| 5 | B, C, A | B 0, C 0 | D0, D0 | **A resident** | 2 |
| 6 | B, C, A | **C 0, B 0** | C D0, then B D1 | **A resident** | 2 |
| 7 | B, A | B 0, A 0 | D0, D0 | — | 2 |
| 8 | B, A | **B 0, A 0** | B D0, then A D1 | — | 2 |
| 9 | A | A 0 | D0 | — | 1 |
| 10 | A | A 0 | D0 | — | 1 |

C finished at step 6, B at step 8, and A at step 10. Each produced four tokens with
finish reason `length` (A `[10935, 15928, 554, 21418]`, B `[3591, 7027,
30481, 3049]`, C `[10123, 30925, 4974, 20646]`). The idle call returned
`[]`, called only `schedule`, and advanced the iteration 9 → 10 with no
other change. Free blocks returned to 15601. Every relaxed solution was
integral (fractional count 0).

**Checked by the driver at every nonempty step:** one call each to mapping,
solve, and execution, followed by `schedule` and one `execute_model`; matching
snapshot, problem, and plan identities; mapped limits, policy, utilities, and
free blocks; derived declared order (victims by victim rank; prefills, D0,
D1, each by service rank, then identity) equal to the plan's lists and to
emitted metadata; no ignore or preempt controls; chunk, token, and action
limits; output counts; sampler-ID association; batch state 0 → 1 → 0; no
progress change during scheduling; admission and append allocation deltas
with block-ID prefix preservation; unselected requests unchanged; exact
completion frees; remaining-work decrease equal to emitted tokens; returned
outputs. The 60-step termination bound was not approached (10 steps).

**Checked post hoc** (`trace_analysis.txt`). Service and victim ranks equal
the observed `running` and arrived-`waiting` positions. Each mapped decode
demand equals the observed logical-minus-physical gap. Admission charges equal
the full-context block count (1), and resident charges are 0. At the executed
integer plan: H = m_free − admission charges + victim recovery ≥ decode
demand, H ≥ h when admitting, H ≥ 1 when decoding, and the plan's
`free_blocks_before_decodes`/`decode_demand` equal those values. The observed
free count after scheduling equals H − demand. Retained residents keep their
running order, and admissions join the tail in declared order (step 3: B, C,
A). The resident count is at most 3. The groups are contiguous in the order
prefill, D0, D1.

**Witnesses.**

1. A was an eligible waiting request omitted at steps 1–2. It stayed
   `WAITING` with no blocks.
2. Steps 5 and 6 were boundaries where all three requests were resident,
   prompt-complete, and unfinished.
3. A was the omitted resident at both boundaries and was unchanged through
   each (status, progress, output IDs, logical blocks, central block IDs).
4. A was selected at steps 7–10.
5. All three requests finished.

**Exercised cases.**

- Admission (steps 1, 3), with the admission at step 3 joining the running tail.
- Resident partial prefill (steps 2, 4) and prompt completion (steps 2, 4).
- D0 (steps 3, 5–10) and D1 (steps 4, 6, 8). Both decode groups occurred
  **together in one batch** at steps 6 and 8. At step 6 the emitted order (C,
  then B) differs from running order (B before C). This exercises the corrected
  zero-demand-first rule rather than coinciding with running order.
- Prefill with decode in one batch (steps 3, 4).
- Completion and freeing (steps 6, 8, 10), final drain, and idle.

**Not exercised.** Preemption: every resident was an eligible victim (legal
preemption IDs at steps 2–10 were the current residents in
`decision_trace.json`), but no victim was selected (empty victim order and
`preempt = 0` at every step). Corrected 2026-10-13; earlier text said none
were legal. A binding watermark (h = 156 but H − h ≥ 15442 at every admission). A
fractional relaxation.

**Limitations.**

- The driver does not instrument GPU-worker block tables. Central allocation
  deltas, sampler IDs, and central counts are not evidence of worker
  allocation.
- Identical central/worker physical block IDs are not claimed (inherited
  set-order free).
- The schedule reflects the solver's choice among tied optima. It is not a
  fairness claim.
- Dummy-weight tokens are not evidence of sampler correctness or output
  quality. Their equality with the 2026-10-06 run reflects the same seed and
  weights.
