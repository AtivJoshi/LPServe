# LP scheduler research project status

## Current phase

**Phase D accepted; mapper, native executor, and live scheduler implemented. Bounded CPU integration milestone accepted on 2026-10-05 after review at `4f553b4`. The bounded single-request dummy-weight GPU milestone was also accepted after review at `14a1535`; full Phase F acceptance is not claimed.**

The pure, framework-independent LP-relaxation mathematical layer is implemented
in `lp_relaxation_scheduler.py` and accepted for the scoped MVP. It constructs
and solves the documented continuous relaxation with SciPy/HiGHS, independently
validates the relaxed solution, performs deterministic feasible integer
extraction, validates the resulting plan, and returns structured success or
failure without importing LPServe runtime objects.

Phase E read-only state mapping is implemented in `lpserve_state_mapping.py`,
with focused tests in `tests/test_lpserve_state_mapping.py`. It constructs
immutable mathematical-layer inputs without scheduler or serving-state
mutation. Retained Unity CPU test evidence and its revision limits are recorded
below. Native execution and live `LPScheduler` integration now support the
scoped single-stage prefill/decode path, with CPU replay/completion evidence.
The subsequent GPU milestone establishes only the exercised single-request
execution path. General sampler correctness, generation quality, mixed-batch
GPU behavior, and performance remain unverified. See the 2026-10-05 records below.

## Current repository and provenance

| Field | Recorded value |
|---|---|
| Local branch | `main` |
| Phase C audited code revision | `c3e014363dd50e1830d7c85c3d043eab69fdc9e5` (`gitignore update`) |
| Initial Phase D implementation | `f406eeaac7bafc4478304744c9187316b75b8f67` |
| Zero-valued extraction-tie contract fix | `32036f6a0ddb9bf2b55fc81f48967d61d558b4c2` |
| Regression optimality correction | `db3a64afbe1aec2b6b11d477ced4f2f888288a06` |
| Accepted Phase D implementation/evidence HEAD (2026-09-20) | `67a336222a6ec2ac53d932e7928c597dd4ccbe23` |

`docs/lpserve_scheduler_architecture.md` remains intentionally pinned to
`c3e0143`. The standalone mathematical module does not change the audited
scheduler, engine, sequence, block-management, or execution paths and therefore
does not repin the historical audit. The Mac review checkout was clean and
synchronized with `origin/main` at `67a3362` when acceptance was recorded.
These statements are a dated snapshot and must be rechecked after later changes.

## Phase C work completed

Phase C produced a read-only, code-grounded account of the scheduler-relevant
LPServe/SLAI-derived framework at `c3e0143`. The audit covered scheduler
selection, request and sequence state, queue and residency ownership, scheduler
outputs and replay, block management, native action traces, mutation timing,
pipeline behavior, mixed-batch ordering, metrics, known defects, and gaps that
constrain later LP work. Findings retain the evidence classifications and
limitations defined in `docs/lpserve_scheduler_architecture.md`.

The audit also informed later mathematical/source clarification. In particular,
the checked-in mathematical and research-context sources now distinguish the
true continuous relaxation of the prefill variable, legal preemption support,
and LPServe's fixed prefill-admission charge from claims about existing
framework behavior. `docs/lp_scheduler_design.md` translates the mathematical
target and audited framework constraints into a normative Phase D--F design
while retaining explicit OPEN decisions and BLOCKERs.

### Durable Phase C and readiness outputs

- `docs/lpserve_scheduler_architecture.md` — descriptive, commit-pinned Phase C
  architecture audit; not the proposed scheduler specification.
- `docs/lp_scheduler_design.md` — normative design contract for Phases D, E,
  and F; not evidence of implementation.
- `docs/LP Scheduler Research Context.md` — updated research assumptions,
  implementation principles, validation requirements, and phase boundaries.
- `docs/math/main-llm-serving.tex` — repository-local mathematical source of
  truth for the proposed formulation, especially Primal Heuristic 1.
- `PROJECT_GUIDE.md` and `AGENTS.md` — repository navigation and agent
  instructions; not Phase C code-audit evidence.

### Phase C evidence boundary

Phase C was primarily static source inspection, supplemented only by explicitly
identified Phase B observations. It did not add or run a new automated test
suite, reproduce the audited defects at runtime, perform a new GPU validation,
or establish performance. It did not implement LP problem construction, solver
integration, integer extraction, LPServe state mapping, native LP action
execution, or fixes for the documented framework blockers.

## Historical Phase D acceptance and Phase E handoff (2026-09-20)

Phase D delivered the scoped pure mathematical layer in
`lp_relaxation_scheduler.py`, its focused tests in
`tests/test_lp_relaxation_scheduler.py`, and an explicit SciPy `1.15.3`
dependency. Python implementation names intentionally omit phase labels because
phase names and numbers are project-management terminology rather than runtime
or API terminology.

Observed Unity evidence used Python 3.10.8, NumPy 2.2.6, and SciPy 1.15.3. It
included successful syntax compilation, the independent `main()` smoke plan,
and five passing focused unit tests:

1. build/solve/extract/print smoke behavior;
2. mixed prefill, decode, preemption, capacity, and ordering behavior;
3. continuous fractional prefill with feasible integer extraction;
4. visible infeasible-solver failure without a fabricated plan; and
5. the approved zero-valued decode/prefill tie rule, including sensitivity
   against the pre-fix implementation.

The zero-valued rule is normative in `docs/lp_scheduler_design.md` §10.6 and
the corresponding Primal Heuristic 1 algorithm: normalized
`y == I^P == 0` selects no execution action, while other exact ties select
decode. The regression uses zero recovery and zero preemption penalty so its
fractional `z=0.5` point is a degenerate optimum rather than merely feasible.

Detailed command output and provenance are retained outside the main
documentation listing:

- `docs/handoffs/lp_relaxation_implementation_handoff.md`;
- `docs/handoffs/lp_relaxation_zero_tie_fix_handoff.md`.

At that Phase D acceptance boundary, no Phase E mapping, Phase F execution,
live integration, GPU work, queue or
block mutation, framework repair, approximation guarantee, or permanent
utility/capacity/reserve/decode-charge policy was established. Remaining OPEN
decisions in `docs/lp_scheduler_design.md` remain OPEN unless explicitly
recorded there.

At that boundary, the next permitted implementation work was the smallest
read-only Phase E mapper
described by the design. It must preserve the accepted mathematical interface,
fail visibly on incoherent snapshots, and perform no scheduler or serving-state
mutation.

## Historical Phase E implementation and evidence boundary (2026-10-04)

This dated record is preserved; the 2026-10-05 evidence below supersedes its
then-current implementation and validation boundary.

The mapper was introduced in `a0e35d7`, corrected in `0204393`, and aligned
with the current ordinary-idle contract in `1c9141a`. The retained
`docs/handoffs/state_mapping_idle_contract_handoff.md` records post-commit
syntax checks and five mathematical-layer plus five mapping tests passing on
Unity node `gpu048` at `1c9141a`, using Python 3.10.8, NumPy 2.2.6, and SciPy
1.15.3. That validation was CPU-only; no model, GPU kernel, or benchmark ran.
Earlier implementation and correction evidence remains in
`docs/handoffs/lpserve_state_mapping_handoff.md`.

Subsequent simplifications use the scheduler iteration number as the snapshot
ID (`2c31d5b`), accept a utility mapping (`1996b68`), and retain physical block
counts instead of identities (`78d663a`). Source inspection confirms these
changes and the continued absence of a live `LPScheduler` or LP plan executor
at the inspected revision. The retained `1c9141a` test results do not establish
runtime validation of these later commits. Local review syntax and commit
whitespace checks passed, but focused CPU tests could not start because SciPy
was unavailable in the review environment. Current-revision runtime validation
therefore remains unverified by this review.

The mapper returns a snapshot or mapping failure. Ordinary empty/future-only
live scheduling is specified for the future scheduler; its documented behavior
is not implemented integration evidence. Phase F execution, live integration,
and subsequent GPU correctness validation remain pending and subject to the
existing design gates. No OPEN decision or BLOCKER is resolved by this status
correction.

## Native executor and live CPU integration milestone (2026-10-05)

The user accepted this bounded milestone after source review at clean Mac HEAD
`4f553b4f238fe93e87623de3612d20362888899f` and review of the retained Unity
handoffs. This is not full Phase F acceptance or GPU validation.

| Change | Implementation | Evidence |
|---|---|---|
| Native executor and D-21 prompt-first ordering | `0afba0d7a7d7f5952975332086e1cca953aff5c4` | `74f5d09`, `docs/handoffs/lpserve_plan_execution_handoff.md` |
| Nested-input failure-interface correction | `f053afd74f8f91db5d8feefd91dfc1f8d428717e` | `d1bd69c`, `docs/handoffs/lpserve_plan_execution_failure_interface_handoff.md` |
| Live scheduler, explicit configuration, registration, CPU replay tests | `276e5f4c188b6137fd5f953eda67503d41b0718f` | `ec15af2`, with post-commit evidence recorded in `4f553b4`, `docs/handoffs/lp_scheduler_integration_handoff.md` |

The integration handoff reports post-commit checks at `276e5f4` on Unity
`gpu048`, allocation `65250285`, using the existing repository environment:
Python 3.10.8, NumPy 2.2.6, SciPy 1.15.3, Torch 2.3.0+cu121, and Transformers
4.57.6. Its recorded commands run the mathematical, mapper, executor, and live
scheduler suites separately: 5, 7, 9, and 13 tests passed, respectively. Syntax
and whitespace checks also passed. No model, CUDA execution, or benchmark ran.
This evidence includes the mapper simplifications that lacked current-revision
runtime evidence in the 2026-10-04 record.

Exercised CPU behavior includes synchronous mapping through execution, ordinary
idle, iteration and running-batch bookkeeping, immutable pre-mutation failure
propagation, destructive post-mutation exception propagation, and native
central/worker replay. Synthetic sampler outputs drive admission, partial
prefill, decode at physical block gaps zero and one, completion/free, mixed
prompt-first output, and preservation of an unselected resident. Detokenization
is stubbed. Central/worker block equality is established only for the exercised
CPU cases, not for model or GPU execution.

Local review on `atmac.local` used Python 3.14.7 at
`/opt/homebrew/opt/python@3.14/bin/python3.14`. In-memory syntax compilation,
`git diff --check d1bd69c HEAD`, and the new-module naming check passed. The
command `python3 -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v`
could not import because NumPy was unavailable. No substantive integration
tests ran on the Mac; Unity passes above are reported handoff evidence.

The mathematical layer, mapper, and executor were unchanged by live integration.
D-21 is resolved for the supported single-stage ordering; provisional utilities
and capacities do not resolve the remaining research-policy decisions. Runtime
preemption, ignore controls, pipeline execution, and overlapping calls remain
unsupported. LP scheduling requires the repository root on Python's import path.
Direct configuration/registry construction is supported; `EngineArgs` and
benchmark options remain deferred.

At this CPU milestone boundary, the next permitted work was planning the
smallest GPU correctness test, including review of the model runner's default
memory-profiling path for `SchedulerType.LP` and the launch/import path. Model
execution and GPU correctness were then unverified; the subsequent bounded
GPU evidence is recorded below. General sampler correctness, generation
quality, arbitrary-workload success, and performance remain unverified. The
inherited limitations in design §16 remain in force.

The evidence commit `ec15af2` also began tracking the pre-existing `CLAUDE.md`
containing `@AGENTS.md`, outside the implementation prompt's allowed paths.
The integration handoff's dated addendum records this file-scope discrepancy.

## Bounded single-request dummy-weight GPU milestone (2026-10-05)

The user accepted this milestone after read-only review at clean HEAD
`14a15356e64e175f5def042d582412b40af4de2d`. The script is committed as
`54bcd72a62571457fe074d2edc814ce6f8ea7827`; the handoff and retained run
artifacts are committed as `14a1535`. See
`docs/handoffs/lp_scheduler_gpu_handoff.md` for commands, full environment,
configuration, and artifact hashes.

The single Unity GPU attempt tested base
`a3be58a4e632da79edcc4b3b5d70b0471ed013d9` plus the uncommitted validation
script. Its full SHA-256 matches the committed script. No serving implementation
changed. The run used `gpu048`, allocation `65250285`, an NVIDIA A16, Python
3.10.8, Torch 2.3.0+cu121, NumPy 2.2.6, and SciPy 1.15.3. The handoff reports
the four prerequisite CPU suites passing (5/7/9/13 tests) at the base revision.

TinyLlama dummy weights, one 16-token prompt, two generated tokens, one pipeline
stage, and one tensor-parallel worker exercised the real engine/worker path.
The selected actions were prefill 8, prefill 8, decode, decode. Admission
allocated one block; resident prefill and the first decode allocated none; the
second decode appended one block before completion freed both. The request
finished by length, all 15,612 central blocks were free again, and a final idle
call advanced the iteration without LP or worker execution. The retained
console reports `RESULT: PASS` and exit status 0. Existing enabled metrics mode
was used because disabled mode cannot complete an engine step; no framework
repair or plotting was added.

Mac review on `atmac.local`, Python 3.14.7, checked the script source, syntax,
commit-range whitespace, exact script/artifact hashes, and JSON consistency for
the actions, solver outcomes, completion, memory restoration, and idle call.
It did not rerun GPU execution. The worker profiling batch was not directly
observed; its one-32-token shape was established from source and configured
inputs. Successful execution is evidence for this bounded run, not a general
memory-profile guarantee.

This acceptance excludes semantic generation quality, reference numerical
agreement, mixed-batch GPU correctness, independent GPU-worker block-table
equality, runtime preemption, arbitrary-workload success, and performance.
The next proposed task is a tiny mixed-batch GPU check with one sampling method.
Full Phase F acceptance is not claimed.

The user also approved simplifying the commit workflow: one cohesive authorized
commit may include code, tests, documentation, and handoff evidence. Tested-code
provenance remains required under `AGENTS.md`; a separate evidence commit and
post-commit rerun are not required solely because a commit was created.

## Historical repository baseline (Phases A and B)

Phase B completed on 2026-09-02. It reproduced the controlled baseline workload
twice each for the existing `sarathi`, `slai_scheduler`, and `vllm` scheduler
providers at repository commit
`6f285d184546a87a0c57ab89581bf7e14a5d413f` (`Document Unity Phase A
baseline`). The `vllm` provider in these records is the vLLM-style policy
implemented inside LPServe/SLAI, not current upstream vLLM.

| Field | Recorded value |
|---|---|
| Unity path | `/home/atjoshi_umass_edu/LPServe` |
| Origin | `https://github.com/AtivJoshi/LPServe.git` |
| Branch | `main` |
| Phase A smoke-test revision | `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380` |
| Repository revision used for Phase B | `6f285d184546a87a0c57ab89581bf7e14a5d413f` |
| Revision subject | `Document Unity Phase A baseline` |

## Validated Unity environment

Phase B used one NVIDIA A16 GPU with the following verified identity:

| Component | Recorded value |
|---|---|
| GPU | NVIDIA A16 |
| GPU UUID | `GPU-8e5fef81-4f36-8118-7fbf-54a5847c5ad7` |
| GPU memory | 15,356 MiB |
| NVIDIA driver | 595.71.05 |

The Phase A dependency snapshot, excluding the editable LPServe entry, remained
unchanged. The full `pip freeze` hash is not claimed unchanged because the
editable repository entry changed with the commit.

## Phase B run record

Phase B artifacts are recorded in `docs/experiment_reference.md`. The six run
directories were:

- `benchmark_output/phase_b/sarathi_r1/2026-09-02_08-00-13-580183`
- `benchmark_output/phase_b/sarathi_r2/2026-09-02_08-24-19-684337`
- `benchmark_output/phase_b/slai_r1/2026-09-02_08-09-31-346862`
- `benchmark_output/phase_b/slai_r2/2026-09-02_08-33-45-436111`
- `benchmark_output/phase_b/vllm_r1/2026-09-02_08-17-18-214863`
- `benchmark_output/phase_b/vllm_r2/2026-09-02_08-39-01-733205`

Each repetition parent directory contains `console.log`. Its timestamped run directory contains `benchmark_config.yml`, `requests.json`, `replica_0/sequence_metrics.csv`, and `replica_0/batch_metrics.csv`.

## What Phase B verified

Phase B verified scheduler selection, the benchmark harness, metric generation,
and deterministic discrete behavior for a deliberately tiny homogeneous
synthetic workload. Each provider completed two successful runs with seed 42,
TinyLlama/TinyLlama-1.1B-Chat-v1.0, dummy model loading, one replica, tensor and
pipeline parallel degree 1, maximum model length 32, six synthetic requests,
fixed 16-token prefill and 4-token decode lengths, Poisson QPS 1,000,000,
maximum batch size 2, and GPU memory utilization 0.5.

All six `requests.json` files had SHA-256
`c9e4fe8f1ad7e64e3697a3ba9640fd60d476c1f4668ee0ba020ec6fa8d03de7d`.
Every run completed six requests in 15 iterations. The generated sequence
metrics recorded 16 prefill and 4 decode tokens per request, zero ignored
requests, zero restarts, and five request pauses per request. Those recorded
pauses must not be described as memory preemptions: batch-level scheduler
preemption counters were zero for both prefill and decode in every run.

The discrete batch structure was identical for both repetitions of every
provider: 15 total batches, consisting of three 32-token two-sequence prefill
batches and four 2-token two-sequence decode batches after each admitted pair.

Expected non-blocking console messages were observed in every run:

- the `torch_dtype` deprecation notice; and
- the bfloat16-to-float16 casting warning.

## Policy-specific Phase B settings

| Provider | Recorded limit settings |
|---|---|
| `sarathi` | chunk size 32, FCFS enabled, dynamic chunking disabled |
| `slai_scheduler` | token budget 32, FCFS disabled, fixed offset disabled, below-memory-limit offset 5, above-memory-limit offset 10, memory limit 0.96, user priority disabled, time between tokens 0.2, total decode limit 128 |
| `vllm` | vLLM-style maximum tokens per batch 32 |

These are existing policies running inside the same LPServe framework.

## Limitations and risks

The Phase B workload is intentionally tiny and homogeneous. Identical schedules
in this controlled workload do not establish policy equivalence, and the
recorded smoke-run execution times are not throughput, latency, or comparative
performance results.

Phase B did not perform an LP scheduler implementation, architecture audit,
large-scale benchmark sweep, solver integration, multi-GPU validation,
checkpoint-weight loading validation, or semantic generation-quality check.

## Phase A reference

Phase A remains the Unity environment and serving-stack smoke-test baseline
recorded in `docs/unity_setup.md`. It established that unmodified LPServe could
initialize TinyLlama with dummy weights on one A16 GPU, allocate KV cache, run
the existing Sarathi scheduler, and generate metrics for a one-request smoke
test. Phase A did not establish performance.
