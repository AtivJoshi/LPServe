# LP scheduler research project status

## Current phase

**Phase D accepted / Phase E preparation as of 2026-09-20.**

The pure, framework-independent LP-relaxation mathematical layer is implemented
in `lp_relaxation_scheduler.py` and accepted for the scoped MVP. It constructs
and solves the documented continuous relaxation with SciPy/HiGHS, independently
validates the relaxed solution, performs deterministic feasible integer
extraction, validates the resulting plan, and returns structured success or
failure without importing LPServe runtime objects.

Phase E read-only state mapping is the next implementation phase. Phase E must
construct coherent immutable inputs for the accepted mathematical layer without
mutating queues, statuses, block state, prompt progress, or GPU state. Phase F
execution and live scheduler integration remain out of scope until their design
gates are satisfied.

## Current repository and provenance

| Field | Recorded value |
|---|---|
| Local branch | `main` |
| Phase C audited code revision | `c3e014363dd50e1830d7c85c3d043eab69fdc9e5` (`gitignore update`) |
| Initial Phase D implementation | `f406eeaac7bafc4478304744c9187316b75b8f67` |
| Zero-valued extraction-tie contract fix | `32036f6a0ddb9bf2b55fc81f48967d61d558b4c2` |
| Regression optimality correction | `db3a64afbe1aec2b6b11d477ced4f2f888288a06` |
| Accepted implementation/evidence HEAD | `67a336222a6ec2ac53d932e7928c597dd4ccbe23` |

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

## Phase D acceptance and Phase E handoff

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

No Phase E mapping, Phase F execution, live integration, GPU work, queue or
block mutation, framework repair, approximation guarantee, or permanent
utility/capacity/reserve/decode-charge policy was established. Remaining OPEN
decisions in `docs/lp_scheduler_design.md` remain OPEN unless explicitly
recorded there.

The next permitted implementation work is the smallest read-only Phase E mapper
described by the design. It must preserve the accepted mathematical interface,
fail visibly on incoherent snapshots, and perform no scheduler or serving-state
mutation.

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
