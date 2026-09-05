# LP scheduler research project status

## Current phase

**Phase C closure / Phase D preparation as of 2026-09-05.**

The code-grounded Phase C architecture audit and its durable reference have been
completed. The audit was read-only: it did not implement the LP scheduler,
change serving code, add a solver, or execute new Phase C unit or GPU tests.
Subsequent mathematical/source clarification and normative design work produced
the Phase D--F design contract now present in the repository.

Phase D is the next implementation phase, but implementation has not started.
The immediate handoff is to resolve only the OPEN decisions required by the
affected Phase D layer, record those decisions in the normative design, and then
implement and validate the pure mathematical layer.

## Current repository and Phase C provenance

| Field | Recorded value |
|---|---|
| Local branch | `main` |
| Phase C audited code revision | `c3e014363dd50e1830d7c85c3d043eab69fdc9e5` (`gitignore update`) |
| Documentation integration revision | `6fbc046eca7c0cb7988f08690757d140a51a03e3` (`Initiated docs`) |
| HEAD when this status was updated | `6fbc046eca7c0cb7988f08690757d140a51a03e3` |

`docs/lpserve_scheduler_architecture.md` remains intentionally pinned to
`c3e0143`. The committed diff from `c3e0143` through `6fbc046` adds only
documentation: the architecture reference, scheduler design, updated research
context, mathematical source, and bibliography. No audited implementation,
configuration, or experiment-tooling path changed between those revisions.
This establishes that the checked-in scheduler-relevant code at `6fbc046`
matches the audited code baseline; it does not repin the historical audit.

At the time of this update, the local working tree additionally contains
uncommitted documentation work: untracked `PROJECT_GUIDE.md` and `AGENTS.md`,
plus this modification to `docs/project_status.md`. No source-code modification
is reported by the working tree. These statements are a dated snapshot and must
be rechecked after later local or committed changes.

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
- `PROJECT_GUIDE.md` and `AGENTS.md` — local Phase D readiness and agent
  navigation infrastructure. Both remain uncommitted at this snapshot and are
  not Phase C code-audit evidence.

### Phase C evidence boundary

Phase C was primarily static source inspection, supplemented only by explicitly
identified Phase B observations. It did not add or run a new automated test
suite, reproduce the audited defects at runtime, perform a new GPU validation,
or establish performance. It did not implement LP problem construction, solver
integration, integer extraction, LPServe state mapping, native LP action
execution, or fixes for the documented framework blockers.

## Phase D handoff

Phase D is limited to a pure, framework-light, independently CPU-testable
mathematical layer containing:

- LP problem representation and construction;
- a solver adapter/interface with explicit result handling;
- relaxed-solution validation;
- deterministic integer extraction;
- integer-plan validation; and
- focused CPU and synthetic tests required by `docs/lp_scheduler_design.md`
  Section 15.

Phase D must not import or accept mutable LPServe scheduler, `Sequence`, block
manager, engine, queue, or GPU state; perform LPServe mutations; or prematurely
implement the Phase E mapper or Phase F executor.

Before affected Phase D implementation proceeds:

1. resolve the necessary Phase-D-blocking OPEN decisions explicitly rather than
   inventing defaults;
2. update the relevant normative documentation with the approved decisions and
   their test implications; and
3. create a clean Phase C / Phase D handoff commit that records the reviewed
   documentation state.

No OPEN decision is resolved by this status update, and no Phase D
implementation is claimed to have started.

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
