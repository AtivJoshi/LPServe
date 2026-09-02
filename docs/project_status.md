# LP scheduler research project status

## Current phase

**Phase B: complete as of 2026-09-02.**

Phase B reproduced the controlled baseline workload twice each for the existing `sarathi`, `slai_scheduler`, and `vllm` scheduler providers at repository commit `6f285d184546a87a0c57ab89581bf7e14a5d413f` (`Document Unity Phase A baseline`). The `vllm` provider in these records is the vLLM-style policy implemented inside LPServe/SLAI, not current upstream vLLM.

Phase C is the next phase.

## Repository baseline

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
