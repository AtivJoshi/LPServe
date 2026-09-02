# Phase B experiment reference

This document is the durable reproducibility record for Phase B baseline
reproduction. It summarizes evidence from the saved run artifacts and console
logs without copying the full generated configurations.

## Provenance

| Field | Recorded value |
|---|---|
| Phase | B |
| Validation date | 2026-09-02 |
| Repository commit | `6f285d184546a87a0c57ab89581bf7e14a5d413f` |
| Commit subject | `Document Unity Phase A baseline` |
| GPU | NVIDIA A16 |
| GPU UUID | `GPU-8e5fef81-4f36-8118-7fbf-54a5847c5ad7` |
| Driver | 595.71.05 |
| GPU memory | 15,356 MiB |

The Phase A dependency snapshot, excluding the editable LPServe entry, remained
unchanged. Do not claim that the full `pip freeze` hash remained unchanged.

## Run paths

| Provider | Repetition | Run directory | Console-reported smoke-run time |
|---|---:|---|---:|
| `sarathi` | 1 | `benchmark_output/phase_b/sarathi_r1/2026-09-02_08-00-13-580183` | 0.55 seconds |
| `sarathi` | 2 | `benchmark_output/phase_b/sarathi_r2/2026-09-02_08-24-19-684337` | 0.35 seconds |
| `slai_scheduler` | 1 | `benchmark_output/phase_b/slai_r1/2026-09-02_08-09-31-346862` | 0.35 seconds |
| `slai_scheduler` | 2 | `benchmark_output/phase_b/slai_r2/2026-09-02_08-33-45-436111` | 0.35 seconds |
| `vllm` | 1 | `benchmark_output/phase_b/vllm_r1/2026-09-02_08-17-18-214863` | 0.35 seconds |
| `vllm` | 2 | `benchmark_output/phase_b/vllm_r2/2026-09-02_08-39-01-733205` | 0.35 seconds |

The timing values above are console-reported smoke-run observations from
`benchmark_runner.py`; they are not performance results.

Each run directory contains:

- `benchmark_config.yml`
- `requests.json`
- `replica_0/sequence_metrics.csv`
- `replica_0/batch_metrics.csv`

Each repetition root also contains `console.log`.

## Common workload configuration

Every generated `benchmark_config.yml` recorded the following common settings:

| Setting | Value |
|---|---|
| seed | 42 |
| model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| model load format | `dummy` |
| replicas | 1 |
| tensor parallel degree | 1 |
| pipeline parallel degree | 1 |
| maximum model length | 32 |
| request generator | synthetic |
| synthetic requests | 6 |
| length generator | fixed |
| prefill tokens per request | 16 |
| decode tokens per request | 4 |
| interval generator | Poisson |
| Poisson QPS | 1,000,000 |
| maximum batch size | 2 |
| GPU memory utilization | 0.5 |

All six `requests.json` files had SHA-256:

```text
c9e4fe8f1ad7e64e3697a3ba9640fd60d476c1f4668ee0ba020ec6fa8d03de7d
```

The saved request payloads contained the same six synthetic requests, each with
16 prefill tokens, 4 decode tokens, and time between tokens 0.2.

## Policy-specific settings

| Provider | Relevant generated settings |
|---|---|
| `sarathi` | `sarathi_scheduler_chunk_size: 32`; `sarathi_scheduler_fcfs: true`; `sarathi_scheduler_enable_dynamic_chunking_schedule: false` |
| `slai_scheduler` | `slai_scheduler_token_budget: 32`; `slai_scheduler_fcfs: false`; `slai_scheduler_fixed_offset: false`; `slai_scheduler_below_memory_limit_offset: 5`; `slai_scheduler_above_memory_limit_offset: 10`; `slai_scheduler_memory_limit: 0.96`; `slai_scheduler_user_priority: false`; `slai_scheduler_time_between_tokens: 0.2`; `slai_scheduler_limit_total_decodes: 128` |
| `vllm` | `vllm_scheduler_max_tokens_in_batch: 32` |

The `vllm` provider is the vLLM-style policy implemented inside LPServe/SLAI,
not current upstream vLLM.

## Validation results

All six runs were observed to return `LPServe exit status: 0` and completed six
requests in 15 iterations. Five saved console logs contain the exit-status
wrapper line. For `sarathi_r1`, the status was printed after the `tee` pipeline
and therefore was not retained in `console.log`; that log independently records
normal benchmark completion and metrics generation.

Every `sequence_metrics.csv` contained six rows with:

- `request_num_prefill_tokens` equal to 16;
- `request_num_decode_tokens` equal to 4;
- `request_num_tokens` equal to 20;
- `request_num_restarts` equal to 0;
- `request_num_pauses` equal to 5; and
- `request_num_ignored` equal to 0.

The five recorded request pauses per request do not by themselves establish
memory preemption. In every `batch_metrics.csv`, both
`batch_num_preempted_seq_prefill` and `batch_num_preempted_seq_decode` summed to
zero, and every sequence recorded zero restarts. The precise meaning of the
pause metric remains to be established from the LPServe implementation during
the architecture audit.

Every `batch_metrics.csv` contained the same discrete batch structure:

- 15 total batches;
- batch IDs 0, 5, and 10 were 32-token, two-sequence prefill batches; and
- each admitted pair was followed by four 2-token, two-sequence decode batches.

Every console log contained the expected non-blocking messages:

- `torch_dtype` is deprecated; and
- `Casting torch.bfloat16 to torch.float16`.

Phase B therefore verifies scheduler selection, the benchmark harness, metric
generation, and deterministic discrete behavior for this controlled workload.

## Limitations

This workload is intentionally tiny and homogeneous. Identical schedules in
these runs do not establish policy equivalence. Do not draw throughput, latency,
or comparative-performance conclusions from these smoke runs.

The recorded providers are existing policies running inside the same LPServe
framework. Phase B did not implement or validate a new LP scheduler, audit the
architecture for Phase C, test current upstream vLLM, exercise multi-GPU
execution, or validate checkpoint-weight loading or generation quality.
