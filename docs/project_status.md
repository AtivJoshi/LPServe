# LP scheduler research project status

## Current phase

**Phase A: complete as of 2026-09-01.**

Unmodified LPServe completed the smallest useful end-to-end GPU smoke run on
the Unity HPC cluster. The repository state, Unity allocation, software
environment, dependency versions, compatibility settings, command, and output
evidence have been recorded.

No LP scheduler implementation, scheduler-algorithm modification, architecture
audit, performance comparison, benchmark sweep, solver integration, or utility
and memory-coefficient design was performed in Phase A.

## Repository baseline

| Field | Recorded value |
|---|---|
| Unity path | `/home/atjoshi_umass_edu/LPServe` |
| Origin | `https://github.com/AtivJoshi/LPServe.git` |
| Branch | `main` |
| Commit | `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380` |
| Working tree after smoke run | Clean |
| Tracking state | `main...origin/main` |
| Upstream comparison | Identical to `agrimUT/SLAI:main` at this commit |

Only the `origin` remote was configured in the Unity checkout. No upstream
remote was added during Phase A.

## Validated Unity environment

The successful run used Slurm job `63889100` on `gpu048`. The hostname records
the observed allocation only; it is not a fixed deployment target.

| Component | Known-good value |
|---|---|
| Partition | `gpu-preempt` |
| GPU request | `--gres=gpu:a16:1` |
| CPUs and memory | 8 CPUs, 16 GiB RAM |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.8.0-136-generic |
| CPU | AMD EPYC 9354; AVX-512 present |
| GPU | NVIDIA A16, 15,356 MiB |
| Compute capability | 8.6 |
| Driver | 595.71.05 |
| Driver-advertised CUDA compatibility | 13.2 |
| Loaded CUDA toolkit | 12.1.1 |
| `nvcc` | 12.1.105 |
| GCC | 12.2.0 |
| Python | 3.10.8 |
| Environment | `/home/atjoshi_umass_edu/LPServe/env` |

The driver compatibility level and the loaded toolkit must not be conflated.
LPServe and PyTorch used CUDA 12.1.

## Key dependency versions

| Distribution | Version |
|---|---:|
| pip | 26.2.1 |
| setuptools | 84.0.0 |
| sarathi | 0.1.7 |
| torch | 2.3.0+cu121 |
| transformers | 4.57.6 |
| flashinfer | 0.2.0.post1+cu121torch2.3 |
| vllm-flash-attn | 2.5.9 |
| ray | 2.58.0 |
| numpy | 2.2.6 |
| nvidia-ml-py | 13.595.45 |
| plotly | 7.0.0 |
| kaleido | 1.4.0 |
| choreographer | 1.3.0 |
| ninja | 1.13.2 |
| Chrome for Testing | 142.0.7444.175 |

`python -m pip check` reported no broken requirements. The full known-good
snapshot contained 172 distributions and had SHA-256:

```text
7fd7969c61eaa1699f279b00b50ad567136fee29d626fa5af263dac2a2b5df1a
```

Snapshot location on Unity:

```text
benchmark_output/phase_a_smoke_known_good/pip-freeze.txt
```

## Successful smoke-test configuration

| Setting | Value |
|---|---|
| Existing entry point | `python -m sarathi.benchmark.main` |
| Model configuration | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Weight format | `dummy` |
| Tensor parallel degree | 1 |
| Pipeline parallel degree | 1 |
| Maximum model length | 64 |
| Attention backend | `flash_attention` |
| Scheduler | Existing `sarathi` scheduler |
| Scheduler chunk size | 16 |
| Maximum batch size | 1 |
| Request generator | Synthetic, fixed length, static arrival |
| Number of requests | 1 |
| Prefill tokens | 16 |
| Decode tokens | 4 |
| GPU-memory utilization limit | 0.5 |
| Metrics | Enabled |
| Request-output JSON | Disabled because of an upstream field-name defect |

The command is recorded in `docs/unity_setup.md`.

## Success evidence

The known-good run produced the following evidence:

- LPServe selected worker 0 on `cuda:0`.
- The TinyLlama model initialized successfully.
- LPServe created 15,615 GPU KV-cache blocks.
- The existing Sarathi scheduler initialized successfully.
- The progress indicator reached one of one processed requests.
- Metrics recorded 20 total tokens: 16 prefill tokens and 4 decode tokens.
- The command printed `LPServe exit status: 0`.
- The successful log contained no `Traceback`, `ERROR`, or `Exception` marker.
- At verification time, the output directory contained 114 files and occupied
  approximately 1.4 MiB, before adding `pip-freeze.txt`.
- `git status --short --branch --untracked-files=all` showed only
  `## main...origin/main` because the environment and benchmark-output paths
  are ignored.

Successful run artifacts on Unity:

```text
benchmark_output/phase_a_smoke_known_good/console.log
benchmark_output/phase_a_smoke_known_good/pip-freeze.txt
benchmark_output/phase_a_smoke_known_good/plots/
```

## What the smoke test establishes

The successful test establishes that the following unmodified LPServe path
works on the recorded Unity environment:

1. configuration and tokenizer loading;
2. Ray initialization and worker placement;
3. CUDA device selection;
4. TinyLlama model construction with random dummy weights;
5. LPServe native-extension and FlashAttention loading;
6. GPU KV-cache allocation;
7. existing Sarathi scheduling;
8. one prefill execution and four decode executions; and
9. ordinary metric aggregation and Plotly/Kaleido image generation.

It does not establish checkpoint-weight loading, semantic generation quality,
multi-GPU execution, large-model capacity, sustained-load stability, or
performance. The recorded latency is not a benchmark result.

## Environment and compatibility findings

No LPServe source files were changed. The following environment adjustments
were necessary:

1. Use `pip`, rather than `uv`, for LPServe's editable installation against the
   legacy FlashInfer wheel index.
2. Pin Transformers below major version 5; the open-ended repository
   requirement otherwise selected a release incompatible with PyTorch 2.3.
3. Install `setuptools` explicitly for the FlashInfer/PyTorch extension import
   path.
4. Install `nvidia-ml-py==13.595.45`; LPServe imports `pynvml` but does not
   declare the distribution.
5. Install Chrome for Testing under the virtual environment and provide
   `BROWSER_PATH` for Kaleido 1.4.0.

Two source defects were observed and left unmodified:

- `write_metrics=false` leaves `MetricsStore` partially initialized, but the
  benchmark subsequently calls `reset()` and accesses a missing attribute.
- `RequestOutput` defines `seq_id`, while request-output serialization expects
  `request_id`. The successful run therefore retained metrics but set
  `metrics_store_enable_request_outputs=false`.

These are baseline compatibility findings, not LP-scheduler changes.

## Source-change status

Phase A introduced no changes to:

- scheduler behavior;
- scheduling policies;
- model execution;
- request admission or preemption;
- LP formulation or extraction;
- utility weights;
- memory coefficients; or
- solver integration.

The repository remained at the original commit throughout the successful run.

## Phase boundary

Phase A is closed. Later work must begin as a separately scoped phase. In
particular, this status does not authorize:

- implementation of the LP scheduler;
- an LP-to-LPServe architecture mapping or audit;
- baseline-comparison experiments;
- scheduler-performance claims;
- benchmark sweeps; or
- solver-overhead optimization.

Before later experiments, preserve the successful console log and exact package
snapshot, and reproduce the exit-status-zero smoke test after any intentional
environment or repository change.
