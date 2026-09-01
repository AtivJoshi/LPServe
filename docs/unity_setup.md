# LPServe setup on the Unity HPC cluster

This document records the Phase A environment that successfully ran unmodified
LPServe on one NVIDIA A16 GPU on the Unity cluster. It is an environment and
serving-stack smoke test, not a performance benchmark.

## Validated baseline

- Validation date: 2026-09-01
- Repository: `https://github.com/AtivJoshi/LPServe.git`
- Repository path on Unity: `/home/atjoshi_umass_edu/LPServe`
- Branch: `main`
- Commit: `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380`
- Python environment: `/home/atjoshi_umass_edu/LPServe/env`
- GPU: NVIDIA A16, 15,356 MiB, compute capability 8.6
- NVIDIA driver: 595.71.05
- Loaded CUDA toolkit: CUDA 12.1.1
- PyTorch: 2.3.0+cu121

At the validated commit, `AtivJoshi/LPServe:main` and
`agrimUT/SLAI:main` pointed to the same commit. Only the `origin` remote was
configured in the Unity checkout. The successful run left the Git working tree
clean.

## Important operating rules

- Start a `tmux` session before requesting resources.
- Never run model initialization, CUDA checks, or benchmarks on a login node.
- A Unity GPU hostname such as `gpu048` belongs to one allocation and is not a
  reusable hostname.
- The `gpu-preempt` partition is preemptible. Preserve logs under the repository
  or another persistent filesystem.
- Run `ray stop` after a failed or completed benchmark to remove local Ray
  processes.
- Do not use the smoke-test timings as performance results.

## 1. Obtain an A16 allocation

Log in and start `tmux`:

```bash
ssh unity
tmux new -s lpserve
```

The validated resource request was:

```bash
salloc \
  --partition=gpu-preempt \
  --gres=gpu:a16:1 \
  --cpus-per-task=8 \
  --mem=16G \
  --time=04:00:00 \
  --job-name=lpserve-a16
```

The successful Phase A allocation was job `63889100` on `gpu048`. Future jobs
will generally receive different job IDs and may receive different nodes.

Confirm that the shell is inside the allocation:

```bash
echo "$SLURM_JOB_ID"
hostname --fqdn
scontrol show job "$SLURM_JOB_ID"
nvidia-smi
```

## 2. Verify the repository state

```bash
cd ~/LPServe
pwd -P
git remote -v
git branch --show-current
git rev-parse HEAD
git status --short --branch --untracked-files=all
```

For the validated baseline, these commands should show branch `main`, commit
`5098a7aba05e3edbcfa3a509d6cc9cd248fc4380`, and no working-tree entries below
the branch-status line.

Do not modify scheduler or serving source code while reproducing Phase A.

## 3. Load the compiler, Python, and CUDA modules

```bash
module purge
module load uri/main
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
module -t list
```

The validated toolchain was:

| Component | Version |
|---|---:|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.8.0-136-generic |
| CPU | AMD EPYC 9354 32-Core Processor |
| CPU feature | AVX-512 present |
| Python module | 3.10.8, GCCcore 12.2.0 build |
| GCC | 12.2.0 |
| CUDA toolkit | 12.1.1 |
| `nvcc` | 12.1.105 |

`nvidia-smi` reported CUDA 13.2. That value is the maximum CUDA compatibility
advertised by the installed driver; it is not the toolkit used to build or run
LPServe. The loaded toolkit and PyTorch runtime are CUDA 12.1.

## 4. Create a fresh environment

The repository ignores both `env/` and `.venv/`. Phase A used `env/`:

```bash
cd ~/LPServe
uv venv --python "$(command -v python)" env
source env/bin/activate

python --version
python -c "import sys; print(sys.executable); print(sys.prefix)"
```

Expected environment path:

```text
/home/atjoshi_umass_edu/LPServe/env
```

## 5. Install the known-good dependencies

Install the CUDA 12.1 build of PyTorch first because LPServe imports PyTorch
during its native-extension build:

```bash
uv pip install torch==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install `pip` into the otherwise unseeded `uv` environment, then install the
compatibility pins established during Phase A:

```bash
uv pip install pip==26.2.1

python -m pip install \
  setuptools==84.0.0 \
  transformers==4.57.6 \
  nvidia-ml-py==13.595.45
```

Install LPServe in editable mode using `pip` and the repository's documented
FlashInfer wheel index:

```bash
python -m pip install -e . \
  --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
```

Do not replace this last command with `uv pip install -e .` for this baseline.
During Phase A, `uv` rejected newer entries in the legacy FlashInfer index
because their metadata used the renamed distribution `flashinfer-python`.
`pip` selected the compatible wheel
`flashinfer-0.2.0.post1+cu121torch2.3`.

Check the resulting environment:

```bash
python -m pip check

python -c "import torch; import transformers; print(torch.__version__); print(torch.version.cuda); print(transformers.__version__)"

python -c "import sarathi; import flashinfer; import sarathi.pos_encoding_ops; import sarathi.layernorm_ops; import sarathi.activation_ops; print('LPServe, FlashInfer, and native extensions imported successfully')"
```

The validated key package versions were:

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

The complete validated snapshot contained 172 distributions. Its
`pip-freeze.txt` SHA-256 was:

```text
7fd7969c61eaa1699f279b00b50ad567136fee29d626fa5af263dac2a2b5df1a
```

The snapshot was created with:

```bash
python -m pip freeze \
  > benchmark_output/phase_a_smoke_known_good/pip-freeze.txt
sha256sum benchmark_output/phase_a_smoke_known_good/pip-freeze.txt
```

Preserve that file with the project records if exact transitive dependency
reconstruction is required.

## 6. Install Chrome for Plotly/Kaleido

Kaleido 1.x requires a separate Chrome or Chromium executable for static-image
generation. Unity did not provide a browser executable or module. Phase A
installed Chrome for Testing inside the virtual environment:

```bash
mkdir -p "$VIRTUAL_ENV/plotly-chrome"
plotly_get_chrome -y --path "$VIRTUAL_ENV/plotly-chrome"
```

Known-good browser:

```text
/home/atjoshi_umass_edu/LPServe/env/plotly-chrome/chrome-linux64/chrome
Google Chrome for Testing 142.0.7444.175
```

The browser directory occupied 356 MiB. Set its path in every shell that runs a
metrics-enabled benchmark:

```bash
export BROWSER_PATH="$VIRTUAL_ENV/plotly-chrome/chrome-linux64/chrome"
```

Optional image-export check:

```bash
mkdir -p benchmark_output/phase_a_smoke

python -c "import plotly.express as px; fig=px.scatter(x=[0,1], y=[0,1]); fig.write_image('benchmark_output/phase_a_smoke/kaleido_probe.png'); print('Kaleido image export completed')"
```

See the Plotly static-image documentation for the browser requirement:
<https://plotly.com/python/static-image-export/>.

## 7. Verify CUDA and NVML on the allocated node

```bash
python -c "import torch; print('torch:', torch.__version__); print('compiled CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('capability:', torch.cuda.get_device_capability(0))"
```

Expected key results are CUDA availability `True`, GPU `NVIDIA A16`, and compute
capability `(8, 6)`.

Verify the NVML calls imported by LPServe:

```bash
python -c "from importlib.metadata import version; from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates; print('nvidia-ml-py:', version('nvidia-ml-py')); nvmlInit(); h=nvmlDeviceGetHandleByIndex(0); u=nvmlDeviceGetUtilizationRates(h); print('NVML GPU utilization:', u.gpu); print('NVML memory utilization:', u.memory); nvmlShutdown(); print('NVML check completed')"
```

## 8. Run the known-good end-to-end smoke test

This test uses LPServe's existing benchmark entry point. It constructs a
TinyLlama model with dummy/random weights, creates a one-GPU Ray worker,
initializes FlashAttention and the KV cache, runs the existing Sarathi
scheduler, processes one 16-token prefill followed by four decode steps, and
writes metric plots.

`load_format=dummy` avoids downloading approximately 1.1 billion checkpoint
parameters. The model configuration and tokenizer may still be downloaded from
Hugging Face. Dummy weights make this an execution-path test, not a semantic
generation-quality test.

```bash
ray stop
mkdir -p benchmark_output/phase_a_smoke_known_good

BROWSER_PATH="$VIRTUAL_ENV/plotly-chrome/chrome-linux64/chrome" \
python -m sarathi.benchmark.main \
  --output_dir ./benchmark_output/phase_a_smoke_known_good \
  --log_level info \
  --write_json_trace false \
  --write_chrome_trace false \
  --write_metrics true \
  --gpu_memory_utilization 0.5 \
  --time_limit 300 \
  --cluster_num_replicas 1 \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model_tensor_parallel_degree 1 \
  --model_pipeline_parallel_degree 1 \
  --model_max_model_len 64 \
  --model_load_format dummy \
  --model_attention_backend flash_attention \
  --request_generator_provider synthetic \
  --synthetic_request_generator_length_provider fixed \
  --synthetic_request_generator_interval_provider static \
  --synthetic_request_generator_num_requests 1 \
  --fixed_request_length_generator_prefill_tokens 16 \
  --fixed_request_length_generator_decode_tokens 4 \
  --replica_scheduler_provider sarathi \
  --replica_scheduler_max_batch_size 1 \
  --sarathi_scheduler_chunk_size 16 \
  --sarathi_scheduler_fcfs true \
  --sarathi_scheduler_enable_dynamic_chunking_schedule false \
  --metrics_store_enable_request_outputs false \
  --metrics_store_keep_individual_batch_metrics true \
  2>&1 | tee benchmark_output/phase_a_smoke_known_good/console.log

lpserve_status=${PIPESTATUS[0]}
echo "LPServe exit status: $lpserve_status"
```

The validated run printed `LPServe exit status: 0`. Its concise evidence can be
extracted with:

```bash
smoke_log=benchmark_output/phase_a_smoke_known_good/console.log

rg -n \
  'Model initialized|# GPU blocks|Scheduler initialised|processed requests|exiting after processing|Traceback|ERROR|Exception' \
  "$smoke_log"
```

The successful log showed:

- model initialization on worker 0 and `cuda:0`;
- 15,615 GPU KV-cache blocks;
- successful initialization of the existing Sarathi scheduler;
- one of one requests completed;
- 16 prefill tokens and 4 decode tokens in the generated metrics; and
- no traceback, error, or exception markers.

At verification time, the output directory contained 114 files and occupied
approximately 1.4 MiB, before adding the package-freeze snapshot.

## 9. Known compatibility findings

These findings apply to commit
`5098a7aba05e3edbcfa3a509d6cc9cd248fc4380`. No source files were changed to
work around them.

### Transformers must remain below major version 5

LPServe declares `transformers >= 4.37.0` without an upper bound. The initial
resolver selected Transformers 5.16.1, which disabled PyTorch integration
because that release required PyTorch 2.5 or newer. LPServe pins PyTorch 2.3.
The known-good environment uses Transformers 4.57.6.

### `setuptools` is required at runtime

A fresh `uv` environment does not contain `setuptools`. FlashInfer reaches it
through `torch.utils.cpp_extension`, so the import path failed until
`setuptools` was installed explicitly.

### LPServe omits its NVML Python dependency

`sarathi/worker/base_worker.py` imports the `pynvml` module, but LPServe's
requirements do not declare an NVML binding. The known-good environment uses
NVIDIA's `nvidia-ml-py==13.595.45` distribution.

### Metrics cannot be disabled cleanly

With `write_metrics=false`, `MetricsStore.__init__` returns before initializing
`_keep_individual_batch_metrics`, but the benchmark later calls `reset()` and
accesses that missing attribute. Keep `write_metrics=true` for this baseline.

Relevant source:
<https://github.com/AtivJoshi/LPServe/blob/5098a7aba05e3edbcfa3a509d6cc9cd248fc4380/sarathi/metrics/metrics_store.py>.

### Request-output serialization is broken

`RequestOutput` defines the request identifier as `seq_id`, while
`MetricsStore._store_request_outputs()` attempts to sort on `request_id`.
Therefore, the known-good smoke command uses
`metrics_store_enable_request_outputs=false` while retaining ordinary metrics.

Relevant sources:

- <https://github.com/AtivJoshi/LPServe/blob/5098a7aba05e3edbcfa3a509d6cc9cd248fc4380/sarathi/core/datatypes/request_output.py>
- <https://github.com/AtivJoshi/LPServe/blob/5098a7aba05e3edbcfa3a509d6cc9cd248fc4380/sarathi/metrics/metrics_store.py>

### Non-blocking warnings

The successful run emitted a Transformers deprecation warning for
`torch_dtype` and reported that the model's bfloat16 configuration was cast to
float16. Neither warning prevented successful model initialization or request
completion on the A16.

## 10. Shut down the local runtime

```bash
ray stop
```

Exit the allocation when no further GPU work is required.
