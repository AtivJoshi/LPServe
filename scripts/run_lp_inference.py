"""Run one to three text prompts through LPServe's registered LPScheduler.

Usage (from the repository root, with the root on PYTHONPATH so the Ray
worker can import the LP modules):

    python -B scripts/run_lp_inference.py \\
        --model-path <existing local snapshot> \\
        --prompts-file <JSON file> \\
        --output-dir <new results directory>

The prompts file is a JSON array of one to three nonempty strings. Each string
is used exactly as given: no chat template, no trimming. Input position is the
user-facing request identifier; ``results.json`` maps it to the engine's
sequence ID. Prompts are encoded with the engine tokenizer's ``encode`` (the
same call ``BaseLLMEngine.add_request`` makes); every prompt must encode to at
least one token and leave room for the output tokens within the model length.
Nothing is truncated or adjusted.

Fixed provisional configuration (scoped to this program's first version, not
research policy or a guarantee for other workloads; saved in summary.json):

- Model and tokenizer: the given local TinyLlama/TinyLlama-1.1B-Chat-v1.0
  snapshot, real weights (load_format "auto"), float16, flash_attention,
  tokenizer mode "auto", trust_remote_code True.
- Tensor and pipeline parallelism 1; seed 42; model length 32; block size 16;
  GPU memory utilization 0.5.
- LP scheduler: resident limit 3, b_max 96, c_max 8, s_max 2, planning reserve
  1 block, decode charge conservative_one_block_v1, decode/prefill-token
  utility and preemption penalty 1/1/1, numerical policy lp_relaxation_mvp_v1
  (tolerances 1e-7, 1e-6, 1e-9, 1e-9).
- Greedy sampling: temperature 0, 4 output tokens, ignore_eos True, no stop
  strings.
- Metrics: enabled mode with every optional output off; never plotted.

All requests are submitted before the first step. A scheduling or execution
failure stops the run (no retry, fallback, or engine reuse); selected
preemption is unsupported by the LP executor and fails visibly. Exit status is
0 on success; 1 on a run failure, including a prompt that fails the token
checks (summary.json records it); and 2 on invalid arguments or a malformed
prompts file (nothing is run or written).
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.json"
MAX_PROMPTS = 3

# Provisional configuration; see the module documentation.
MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "auto"
DTYPE = "float16"
ATTENTION_BACKEND = "flash_attention"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
SEED = 42
MAX_MODEL_LEN = 32
BLOCK_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.5
TOKENIZER_MODE = "auto"
TRUST_REMOTE_CODE = True
MAX_NUM_SEQS = 3
B_MAX = 96
C_MAX = 8
S_MAX = 2
MEMORY_RESERVE = 1
DECODE_POLICY_ID = "conservative_one_block_v1"
DECODE_UTILITY = 1.0
PREFILL_TOKEN_UTILITY = 1.0
PREEMPTION_PENALTY = 1.0
NUMERICAL_POLICY_FIELDS = dict(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)
TEMPERATURE = 0.0
MAX_TOKENS = 4
IGNORE_EOS = True

SETTINGS = dict(
    model_repository=MODEL_REPO, load_format=LOAD_FORMAT, dtype=DTYPE,
    attention_backend=ATTENTION_BACKEND,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    pipeline_parallel_size=PIPELINE_PARALLEL_SIZE, seed=SEED,
    max_model_len=MAX_MODEL_LEN, block_size=BLOCK_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
    scheduler="LPScheduler", max_num_seqs=MAX_NUM_SEQS, b_max=B_MAX,
    c_max=C_MAX, s_max=S_MAX, memory_reserve=MEMORY_RESERVE,
    decode_memory_policy_id=DECODE_POLICY_ID, decode_utility=DECODE_UTILITY,
    prefill_token_utility=PREFILL_TOKEN_UTILITY,
    preemption_penalty=PREEMPTION_PENALTY,
    numerical_policy=NUMERICAL_POLICY_FIELDS,
    sampling=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                  ignore_eos=IGNORE_EOS, stop=[]),
    metrics_mode="enabled, all optional outputs off, never plotted",
    max_prompts=MAX_PROMPTS,
)

# Files whose hashes identify the tested code in summary.json.
PROVENANCE_FILES = [
    "scripts/run_lp_inference.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/engine/base_llm_engine.py",
    "sarathi/config.py",
    "lp_relaxation_scheduler.py",
    "lpserve_state_mapping.py",
    "lpserve_plan_execution.py",
]


class InputError(ValueError):
    """Invalid prompts file or prompt; raised before any request is added."""


class RunError(RuntimeError):
    """The run did not complete as one finished output per request."""


def read_prompts(path):
    """Return the prompt strings from a JSON array of 1..MAX_PROMPTS
    nonempty strings, unchanged."""
    try:
        prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise InputError(f"cannot read prompts file {path}: {err}") from err
    if not isinstance(prompts, list):
        raise InputError("prompts file must hold a JSON array, got "
                         f"{type(prompts).__name__}")
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise InputError(f"expected 1 to {MAX_PROMPTS} prompts, got "
                         f"{len(prompts)}")
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise InputError(f"prompt {index} must be a string, got "
                             f"{type(prompt).__name__}")
        if not prompt.strip():
            raise InputError(f"prompt {index} is empty or whitespace-only")
    return prompts


def encode_prompts(tokenizer, prompts):
    """Encode each prompt as ``add_request`` does and check that it fits."""
    encoded = []
    for index, prompt in enumerate(prompts):
        ids = list(tokenizer.encode(prompt))
        if not ids:
            raise InputError(f"prompt {index} encodes to no tokens")
        if len(ids) + MAX_TOKENS > MAX_MODEL_LEN:
            raise InputError(
                f"prompt {index} encodes to {len(ids)} tokens; with "
                f"{MAX_TOKENS} output tokens it exceeds model length "
                f"{MAX_MODEL_LEN}")
        encoded.append(ids)
    return encoded


def create_engine(model_path, metrics_dir):
    """Build a ``BaseLLMEngine`` with the registered ``LPScheduler``."""
    import lp_relaxation_scheduler as lrs
    from sarathi.config import (CacheConfig, LPSchedulerConfig, MetricsConfig,
                                ModelConfig, ParallelConfig)
    from sarathi.core.scheduler.lp_scheduler import LPScheduler
    from sarathi.engine.base_llm_engine import BaseLLMEngine

    model_config = ModelConfig(
        model=model_path, tokenizer=model_path, tokenizer_mode=TOKENIZER_MODE,
        trust_remote_code=TRUST_REMOTE_CODE, download_dir=None,
        load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED, revision=None,
        max_model_len=MAX_MODEL_LEN, attention_backend=ATTENTION_BACKEND,
    )
    cache_config = CacheConfig(block_size=BLOCK_SIZE,
                               gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
    parallel_config = ParallelConfig(
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE)
    scheduler_config = LPSchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
        num_pipeline_stages=PIPELINE_PARALLEL_SIZE, b_max=B_MAX, c_max=C_MAX,
        s_max=S_MAX, memory_reserve=MEMORY_RESERVE,
        decode_memory_policy_id=DECODE_POLICY_ID,
        decode_utility=DECODE_UTILITY,
        prefill_token_utility=PREFILL_TOKEN_UTILITY,
        preemption_penalty=PREEMPTION_PENALTY,
        numerical_policy=lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS),
    )
    # The disabled metrics mode cannot complete engine.step() (see the
    # single-request GPU handoff), so the enabled mode is used with every
    # optional output off.
    metrics_config = MetricsConfig(
        replica_id=0, write_metrics=True, output_dir=str(metrics_dir),
        wandb_project=None, wandb_group=None, wandb_run_name=None,
        wandb_sweep_id=None, wandb_run_id=None,
        enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
        enable_chrome_trace=False, enable_request_outputs=False,
        keep_individual_batch_metrics=True,
        model_num_layers=model_config.get_total_num_layers(),
    )
    engine = BaseLLMEngine(model_config, cache_config, parallel_config,
                           scheduler_config, metrics_config)
    if type(engine.scheduler) is not LPScheduler:
        raise RunError("registered scheduler is "
                       f"{type(engine.scheduler).__name__}, not LPScheduler")
    return engine


def make_sampling_params():
    from sarathi.core.datatypes.sampling_params import SamplingParams
    return SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                          ignore_eos=IGNORE_EOS, stop=None)


def run_requests(engine, prompts, prompt_ids, sampling_params, progress=None):
    """Submit every prompt, step until all finish, and return one result per
    prompt in input order.

    ``progress``, if given, is a dict updated in place with the sequence IDs
    and finished results so a caller can report them after a failure.
    Exceptions from the engine propagate unchanged; the engine is not stepped
    again after one.
    """
    progress = {} if progress is None else progress
    index_of = progress.setdefault("seq_ids", {})
    finished = progress.setdefault("finished", {})
    seq_map = engine.seq_manager.seq_map

    for index, (prompt, ids) in enumerate(zip(prompts, prompt_ids)):
        known = set(seq_map)
        engine.add_request(prompt, sampling_params)
        new = set(seq_map) - known
        if len(new) != 1:
            raise RunError(f"adding prompt {index} created sequences "
                           f"{sorted(new)}")
        seq_id = new.pop()
        index_of[seq_id] = index
        if list(seq_map[seq_id].prompt_token_ids) != ids:
            raise RunError(f"prompt {index} (sequence {seq_id}) was encoded "
                           "differently from its validated tokens")

    # Every step that does any work completes at least one prompt token or
    # one output token, so this bounds a run that makes progress.
    max_steps = sum(len(ids) for ids in prompt_ids) + len(prompts) * MAX_TOKENS
    progress["max_steps"] = max_steps
    steps = 0
    while engine.has_unfinished_requests():
        if steps >= max_steps:
            raise RunError(f"requests still unfinished after {max_steps} "
                           "steps")
        steps += 1
        progress["steps"] = steps
        for output in engine.step():
            if not output.finished:
                continue
            if output.seq_id not in index_of:
                raise RunError(f"finished output for unknown sequence "
                               f"{output.seq_id!r}")
            if output.seq_id in finished:
                raise RunError(f"sequence {output.seq_id!r} finished twice")
            index = index_of[output.seq_id]
            if output.prompt != prompts[index]:
                raise RunError(f"sequence {output.seq_id!r} returned a "
                               f"different prompt from input {index}")
            finished[output.seq_id] = dict(
                index=index,
                seq_id=output.seq_id,
                prompt=output.prompt,
                prompt_token_ids=list(output.prompt_token_ids),
                generated_text=output.text,
                generated_token_ids=list(output.token_ids),
                finish_reason=output.finish_reason,
            )

    missing = sorted(i for s, i in index_of.items() if s not in finished)
    if missing:
        raise RunError(f"engine reports no unfinished requests, but inputs "
                       f"{missing} have no finished output")
    return sorted(finished.values(), key=lambda r: r["index"])


def write_json(path, data):
    """Write ``data`` to a new file; never overwrites."""
    with open(path, "x", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=30).stdout
    except Exception as err:  # provenance only
        return f"unavailable: {err!r}"


def provenance_record():
    """Tested revision, working-tree state, and code hashes."""
    diff = _run("git", "diff", "HEAD")
    return dict(
        git_head=_run("git", "rev-parse", "HEAD").strip(),
        git_branch=_run("git", "rev-parse", "--abbrev-ref", "HEAD").strip(),
        git_status_short=_run("git", "status", "--short",
                              "--untracked-files=all", "--", "scripts",
                              "tests", "sarathi", "*.py"),
        git_diff_head_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        file_sha256={f: sha256_file(REPO_ROOT / f) for f in PROVENANCE_FILES
                     if (REPO_ROOT / f).exists()},
    )


def environment_record():
    from importlib import metadata
    versions = {}
    for dist in ["torch", "transformers", "tokenizers", "ray", "scipy",
                 "numpy", "vllm-flash-attn", "flashinfer"]:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return dict(
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=_run("nvidia-smi", "--query-gpu=index,name,uuid,driver_version,"
                 "memory.total", "--format=csv,noheader").strip(),
        python_executable=sys.executable,
        python_version=platform.python_version(),
        virtual_env=os.environ.get("VIRTUAL_ENV"),
        loaded_modules=os.environ.get("LOADEDMODULES"),
        hf_offline={k: os.environ.get(k) for k in ("HF_HUB_OFFLINE",
                                                   "TRANSFORMERS_OFFLINE")},
        versions=versions,
    )


def failure_record(err):
    record = dict(type=type(err).__name__, message=str(err),
                  traceback=traceback.format_exc())
    # LPSchedulingError carries an immutable stage/category/reason record.
    for name in ("stage", "category", "reason", "snapshot_id"):
        try:
            value = getattr(err, name)
        except AttributeError:
            continue
        record[name] = value if isinstance(value, (str, int)) else repr(value)
    return record


def _ray_initialized():
    ray = sys.modules.get("ray")
    return ray is not None and ray.is_initialized()


def _repo_root_on_pythonpath():
    return str(REPO_ROOT) in [str(Path(p).resolve()) for p in
                              os.environ.get("PYTHONPATH", "").split(os.pathsep)
                              if p]


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True,
                        help="existing local TinyLlama snapshot directory "
                        "(model and tokenizer)")
    parser.add_argument("--prompts-file", required=True,
                        help="JSON array of 1 to 3 nonempty strings")
    parser.add_argument("--output-dir", required=True,
                        help="directory for results.json and summary.json; "
                        "must not already hold them")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    model_path = os.path.abspath(args.model_path)
    if not (Path(model_path) / "config.json").is_file():
        parser.error(f"--model-path {model_path} is not a local model "
                     "directory with config.json")
    out_dir = Path(args.output_dir)
    results_path, summary_path = out_dir / RESULTS_FILE, out_dir / SUMMARY_FILE
    if results_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")
    try:
        prompts = read_prompts(args.prompts_file)
    except InputError as err:
        parser.error(str(err))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(
        success=False,
        command=dict(sys_argv=[sys.executable, *sys.argv],
                     main_argv=None if argv is None else list(argv)),
        model_path=model_path,
        prompts_file=os.path.abspath(args.prompts_file),
        prompts_file_sha256=sha256_file(args.prompts_file),
        num_prompts=len(prompts),
        settings=SETTINGS,
        provenance=provenance_record(),
        environment=environment_record(),
        failure=None,
    )
    progress = {}
    ray_was_initialized = _ray_initialized()
    engine = None
    try:
        if not _repo_root_on_pythonpath():
            raise RunError(f"repository root {REPO_ROOT} must be on "
                           "PYTHONPATH so the Ray worker can import the LP "
                           "modules")
        engine = create_engine(model_path, out_dir / "metrics_store_unused")
        summary["model"] = dict(
            resolved_dtype=str(engine.model_config.dtype),
            max_model_len=engine.model_config.max_model_len,
            hf_config_commit_hash=getattr(engine.model_config.hf_config,
                                          "_commit_hash", None),
            tokenizer_class=type(engine.tokenizer).__name__,
            num_gpu_blocks=engine.scheduler.block_manager.num_total_gpu_blocks,
        )
        prompt_ids = encode_prompts(engine.tokenizer, prompts)
        summary["prompt_token_counts"] = [len(ids) for ids in prompt_ids]
        results = run_requests(engine, prompts, prompt_ids,
                               make_sampling_params(), progress)
        write_json(results_path, results)
        summary["success"] = True
    except BaseException as err:
        summary["failure"] = failure_record(err)
        print(f"run failed: {type(err).__name__}: {err}", file=sys.stderr,
              flush=True)
    finally:
        # Never reused after a failure.
        engine = None
        summary["seq_ids"] = {str(i): s for s, i in
                              progress.get("seq_ids", {}).items()}
        summary["steps"] = progress.get("steps")
        summary["max_steps"] = progress.get("max_steps")
        if not summary["success"]:
            # Diagnostic only: the workload did not complete.
            summary["finished_before_failure"] = sorted(
                (dict(index=r["index"], seq_id=r["seq_id"])
                 for r in progress.get("finished", {}).values()),
                key=lambda r: r["index"])
        # Stop only a Ray runtime this invocation started.
        started_ray = not ray_was_initialized and _ray_initialized()
        if started_ray:
            sys.modules["ray"].shutdown()
        summary["ray"] = dict(started_by_this_run=started_ray,
                              initialized_after_shutdown=_ray_initialized())
        write_json(summary_path, summary)
    print(f"{'SUCCESS' if summary['success'] else 'FAILURE'}: see "
          f"{summary_path}", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
