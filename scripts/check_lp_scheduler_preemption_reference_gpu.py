"""Real-weight comparison for the native LP preemption workload.

Each invocation runs one case in its own process on a real ``BaseLLMEngine``
with real TinyLlama weights from an existing local snapshot, one GPU worker,
one pipeline stage, and a cache deliberately initialized with four 16-token
blocks after normal profiling:

- ``--scheduler vllm --request A|B``: the registered ``VLLMScheduler`` runs the
  one selected request alone (full prefill, one decode, idle).
- ``--scheduler lp``: the registered ``LPScheduler`` runs the bounded
  partial-prefill preemption workload (A is prefilled for 16 tokens, B is
  then submitted, A is preempted, B finishes, A is readmitted, recomputed
  from prompt token zero, and finishes). Before the LP engine is built, both
  reference summaries are validated; afterwards each LP request's generated
  token IDs are compared with its own reference.

Real-weight loading is evidenced by a worker-side fingerprint: selected
loaded parameter rows must equal the same rows read from the local
safetensors file and cast to the engine dtype.

Execution success and token agreement are recorded separately. Agreement
covers this one bounded case only; both scheduler paths share model and
sampler code, so it cannot exclude a shared defect. Native text is not
evidence of correct token conversion.

Run from the repository root with the root on PYTHONPATH:

    PYTHONPATH="$PWD" timeout 300s python -B \\
        scripts/check_lp_scheduler_preemption_reference_gpu.py \\
        --scheduler vllm --request A --model-path <snapshot> \\
        --output-dir <root>/reference/A
    (likewise --request B into <root>/reference/B)
    PYTHONPATH="$PWD" timeout 300s python -B \\
        scripts/check_lp_scheduler_preemption_reference_gpu.py \\
        --scheduler lp --model-path <snapshot> \\
        --references-dir <root>/reference --output-dir <root>/lp

Exit status is zero only if every check passes (for ``lp``, execution and
both token comparisons).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Read-only helpers; no imported main() is called and no helper constant is
# changed. Constants the reused functions depend on are checked below.
import check_lp_scheduler_gpu as single
import check_lp_scheduler_preemption_gpu as preemption
import check_lp_scheduler_reference_gpu as reference_helpers
from check_lp_scheduler_gpu import (CheckFailure, Observer, check,
                                    environment_record, outputs_record,
                                    request_output_record, to_jsonable)
from check_lp_scheduler_preemption_gpu import (
    BOUNDARY_STEP, EXPECTED_STEPS as LP_EXPECTED_STEPS, READMISSION_STEP,
    CacheCapacitySelection, central_state, check_boundary, check_decision,
    check_quiescent)
from check_lp_scheduler_reference_gpu import (compare_tokens, inspect_snapshot,
                                              sha256)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Scoped provisional comparison inputs (not project-wide policy).
MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "auto"
DTYPE = "float16"
ATTENTION_BACKEND = "flash_attention"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 64
BLOCK_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.5
SEED = 42
TRUST_REMOTE_CODE = True
TOKENIZER_MODE = "auto"
INITIALIZED_NUM_GPU_BLOCKS = 4
TEMPERATURE = 0.0
MAX_TOKENS = 1
IGNORE_EOS = True
METRICS_MODE = "enabled, all optional outputs off, never plotted"
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
PROMPT_TOKEN_IDS = dict(A=list(range(1000, 1017)), B=list(range(2000, 2048)))
FINISH_REASON = "length"

# Reference scheduler.
VLLM_MAX_NUM_SEQS = 1
VLLM_MAX_NUM_BATCHED_TOKENS = 64
REFERENCE_SCHEDULER_IDENTITY = dict(
    scheduler_class="VLLMScheduler", type="VLLM",
    max_num_seqs=VLLM_MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
    num_pipeline_stages=PIPELINE_PARALLEL_SIZE,
    max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS)
REFERENCE_BLOCKS = dict(A=2, B=3)
VLLM_CALLS = ["schedule", "_run_workers"]

# LP scheduler.
LP_MAX_NUM_SEQS = 1
B_MAX = 64
C_MAX = 16
S_MAX = 1
MEMORY_RESERVE = 0
DECODE_POLICY_ID = "conservative_one_block_v1"
DECODE_UTILITY = 40.0
PREFILL_TOKEN_UTILITY = 1.0
PREEMPTION_PENALTY = 1.0
NUMERICAL_POLICY_FIELDS = dict(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)
LP_SEQ_IDS = dict(A=0, B=1)
LP_FINISH_STEPS = (4, 7)

# Constants of the reused helper modules that the reused functions read.
REQUIRED_HELPER_CONSTANTS = {
    preemption: dict(
        INITIALIZED_NUM_GPU_BLOCKS=INITIALIZED_NUM_GPU_BLOCKS,
        TENSOR_PARALLEL_SIZE=TENSOR_PARALLEL_SIZE,
        MAX_NUM_SEQS=LP_MAX_NUM_SEQS, B_MAX=B_MAX, C_MAX=C_MAX, S_MAX=S_MAX,
        MEMORY_RESERVE=MEMORY_RESERVE, DECODE_POLICY_ID=DECODE_POLICY_ID,
        DECODE_UTILITY=DECODE_UTILITY,
        PREFILL_TOKEN_UTILITY=PREFILL_TOKEN_UTILITY,
        PREEMPTION_PENALTY=PREEMPTION_PENALTY,
        BOUNDARY_STEP=1, READMISSION_STEP=5,
        EXPECTED_STEPS=[
            dict(name="a_admission_prefill", preempted=[], scheduled=[[0, 16]]),
            dict(name="a_preemption_b_admission", preempted=[0],
                 scheduled=[[1, 16]]),
            dict(name="b_resident_prefill", preempted=[], scheduled=[[1, 16]]),
            dict(name="b_prompt_completion", preempted=[], scheduled=[[1, 16]]),
            dict(name="b_decode_finish", preempted=[], scheduled=[[1, 0]]),
            dict(name="a_readmission_prefill", preempted=[],
                 scheduled=[[0, 16]]),
            dict(name="a_prompt_completion", preempted=[], scheduled=[[0, 1]]),
            dict(name="a_decode_finish", preempted=[], scheduled=[[0, 0]]),
        ]),
    reference_helpers: dict(MODEL_REPO=MODEL_REPO),
}

# Loaded parameters compared with the local safetensors file.
FINGERPRINT_SPEC = [
    ["model.embed_tokens.weight", [1000, 1016, 2000, 2047]],
    ["lm_head.weight", [1000, 1016, 2000, 2047]],
    ["model.norm.weight", None],
]
WEIGHT_FINGERPRINT_METHOD = "weight_fingerprint"

# Environment facts that must match between the reference and LP runs.
ENVIRONMENT_KEYS = ("hostname", "slurm", "cuda_visible_devices", "gpu",
                    "torch_cuda", "python_executable", "python_version",
                    "virtual_env", "versions")
PROVENANCE_KEYS = ("git_head", "git_diff_head_sha256",
                   "executed_script_sha256", "helper_sha256", "file_sha256")
PROVENANCE_FILES = [
    "scripts/check_lp_scheduler_preemption_reference_gpu.py",
    "scripts/check_lp_scheduler_preemption_gpu.py",
    "scripts/check_lp_scheduler_reference_gpu.py",
    "scripts/check_lp_scheduler_gpu.py",
    "tests/test_check_lp_scheduler_preemption_reference_gpu.py",
    "tests/test_check_lp_scheduler_preemption_gpu.py",
    "tests/test_lp_scheduler.py",
    "tests/test_lpserve_plan_execution.py",
    "sarathi/core/scheduler/vllm_scheduler.py",
    "sarathi/core/scheduler/base_scheduler.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/core/block_space_manager/base_block_space_manager.py",
    "sarathi/core/block_space_manager/vllm_block_space_manager.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/core/sequence_manager/worker_sequence_manager.py",
    "sarathi/core/sequence_manager/engine_sequence_manager.py",
    "sarathi/engine/base_llm_engine.py",
    "sarathi/worker/base_worker.py",
    "sarathi/worker/cache_engine.py",
    "sarathi/model_executor/model_runner.py",
    "sarathi/model_executor/model_loader.py",
    "sarathi/model_executor/weight_utils.py",
    "sarathi/model_executor/models/llama.py",
    "sarathi/model_executor/layers/sampler.py",
    "sarathi/config.py",
    "lp_relaxation_scheduler.py",
    "lpserve_state_mapping.py",
    "lpserve_plan_execution.py",
]


def check_helper_constants():
    """Fail unless every reused helper constant has this case's value."""
    found = {}
    for module, required in REQUIRED_HELPER_CONSTANTS.items():
        for name, value in required.items():
            actual = getattr(module, name)
            found[f"{module.__name__}.{name}"] = to_jsonable(actual)
            check(actual == value,
                  f"helper constant {module.__name__}.{name}={actual!r} "
                  f"differs from this case's {value!r}")
    return found


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=30).stdout
    except Exception as err:  # provenance only; never affects the check
        return f"unavailable: {err!r}"


def provenance_record():
    return dict(
        git_head=git("rev-parse", "HEAD").strip(),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        git_status_short=git("status", "--short", "--untracked-files=all"),
        git_diff_head_sha256=hashlib.sha256(
            git("diff", "HEAD").encode()).hexdigest(),
        file_sha256={f: (sha256(REPO_ROOT / f) if (REPO_ROOT / f).exists()
                         else None) for f in PROVENANCE_FILES},
        executed_script_sha256=sha256(__file__),
        helper_sha256={m.__name__: sha256(m.__file__)
                       for m in (single, preemption, reference_helpers)},
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def shared_inputs(asset):
    """Inputs that must be identical in every case."""
    return dict(
        asset=asset, load_format=LOAD_FORMAT, dtype=DTYPE,
        attention_backend=ATTENTION_BACKEND,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=SEED,
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        initialized_num_gpu_blocks=INITIALIZED_NUM_GPU_BLOCKS,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        sampling=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                      ignore_eos=IGNORE_EOS, stop=[]),
        metrics_mode=METRICS_MODE,
        offline_env={k: os.environ.get(k) for k in OFFLINE_ENV},
    )


def validate_prompts(tokenizer, vocab_size):
    """Both exact prompts must be ordinary vocabulary tokens; no substitute
    prompt is ever chosen."""
    special = set(tokenizer.all_special_ids)
    tokens = {}
    for label, ids in PROMPT_TOKEN_IDS.items():
        pieces = tokenizer.convert_ids_to_tokens(ids)
        for token_id, piece in zip(ids, pieces):
            check(0 <= token_id < vocab_size and token_id < len(tokenizer)
                  and token_id not in special and piece is not None
                  and piece != tokenizer.unk_token,
                  f"prompt {label} token id {token_id} ({piece!r}) is not "
                  "an ordinary vocabulary token")
        tokens[label] = pieces
    return tokens


def build_reference_worker_class():
    """The preemption driver's validation worker plus a read-only loaded
    weight fingerprint. Built lazily; serialized by value for Ray."""
    base = preemption.build_worker_class()

    class WeightObservingWorker(base):
        def weight_fingerprint(self, spec):
            params = dict(self.model_runner.model.named_parameters())
            result = {}
            for name, rows in spec:
                param = params[name].detach()
                values = param if rows is None else param[rows]
                result[name] = dict(dtype=str(param.dtype),
                                    shape=list(param.shape), rows=rows,
                                    values=values.float().cpu().tolist())
            return result

    return WeightObservingWorker


def build_reference_engine_class(selection):
    """The preemption driver's four-block engine with the weight-observing
    worker; cache sizing and all other worker calls are unchanged."""
    base = preemption.build_engine_class(selection)

    class RealWeightValidationEngine(base):
        def _get_worker_impl(self):
            return build_reference_worker_class()

    return RealWeightValidationEngine


def file_weight_fingerprint(asset, torch_dtype):
    """The same rows read on the CPU from the local safetensors files and
    cast to the engine dtype."""
    import torch
    from safetensors import safe_open

    check(asset["weight_format"] == "safetensors",
          f"weight format {asset['weight_format']} is not safetensors")
    result = {}
    for name, rows in FINGERPRINT_SPEC:
        for record in asset["weight_files"]:
            with safe_open(record["resolved"], framework="pt") as f:
                if name not in f.keys():
                    continue
                tensor = f.get_tensor(name)
                values = tensor if rows is None else tensor[rows]
                result[name] = dict(
                    file=record["name"], file_dtype=str(tensor.dtype),
                    shape=list(tensor.shape), rows=rows,
                    values=values.to(torch_dtype).float().tolist())
                break
        check(name in result, f"{name} not found in the weight files")
    return result


def compare_fingerprints(loaded, from_file):
    record = {}
    for name, rows in FINGERPRINT_SPEC:
        a, b = loaded[name], from_file[name]
        record[name] = dict(
            rows=rows, loaded_dtype=a["dtype"], file_dtype=b["file_dtype"],
            loaded_shape=a["shape"], file_shape=b["shape"],
            num_values=sum(len(v) if isinstance(v, list) else 1
                           for v in a["values"]),
            equal=a["values"] == b["values"] and a["shape"] == b["shape"],
            loaded_values_sha256=hashlib.sha256(
                json.dumps(a["values"]).encode()).hexdigest(),
            first_values=(a["values"][0][:4] if rows else a["values"][:4]),
        )
    return dict(parameters=record,
                all_equal=all(r["equal"] for r in record.values()))


def validate_reference(reference, label, expected):
    """Validate one completed reference summary against this run's
    provenance, environment, and shared inputs; return its comparison
    record. Raises ``CheckFailure`` on any incompatibility."""
    check(label in PROMPT_TOKEN_IDS, f"unknown request label {label!r}")
    check(reference.get("scheduler") == "vllm"
          and reference.get("request") == label,
          f"reference is {reference.get('scheduler')!r} request "
          f"{reference.get('request')!r}, expected vllm request {label}")
    check(reference.get("passed") is True and reference.get("failure") is None,
          f"reference {label} did not pass")
    check(reference.get("scheduler_identity") == REFERENCE_SCHEDULER_IDENTITY,
          f"reference {label} scheduler identity "
          f"{reference.get('scheduler_identity')}")
    check(reference.get("prompt_token_ids") == PROMPT_TOKEN_IDS[label],
          f"reference {label} prompt differs from the exact prompt")
    real = reference.get("real_weights") or {}
    check(real.get("model_config_load_format") == LOAD_FORMAT
          and real.get("fingerprint_matches_file") is True,
          f"reference {label} lacks verified real-weight loading: {real}")
    check(reference.get("shared") == to_jsonable(expected["shared"]),
          f"reference {label} shared inputs or assets differ from this run")
    capacity = reference.get("cache_capacity") or {}
    check(capacity.get("chosen_initialized_blocks") == INITIALIZED_NUM_GPU_BLOCKS
          and capacity.get("central_manager_total_blocks")
          == INITIALIZED_NUM_GPU_BLOCKS
          and (capacity.get("worker") or {}).get("cache_tensor_block_dims")
          == [INITIALIZED_NUM_GPU_BLOCKS],
          f"reference {label} initialized capacity {capacity}")
    provenance = reference.get("provenance") or {}
    for key in PROVENANCE_KEYS:
        check(provenance.get(key) == expected["provenance"][key],
              f"reference {label} provenance {key} differs from this run")
    environment = reference.get("environment") or {}
    for key in ENVIRONMENT_KEYS:
        check(environment.get(key) == expected["environment"][key],
              f"reference {label} environment {key} differs from this run")
    tokens = reference.get("generated_token_ids")
    vocab = (reference.get("shared_after_init") or {}).get("vocab_size")
    check(isinstance(tokens, list) and len(tokens) == MAX_TOKENS
          and all(isinstance(t, int) and not isinstance(t, bool)
                  and isinstance(vocab, int) and 0 <= t < vocab
                  for t in tokens),
          f"reference {label} generated token ids {tokens!r} malformed")
    final = reference.get("final_output") or {}
    check(final.get("finished") is True
          and final.get("finish_reason") == FINISH_REASON
          and final.get("token_ids") == tokens
          and final.get("prompt_token_ids") == PROMPT_TOKEN_IDS[label],
          f"reference {label} final output {final}")
    return dict(label=label, generated_token_ids=tokens,
                shared_after_init=reference["shared_after_init"],
                weight_fingerprint_sha256=reference["weight_fingerprint_sha256"])


def load_references(references_dir, expected):
    references = {}
    for label in PROMPT_TOKEN_IDS:
        path = (Path(references_dir) / label / "summary.json").resolve()
        check(path.is_file(), f"reference summary {path} does not exist")
        record = validate_reference(json.loads(path.read_text()), label,
                                    expected)
        record.update(path=str(path), sha256=sha256(path))
        references[label] = record
    check(references["A"]["shared_after_init"]
          == references["B"]["shared_after_init"]
          and references["A"]["weight_fingerprint_sha256"]
          == references["B"]["weight_fingerprint_sha256"],
          "references A and B differ in model/tokenizer facts or weights")
    return references


def compare_lp_tokens(references, lp_tokens):
    comparisons = {label: compare_tokens(references[label]["generated_token_ids"],
                                         lp_tokens.get(label))
                   for label in PROMPT_TOKEN_IDS}
    return comparisons, all(c["equal"] for c in comparisons.values())


def observe_block_ops(manager):
    """Record central free/allocate calls; each delegates unchanged."""
    ops = []
    native_free, native_allocate = manager.free, manager.allocate

    def free(seq):
        table = manager.block_tables.get(seq.seq_id, [])
        ops.append(["free", seq.seq_id, [b.block_number for b in table]])
        return native_free(seq)

    def allocate(seq):
        result = native_allocate(seq)
        ops.append(["allocate", seq.seq_id, [
            b.block_number for b in manager.block_tables[seq.seq_id]]])
        return result

    manager.free, manager.allocate = free, allocate
    return ops


def run_reference(engine, label, observer, central_ops, worker_state, submit,
                  trace):
    """Selected request alone: full prefill, one decode, then drained."""
    blocks = REFERENCE_BLOCKS[label]
    prompt_len = len(PROMPT_TOKEN_IDS[label])
    submit(label)
    check(sorted(engine.seq_manager.seq_map) == [0], "expected seq 0")
    expected = [[[0, prompt_len]], [[0, 0]]]
    final_output = None
    for index, scheduled in enumerate(expected):
        tag = f"reference {label} step {index}"
        before = central_state(engine)
        worker_before = worker_state()
        check_quiescent(f"{tag} (before)", before, worker_before)
        observer.events.clear()
        del central_ops[:]
        step_outputs = engine.step()
        events = list(observer.events)
        step_ops = list(central_ops)
        after = central_state(engine)
        worker_after = worker_state()
        worker_ops = worker_after["block_ops"]
        outputs = [request_output_record(o) for o in step_outputs]
        trace["steps"].append(dict(
            index=index, before=before, worker_before=worker_before,
            events=events, central_block_ops=step_ops, after=after,
            worker_after=worker_after, request_outputs=outputs))

        check([e["call"] for e in events] == VLLM_CALLS
              and events[1]["method"] == "execute_model",
              f"{tag}: calls {[e['call'] for e in events]}")
        emitted = events[0]["outputs"]
        check(emitted["scheduled"] == scheduled
              and emitted["preempted_seq_ids"] == []
              and emitted["ignored_seq_ids"] == []
              and emitted["id"] == before["iteration_id"] + 1,
              f"{tag}: emitted {emitted}; expected {scheduled}")
        check([s for s, _ in events[1]["sampler_outputs"]] == [0],
              f"{tag}: sampler outputs {events[1]['sampler_outputs']}")
        sched = events[0]["state"]
        check(sched["running"] == [0] and sched["waiting"] == []
              and len(sched["block_tables"]["0"]) == blocks
              and sched["free_blocks"] == INITIALIZED_NUM_GPU_BLOCKS - blocks,
              f"{tag}: state after scheduling {sched}")
        check_quiescent(f"{tag} (after)", after, worker_after)
        if index == 0:
            check([op[:2] for op in step_ops] == [["allocate", 0]]
                  and len(step_ops[0][2]) == blocks
                  and [op[:2] for op in worker_ops] == [["allocate", 0]]
                  and len(worker_ops[0][2]) == blocks,
                  f"{tag}: allocation {step_ops} {worker_ops}")
            seq = after["seqs"]["0"]
            check(seq["status"] == "PAUSED"
                  and seq["prompt_tokens_processed"] == prompt_len
                  and seq["prompt_processing_finished"] is True
                  and seq["output_token_ids"] == []
                  and worker_after["seqs"]["0"]["prompt_tokens_processed"]
                  == prompt_len,
                  f"{tag}: after full prefill {seq}")
        else:
            check(step_ops == [["free", 0, sched["block_tables"]["0"]]]
                  and [op[:2] for op in worker_ops] == [["free", 0]],
                  f"{tag}: release {step_ops} {worker_ops}")
            check(len(outputs) == 1 and outputs[0]["finished"] is True
                  and outputs[0]["finish_reason"] == FINISH_REASON
                  and len(outputs[0]["token_ids"]) == MAX_TOKENS
                  and outputs[0]["prompt_token_ids"] == PROMPT_TOKEN_IDS[label],
                  f"{tag}: final output {outputs}")
            check(outputs[0]["token_ids"] == [events[1]["sampler_outputs"][0][1]],
                  f"{tag}: generated token not the decode's sampler output")
            final_output = outputs[0]
        print(f"{tag}: scheduled {scheduled} sampler "
              f"{events[1]['sampler_outputs']} central free "
              f"{after['free_blocks']} worker free {worker_after['free_blocks']}",
              flush=True)
    return {label: final_output}


def run_lp(engine, observer, central_ops, worker_state, submit, trace,
           policy_json, tol):
    """The validated partial-prefill preemption workload."""
    final_outputs = {}
    finish_order = []
    for index, expected in enumerate(LP_EXPECTED_STEPS):
        tag = f"lp step {index} {expected['name']}"
        if index == 0:
            submit("A")
            check(sorted(engine.seq_manager.seq_map) == [LP_SEQ_IDS["A"]],
                  "A must be seq 0")
        if index == BOUNDARY_STEP:
            submit("B")
            check(sorted(engine.seq_manager.seq_map) == [0, LP_SEQ_IDS["B"]],
                  "B must be seq 1")
        before = central_state(engine)
        worker_before = worker_state()
        check_quiescent(f"{tag} (before)", before, worker_before)
        observer.events.clear()
        del central_ops[:]
        step_outputs = engine.step()
        events = list(observer.events)
        step_ops = list(central_ops)
        after = central_state(engine)
        worker_after = worker_state()
        worker_ops = worker_after["block_ops"]
        outputs = [request_output_record(o) for o in step_outputs]
        trace["steps"].append(dict(
            index=index, name=expected["name"], before=before,
            worker_before=worker_before, events=events,
            central_block_ops=step_ops, after=after,
            worker_after=worker_after, request_outputs=outputs))

        snap, solved = check_decision(tag, expected, before, events,
                                      policy_json)
        sched_state = events[3]["state"]
        sched_ops = events[3]["block_ops"]
        check(after["iteration_id"] == before["iteration_id"] + 1,
              f"{tag}: iteration advance")
        check_quiescent(f"{tag} (after)", after, worker_after)
        if index == 0:
            a = after["seqs"]["0"]
            check(a["status"] == "PAUSED" and a["prompt_tokens_processed"] == 16
                  and not a["prompt_processing_finished"]
                  and a["output_token_ids"] == []
                  and len(after["block_tables"]["0"]) == 2
                  and after["free_blocks"] == 2
                  and worker_after["free_blocks"] == 2,
                  f"{tag}: A after first prefill {after}")
            check([op[:2] for op in step_ops] == [["allocate", 0]]
                  and [op[:2] for op in worker_ops] == [["allocate", 0]],
                  f"{tag}: admission ops {step_ops} {worker_ops}")
        elif index == BOUNDARY_STEP:
            check_boundary(tag, snap, solved, sched_state, sched_ops,
                           worker_ops, before, after, worker_after, tol)
            check(step_ops == sched_ops, f"{tag}: central ops {step_ops}")
        elif index == READMISSION_STEP:
            a_snap = [r for r in snap["requests"] if r["raw_seq_id"] == 0][0]
            check((a_snap["ownership"], a_snap["status"],
                   a_snap["physical_block_count"],
                   a_snap["prompt_tokens_processed"],
                   a_snap["prefill_fixed_charge"])
                  == ("waiting", "WAITING", 0, 0, 2),
                  f"{tag}: mapped A before readmission {a_snap}")
            check(step_ops == sched_ops
                  and [op[:2] for op in step_ops] == [["allocate", 0]]
                  and len(step_ops[0][2]) == 2
                  and [op[:2] for op in worker_ops] == [["allocate", 0]]
                  and len(worker_ops[0][2]) == 2,
                  f"{tag}: readmission ops {step_ops} {worker_ops}")
            check(after["seqs"]["0"]["prompt_tokens_processed"] == 16
                  and worker_after["seqs"]["0"]["prompt_tokens_processed"] == 16,
                  f"{tag}: A recomputed prompt progress")
        elif index not in LP_FINISH_STEPS:
            check(step_ops == [] and worker_ops == [],
                  f"{tag}: unexpected block operations {step_ops} {worker_ops}")
        if index in LP_FINISH_STEPS:
            finished_id = expected["scheduled"][0][0]
            finished = [o for o in outputs if o["finished"]]
            check([o["seq_id"] for o in finished] == [finished_id]
                  and finished[0]["finish_reason"] == FINISH_REASON
                  and len(finished[0]["token_ids"]) == MAX_TOKENS
                  and finished[0]["token_ids"]
                  == [events[4]["sampler_outputs"][0][1]],
                  f"{tag}: finished outputs {outputs}")
            check(sched_ops == []
                  and [op[:2] for op in step_ops] == [["free", finished_id]]
                  and [op[:2] for op in worker_ops] == [["free", finished_id]],
                  f"{tag}: finish block operations {step_ops} {worker_ops}")
            check(str(finished_id) not in after["seqs"]
                  and str(finished_id) not in worker_after["seqs"],
                  f"{tag}: finished request still held")
            final_outputs[finished_id] = finished[0]
            finish_order.append(finished_id)
        print(f"{tag}: preempted {events[3]['outputs']['preempted_seq_ids']} "
              f"scheduled {events[3]['outputs']['scheduled']} central free "
              f"{after['free_blocks']} worker free {worker_after['free_blocks']}"
              f" sampler {events[4]['sampler_outputs']}", flush=True)
    check(finish_order == [LP_SEQ_IDS["B"], LP_SEQ_IDS["A"]],
          f"finish order {finish_order}")
    return {label: final_outputs[seq_id] for label, seq_id in LP_SEQ_IDS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheduler", required=True, choices=["vllm", "lp"])
    parser.add_argument("--request", choices=sorted(PROMPT_TOKEN_IDS))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--references-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    mode = args.scheduler
    if mode == "vllm" and (args.request is None or args.references_dir):
        parser.error("--scheduler vllm requires --request and no "
                     "--references-dir")
    if mode == "lp" and (args.request is not None or not args.references_dir):
        parser.error("--scheduler lp requires --references-dir and no "
                     "--request")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.json"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")

    # Local assets only; set before the model stack and Ray start.
    for key in OFFLINE_ENV:
        os.environ[key] = "1"

    trace = dict(steps=[], idle=None)
    summary = dict(scheduler=mode, request=args.request, passed=False,
                   failure=None, command=sys.argv,
                   provenance=provenance_record())
    if mode == "lp":
        summary.update(execution_passed=False, comparison_passed=False,
                       comparisons=None)
    engine = None
    try:
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import the LP modules")
        summary["helper_constants"] = check_helper_constants()
        summary["environment"] = environment_record()

        # Assets and references are checked before any engine exists.
        asset = inspect_snapshot(str(Path(args.model_path).resolve()))
        summary["asset"] = asset
        shared = shared_inputs(asset)
        summary["shared"] = to_jsonable(shared)
        print("asset:", json.dumps(asset), flush=True)
        references = None
        if mode == "lp":
            references = load_references(args.references_dir, dict(
                shared=shared, provenance=summary["provenance"],
                environment=summary["environment"]))
            summary["references"] = references
            print("references:", json.dumps(references), flush=True)

        import torch

        import lp_relaxation_scheduler as lrs
        import lpserve_plan_execution as lpe
        import lpserve_state_mapping as lsm
        from sarathi.config import (CacheConfig, LPSchedulerConfig,
                                    MetricsConfig, ModelConfig,
                                    ParallelConfig, VLLMSchedulerConfig)
        from sarathi.core.datatypes.sampling_params import SamplingParams
        from sarathi.core.scheduler.lp_scheduler import LPScheduler
        from sarathi.core.scheduler.vllm_scheduler import VLLMScheduler

        numerical_policy = lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS)
        policy_json = to_jsonable(numerical_policy)
        model_config = ModelConfig(
            model=asset["snapshot_path"], tokenizer=asset["snapshot_path"],
            tokenizer_mode=TOKENIZER_MODE,
            trust_remote_code=TRUST_REMOTE_CODE, download_dir=None,
            load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED, revision=None,
            max_model_len=MAX_MODEL_LEN, attention_backend=ATTENTION_BACKEND,
        )
        check(model_config.load_format == LOAD_FORMAT,
              f"load format {model_config.load_format}")
        cache_config = CacheConfig(block_size=BLOCK_SIZE,
                                   gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE)
        if mode == "vllm":
            scheduler_config = VLLMSchedulerConfig(
                max_num_seqs=VLLM_MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                num_pipeline_stages=PIPELINE_PARALLEL_SIZE,
                max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS)
            scheduler_class = VLLMScheduler
        else:
            scheduler_config = LPSchedulerConfig(
                max_num_seqs=LP_MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                num_pipeline_stages=PIPELINE_PARALLEL_SIZE, b_max=B_MAX,
                c_max=C_MAX, s_max=S_MAX, memory_reserve=MEMORY_RESERVE,
                decode_memory_policy_id=DECODE_POLICY_ID,
                decode_utility=DECODE_UTILITY,
                prefill_token_utility=PREFILL_TOKEN_UTILITY,
                preemption_penalty=PREEMPTION_PENALTY,
                numerical_policy=numerical_policy,
            )
            scheduler_class = LPScheduler
        # Enabled metrics with every optional output off; nothing is plotted.
        metrics_config = MetricsConfig(
            replica_id=0, write_metrics=True,
            output_dir=str(out_dir / "metrics_store_unused"),
            wandb_project=None, wandb_group=None, wandb_run_name=None,
            wandb_sweep_id=None, wandb_run_id=None,
            enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
            enable_chrome_trace=False, enable_request_outputs=False,
            keep_individual_batch_metrics=True,
            model_num_layers=model_config.get_total_num_layers(),
        )
        check(scheduler_config.max_num_batched_tokens == 64
              and scheduler_config.max_num_seqs == 1,
              "profiling inputs must be 64 tokens and one sequence")
        print("environment:", json.dumps(summary["environment"], indent=1),
              flush=True)

        selection = CacheCapacitySelection(INITIALIZED_NUM_GPU_BLOCKS)
        engine_class = build_reference_engine_class(selection)
        t0 = time.monotonic()
        engine = engine_class(model_config, cache_config, parallel_config,
                              scheduler_config, metrics_config)
        init_seconds = time.monotonic() - t0
        scheduler = engine.scheduler
        central_manager = scheduler.block_manager
        check(type(scheduler) is scheduler_class,
              f"registered scheduler is {type(scheduler).__name__}")
        config_type = scheduler_config.type.name
        if mode == "vllm":
            identity = dict(
                scheduler_class=type(scheduler).__name__, type=config_type,
                max_num_seqs=scheduler_config.max_num_seqs,
                max_model_len=scheduler_config.max_model_len,
                num_pipeline_stages=scheduler_config.num_pipeline_stages,
                max_num_batched_tokens=scheduler_config.max_num_batched_tokens)
            check(identity == REFERENCE_SCHEDULER_IDENTITY,
                  f"reference scheduler identity {identity}")
        else:
            identity = dict(
                scheduler_class=type(scheduler).__name__, type=config_type,
                max_num_seqs=LP_MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
                num_pipeline_stages=PIPELINE_PARALLEL_SIZE, b_max=B_MAX,
                c_max=C_MAX, s_max=S_MAX, memory_reserve=MEMORY_RESERVE,
                decode_memory_policy_id=DECODE_POLICY_ID,
                utilities=dict(decode=DECODE_UTILITY,
                               prefill_token=PREFILL_TOKEN_UTILITY,
                               preemption_penalty=PREEMPTION_PENALTY),
                numerical_policy=policy_json)
        summary["scheduler_identity"] = identity

        # Capacity in the central scheduler, worker manager, and worker cache.
        (worker_init,) = engine._run_workers(
            preemption.WORKER_STATE_METHOD, get_all_outputs=True)
        n = INITIALIZED_NUM_GPU_BLOCKS
        capacity = dict(
            **selection.record(),
            engine_cache_config_num_gpu_blocks=cache_config.num_gpu_blocks,
            central_manager_total_blocks=central_manager.num_total_gpu_blocks,
            central_allocator_num_blocks=central_manager.gpu_allocator.num_blocks,
            central_free_blocks=central_manager.get_num_free_gpu_blocks(),
            central_watermark_blocks=central_manager.watermark_blocks,
            worker={k: worker_init[k] for k in (
                "cache_config_num_gpu_blocks", "cache_engine_num_gpu_blocks",
                "cache_layers", "cache_tensor_block_dims",
                "cache_tensor_shapes", "manager_total_blocks",
                "allocator_num_blocks", "free_blocks", "block_ops")},
        )
        summary["cache_capacity"] = to_jsonable(capacity)
        print("cache capacity:", json.dumps(summary["cache_capacity"]),
              flush=True)
        check(selection.profile_calls == 1 and len(selection.initialized) == 1,
              "profiling/initialization did not each run exactly once")
        check(cache_config.num_gpu_blocks == n
              and central_manager.num_total_gpu_blocks == n
              and central_manager.gpu_allocator.num_blocks == n
              and central_manager.get_num_free_gpu_blocks() == n
              and not central_manager.block_tables
              and central_manager.watermark_blocks == 0,
              f"central capacity {capacity}")
        check(worker_init["cache_config_num_gpu_blocks"] == n
              and worker_init["cache_engine_num_gpu_blocks"] == n
              and worker_init["cache_tensor_block_dims"] == [n]
              and worker_init["manager_total_blocks"] == n
              and worker_init["allocator_num_blocks"] == n
              and worker_init["free_blocks"] == n
              and worker_init["block_tables"] == {}
              and worker_init["block_ops"] == [],
              f"worker capacity {worker_init}")

        # Real-weight evidence: loaded rows equal the local file's rows.
        (loaded,) = engine._run_workers(WEIGHT_FINGERPRINT_METHOD,
                                        FINGERPRINT_SPEC, get_all_outputs=True)
        from_file = file_weight_fingerprint(asset, model_config.dtype)
        fingerprint = compare_fingerprints(loaded, from_file)
        summary["weight_fingerprint"] = fingerprint
        summary["weight_fingerprint_sha256"] = hashlib.sha256(json.dumps(
            {k: v["loaded_values_sha256"]
             for k, v in fingerprint["parameters"].items()},
            sort_keys=True).encode()).hexdigest()
        summary["real_weights"] = dict(
            requested_load_format=LOAD_FORMAT,
            model_config_load_format=model_config.load_format,
            weight_files=[w["name"] for w in asset["weight_files"]],
            fingerprint_matches_file=fingerprint["all_equal"])
        print("real weights:", json.dumps(summary["real_weights"]), flush=True)
        check(fingerprint["all_equal"],
              f"loaded weights differ from the local file: {fingerprint}")

        vocab_size = model_config.hf_config.vocab_size
        shared_after_init = dict(
            resolved_dtype=str(model_config.dtype),
            hf_config_commit_hash=getattr(model_config.hf_config,
                                          "_commit_hash", None),
            vocab_size=vocab_size, tokenizer_len=len(engine.tokenizer),
            tokenizer_class=type(engine.tokenizer).__name__,
            eos_token_id=engine.tokenizer.eos_token_id,
            prompt_tokens=validate_prompts(engine.tokenizer, vocab_size),
            max_model_len=model_config.max_model_len,
        )
        summary["shared_after_init"] = shared_after_init
        summary["engine_init_seconds"] = round(init_seconds, 3)
        if references is not None:
            for label, record in references.items():
                check(record["shared_after_init"] == shared_after_init,
                      f"reference {label} model/tokenizer facts differ")
                check(record["weight_fingerprint_sha256"]
                      == summary["weight_fingerprint_sha256"],
                      f"reference {label} loaded weights differ")

        observer = Observer()
        if mode == "lp":
            observer.wrap(lsm, "map_scheduler_state", lambda a, k, r: dict(
                num_running_batches=a[0].num_running_batches,
                iteration_id=a[0]._iteration_id,
                result_type=type(r).__name__, result=to_jsonable(r)))
            observer.wrap(lrs, "solve_and_extract", lambda a, k, r: dict(
                result_type=type(r).__name__, result=to_jsonable(r)))
            observer.wrap(lpe, "execute_plan", lambda a, k, r: dict(
                result_type=type(r).__name__,
                result=(outputs_record(r)
                        if type(r).__name__ == "SchedulerOutputs"
                        else to_jsonable(r))))
        observer.wrap(scheduler, "schedule", lambda a, k, r: dict(
            outputs=outputs_record(r), state=central_state(engine),
            block_ops=list(central_ops)))
        observer.wrap(engine, "_run_workers", lambda a, k, r: dict(
            method=a[0],
            sampler_outputs=([[o.seq_id, o.output_token] for o in r]
                             if a[0] == "execute_model" and r is not None
                             else None)))
        central_ops = observe_block_ops(central_manager)

        def worker_state():
            (state,) = engine._run_workers(preemption.WORKER_STATE_METHOD,
                                           get_all_outputs=True)
            return state

        def submit(label):
            engine.add_request(
                prompt=None,
                sampling_params=SamplingParams(
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                    ignore_eos=IGNORE_EOS, stop=None),
                prompt_token_ids=list(PROMPT_TOKEN_IDS[label]))

        if mode == "vllm":
            finals = run_reference(engine, args.request, observer, central_ops,
                                   worker_state, submit, trace)
        else:
            finals = run_lp(engine, observer, central_ops, worker_state,
                            submit, trace, policy_json,
                            numerical_policy.feasibility_tol)

        drained = central_state(engine)
        worker_drained = worker_state()
        check(drained["waiting"] == [] and drained["running"] == []
              and drained["seqs"] == {} and drained["block_tables"] == {}
              and drained["free_blocks"] == n
              and worker_drained["seqs"] == {}
              and worker_drained["block_tables"] == {}
              and worker_drained["free_blocks"] == n
              and not engine.has_unfinished_requests(),
              f"not drained: {drained} {worker_drained}")

        # One ordinary idle call.
        before = central_state(engine)
        observer.events.clear()
        idle_outputs = engine.step()
        after = central_state(engine)
        idle_events = list(observer.events)
        trace["idle"] = dict(before=before, events=idle_events, after=after,
                             request_outputs=[request_output_record(o)
                                              for o in idle_outputs])
        check(idle_outputs == []
              and [e["call"] for e in idle_events] == ["schedule"]
              and idle_events[0]["outputs"]["scheduled"] == []
              and idle_events[0]["outputs"]["preempted_seq_ids"] == [],
              f"idle step: {idle_events}")
        check(after["iteration_id"] == before["iteration_id"] + 1
              and {**after, "iteration_id": None}
              == {**before, "iteration_id": None},
              f"idle step changed state: {before} -> {after}")
        print(f"idle: calls {[e['call'] for e in idle_events]}, iteration "
              f"{before['iteration_id']} -> {after['iteration_id']}", flush=True)

        if mode == "vllm":
            (final,) = finals.values()
            summary.update(prompt_token_ids=PROMPT_TOKEN_IDS[args.request],
                           final_output=final,
                           generated_token_ids=list(final["token_ids"]),
                           passed=True)
            print(f"generated token ids: {final['token_ids']}", flush=True)
        else:
            summary["execution_passed"] = True
            summary["final_outputs"] = finals
            summary["generated_token_ids"] = {
                label: list(out["token_ids"]) for label, out in finals.items()}
            comparisons, equal = compare_lp_tokens(
                references, summary["generated_token_ids"])
            summary.update(comparisons=comparisons, comparison_passed=equal,
                           passed=equal)
            print("comparisons:", json.dumps(comparisons), flush=True)
            check(equal, "LP generated token ids differ from the references")
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        cleanup = dict(engine_constructed=engine is not None)
        # Stop only the Ray runtime this process started.
        if "ray" in sys.modules:
            ray = sys.modules["ray"]
            cleanup["ray_initialized_before_shutdown"] = ray.is_initialized()
            ray.shutdown()
            cleanup["ray_initialized_after_shutdown"] = ray.is_initialized()
        summary["cleanup"] = cleanup
        print("cleanup:", json.dumps(cleanup), flush=True)
        trace_path.write_text(json.dumps(trace, indent=1))
        summary_path.write_text(json.dumps(summary, indent=1))
    if mode == "lp":
        print("EXECUTION:", "PASS" if summary["execution_passed"] else "FAIL",
              flush=True)
        print("COMPARISON:", "PASS" if summary["comparison_passed"] else
              ("FAIL" if summary["comparisons"] else "NOT PERFORMED"),
              flush=True)
    print("RESULT:", "PASS" if summary["passed"] else "FAIL", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
