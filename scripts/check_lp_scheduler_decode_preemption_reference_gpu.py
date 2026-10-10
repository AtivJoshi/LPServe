"""Real-weight comparison for LP preemption of a request that has generated.

Each invocation runs one case in its own process on a real ``BaseLLMEngine``
with real TinyLlama weights from an existing local snapshot, one GPU worker,
one pipeline stage, and a cache deliberately initialized with four 16-token
blocks after normal profiling:

- ``--scheduler vllm --request A``: the registered ``VLLMScheduler`` runs A
  alone (full prefill, three decodes, idle).
- ``--scheduler vllm --request B``: the same for B (full prefill, one decode).
- ``--scheduler lp``: the registered ``LPScheduler`` runs the bounded case.
  A is submitted alone, fully prefilled, and decodes once. B is then
  submitted; the real mapper, solver, and extraction must preempt A and admit
  B. B finishes, A's expanded context (original prompt plus its first
  generated token) is readmitted and recomputed from token zero, and A
  decodes twice more and finishes. Before the LP engine is built, both
  reference summaries are validated; afterwards A's cumulative generated
  history and B's tokens are compared with the references.

Sampling caps differ on purpose. LP A requests 2 tokens; the inherited reset
clears A's output list, so it produces 1 + 2 = 3 tokens cumulatively
(design 16.1). Reference A requests 3 tokens so that its uninterrupted
history covers the same positions. B requests 1 token in both runs.

Utilities are fixed per request for this case (A: decode 40, prefill-token
1, penalty 1; B: decode 40, prefill-token 20, penalty 1). The configured
uniform triple is A's; a validation-only override of the scheduler's
utility-building method supplies each arrived, unfinished raw ID its fixed
triple and is removed after the run. The mapper still validates the exact
key set. D-03 remains OPEN.

A's cumulative history is assembled from completed decode events (sampler
outputs of decode entries). Native final output IDs hold only the
post-reset tokens and are recorded alongside, not used as the history.

Execution success and token agreement are recorded separately. Agreement
covers this bounded case only; both scheduler paths share model and sampler
code, so it cannot exclude a shared defect. Native text is not evidence of
correct token conversion.

Run from the repository root with the root on PYTHONPATH:

    PYTHONPATH="$PWD" timeout 300s python -B \\
        scripts/check_lp_scheduler_decode_preemption_reference_gpu.py \\
        --scheduler vllm --request A --model-path <snapshot> \\
        --output-dir <root>/reference/A
    (likewise --request B into <root>/reference/B)
    PYTHONPATH="$PWD" timeout 300s python -B \\
        scripts/check_lp_scheduler_decode_preemption_reference_gpu.py \\
        --scheduler lp --model-path <snapshot> \\
        --references-dir <root>/reference --output-dir <root>/lp

Exit status is zero only if every check passes (for ``lp``, execution and
every token comparison).
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
import check_lp_scheduler_preemption_reference_gpu as prior
import check_lp_scheduler_reference_gpu as reference_helpers
from check_lp_scheduler_gpu import (CheckFailure, Observer, check,
                                    environment_record, outputs_record,
                                    request_output_record, to_jsonable)
from check_lp_scheduler_preemption_gpu import (WORKER_STATE_METHOD,
                                               CacheCapacitySelection,
                                               central_state, check_quiescent)
from check_lp_scheduler_preemption_reference_gpu import (
    FINGERPRINT_SPEC, WEIGHT_FINGERPRINT_METHOD, build_reference_engine_class,
    compare_fingerprints, file_weight_fingerprint, observe_block_ops,
    validate_prompts)
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
IGNORE_EOS = True
METRICS_MODE = "enabled, all optional outputs off, never plotted"
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
PROMPT_TOKEN_IDS = dict(A=list(range(1000, 1017)), B=list(range(2000, 2048)))
FINISH_REASON = "length"

# Requested generation caps. They differ for A on purpose (see module doc).
REFERENCE_MAX_TOKENS = dict(A=3, B=1)
LP_MAX_TOKENS = dict(A=2, B=1)
CUMULATIVE_TOKENS = dict(A=3, B=1)
CAP_DIFFERENCE = (
    "LP A requests max_tokens=2 and reference A requests max_tokens=3. The "
    "inherited reset clears A's output list after its first token, so LP A "
    "produces 1 + 2 = 3 tokens cumulatively (design 16.1); reference A's "
    "uninterrupted 3 tokens cover the same positions. B uses max_tokens=1 "
    "in both runs.")

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
REQUEST_UTILITIES = dict(
    A=dict(decode_utility=40.0, prefill_token_utility=1.0,
           preemption_penalty=1.0),
    B=dict(decode_utility=40.0, prefill_token_utility=20.0,
           preemption_penalty=1.0),
)
# LPSchedulerConfig requires a uniform triple; the override replaces it.
CONFIGURED_UTILITIES = REQUEST_UTILITIES["A"]
NUMERICAL_POLICY_FIELDS = dict(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)
LP_SEQ_IDS = dict(A=0, B=1)
LP_CALLS = ["map_scheduler_state", "solve_and_extract", "execute_plan",
            "schedule", "_run_workers"]

# Expected emitted (preempted IDs, scheduled [seq_id, prompt_chunk_len]) per
# nonempty step, and after replay each unfinished request's (prompt tokens
# processed, output-token count, central blocks); A is seq 0 and B is seq 1.
LP_EXPECTED_STEPS = [
    dict(name="a_admission_prefill", preempted=[], scheduled=[[0, 16]],
         after={"0": (16, 0, 2)}),
    dict(name="a_prompt_completion", preempted=[], scheduled=[[0, 1]],
         after={"0": (17, 0, 2)}),
    dict(name="a_first_decode", preempted=[], scheduled=[[0, 0]],
         after={"0": (17, 1, 2)}),
    dict(name="a_preemption_b_admission", preempted=[0], scheduled=[[1, 16]],
         after={"0": (0, 0, 0), "1": (16, 0, 3)}),
    dict(name="b_resident_prefill", preempted=[], scheduled=[[1, 16]],
         after={"0": (0, 0, 0), "1": (32, 0, 3)}),
    dict(name="b_prompt_completion", preempted=[], scheduled=[[1, 16]],
         after={"0": (0, 0, 0), "1": (48, 0, 3)}),
    dict(name="b_decode_finish", preempted=[], scheduled=[[1, 0]],
         after={"0": (0, 0, 0)}),
    dict(name="a_recompute_admission", preempted=[], scheduled=[[0, 16]],
         after={"0": (16, 0, 2)}),
    dict(name="a_recompute_prompt_completion", preempted=[],
         scheduled=[[0, 2]], after={"0": (18, 0, 2)}),
    dict(name="a_second_decode", preempted=[], scheduled=[[0, 0]],
         after={"0": (18, 1, 2)}),
    dict(name="a_third_decode_finish", preempted=[], scheduled=[[0, 0]],
         after={}),
]
B_SUBMIT_STEP = 3
BOUNDARY_STEP = 3
READMISSION_STEP = 7
LP_FINISH_STEPS = {6: "B", 10: "A"}
# Whole-step [operation, seq_id, block count] in the central and worker
# managers (scheduling plus completion); every other step has none.
LP_BLOCK_OPS = {
    0: [["allocate", 0, 2]],
    3: [["free", 0, 2], ["allocate", 1, 3]],
    6: [["free", 1, 3]],
    7: [["allocate", 0, 2]],
    10: [["free", 0, 2]],
}
EXPECTED_RELAXED = {"0": (0, 0, 0, 0.5), "1": (16, 0, 1, 0)}
EXPECTED_RELAXED_OBJECTIVE = 319.5
EXPECTED_INTEGER_OBJECTIVE = 319.0

# Constants of the reused helper modules that the reused functions read.
REQUIRED_HELPER_CONSTANTS = {
    preemption: dict(
        INITIALIZED_NUM_GPU_BLOCKS=INITIALIZED_NUM_GPU_BLOCKS,
        TENSOR_PARALLEL_SIZE=TENSOR_PARALLEL_SIZE,
        MAX_NUM_SEQS=LP_MAX_NUM_SEQS),
    prior: dict(
        PROMPT_TOKEN_IDS=PROMPT_TOKEN_IDS,
        FINGERPRINT_SPEC=[["model.embed_tokens.weight", [1000, 1016, 2000, 2047]],
                          ["lm_head.weight", [1000, 1016, 2000, 2047]],
                          ["model.norm.weight", None]]),
    reference_helpers: dict(MODEL_REPO=MODEL_REPO),
}

# Environment facts that must match between the reference and LP runs.
ENVIRONMENT_KEYS = ("hostname", "slurm", "cuda_visible_devices", "gpu",
                    "torch_cuda", "python_executable", "python_version",
                    "virtual_env", "versions")
PROVENANCE_KEYS = ("git_head", "git_diff_head_sha256",
                   "executed_script_sha256", "helper_sha256", "file_sha256")
PROVENANCE_FILES = [
    "scripts/check_lp_scheduler_decode_preemption_reference_gpu.py",
    "scripts/check_lp_scheduler_preemption_reference_gpu.py",
    "scripts/check_lp_scheduler_preemption_gpu.py",
    "scripts/check_lp_scheduler_reference_gpu.py",
    "scripts/check_lp_scheduler_gpu.py",
    "tests/test_check_lp_scheduler_decode_preemption_reference_gpu.py",
    "tests/test_lp_scheduler.py",
    "sarathi/core/scheduler/vllm_scheduler.py",
    "sarathi/core/scheduler/base_scheduler.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/core/block_space_manager/base_block_space_manager.py",
    "sarathi/core/block_space_manager/vllm_block_space_manager.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/core/sequence_manager/worker_sequence_manager.py",
    "sarathi/core/sequence_manager/engine_sequence_manager.py",
    "sarathi/core/datatypes/sequence.py",
    "sarathi/core/datatypes/request_output.py",
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
        git_status_ignored_tests=git("status", "--short", "--ignored",
                                     "--", "tests"),
        git_diff_head_sha256=hashlib.sha256(
            git("diff", "HEAD").encode()).hexdigest(),
        file_sha256={f: (sha256(REPO_ROOT / f) if (REPO_ROOT / f).exists()
                         else None) for f in PROVENANCE_FILES},
        executed_script_sha256=sha256(__file__),
        helper_sha256={m.__name__: sha256(m.__file__)
                       for m in (single, preemption, prior, reference_helpers)},
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def shared_inputs(asset):
    """Inputs that must be identical in every case. The requested cap is
    recorded per case and compared explicitly, not here."""
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
        greedy_sampling=dict(temperature=TEMPERATURE, ignore_eos=IGNORE_EOS,
                             stop=[]),
        reference_max_tokens=REFERENCE_MAX_TOKENS,
        lp_max_tokens=LP_MAX_TOKENS,
        metrics_mode=METRICS_MODE,
        offline_env={k: os.environ.get(k) for k in OFFLINE_ENV},
    )


def sampling_record(mode, label):
    caps = REFERENCE_MAX_TOKENS if mode == "vllm" else LP_MAX_TOKENS
    return dict(temperature=TEMPERATURE, max_tokens=caps[label],
                ignore_eos=IGNORE_EOS, stop=[])


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
    check(reference.get("sampling") == sampling_record("vllm", label),
          f"reference {label} sampling {reference.get('sampling')} differs "
          f"from {sampling_record('vllm', label)}")
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
    check(isinstance(tokens, list)
          and len(tokens) == REFERENCE_MAX_TOKENS[label]
          and all(isinstance(t, int) and not isinstance(t, bool)
                  and isinstance(vocab, int) and 0 <= t < vocab
                  for t in tokens),
          f"reference {label} generated token ids {tokens!r} malformed; "
          f"expected {REFERENCE_MAX_TOKENS[label]} in-vocabulary integers")
    final = reference.get("final_output") or {}
    check(final.get("finished") is True
          and final.get("finish_reason") == FINISH_REASON
          and final.get("token_ids") == tokens
          and final.get("prompt_token_ids") == PROMPT_TOKEN_IDS[label],
          f"reference {label} final output {final}")
    return dict(label=label, generated_token_ids=tokens,
                max_tokens=REFERENCE_MAX_TOKENS[label],
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


def assemble_history(steps, labels_by_seq_id):
    """Cumulative generated tokens per label from completed steps.

    Each step is ``dict(index, scheduled=[[seq_id, chunk], ...],
    sampler_outputs=[[seq_id, token], ...])``. Sampler outputs must pair
    one-to-one, in order, with scheduled entries. A decode entry (chunk 0)
    contributes its token once; a prefill entry's sample is recorded as
    discarded and never enters a history."""
    history = {label: [] for label in labels_by_seq_id.values()}
    decode_events, discarded = [], []
    for step in steps:
        scheduled, sampled = step["scheduled"], step["sampler_outputs"]
        check(isinstance(sampled, list)
              and [s for s, _ in sampled] == [s for s, _ in scheduled],
              f"step {step['index']}: sampler outputs {sampled} not paired "
              f"with scheduled entries {scheduled}")
        for (seq_id, chunk), (_, token) in zip(scheduled, sampled):
            check(seq_id in labels_by_seq_id,
                  f"step {step['index']}: unknown seq {seq_id}")
            check(isinstance(token, int) and not isinstance(token, bool),
                  f"step {step['index']}: token {token!r} is not an integer")
            label = labels_by_seq_id[seq_id]
            if chunk == 0:
                history[label].append(token)
                decode_events.append([step["index"], label, token])
            else:
                discarded.append([step["index"], label, chunk, token])
    return dict(history=history, decode_events=decode_events,
                discarded_prefill_samples=discarded)


def compare_histories(references, history):
    """Compare each label's cumulative history with its reference, plus A's
    first token separately."""
    comparisons = {label: compare_tokens(references[label]["generated_token_ids"],
                                         history.get(label) or [])
                   for label in PROMPT_TOKEN_IDS}
    comparisons["A_first_token"] = compare_tokens(
        references["A"]["generated_token_ids"][:1],
        (history.get("A") or [])[:1])
    return comparisons, all(c["equal"] for c in comparisons.values())


def install_fixed_utilities(scheduler, labels_by_seq_id, record):
    """Validation-only: replace the configured uniform triple by each
    request's fixed triple for exactly the arrived, unfinished raw IDs that
    the real method selects. Returns a function that removes the override."""
    import lpserve_state_mapping as lsm

    real = scheduler._build_utilities

    def build(now):
        keys = real(now)
        unknown = sorted(set(keys) - set(labels_by_seq_id))
        check(not unknown, f"no fixed utility for raw seq IDs {unknown}")
        utilities = {seq_id: lsm.RequestUtility(
            **REQUEST_UTILITIES[labels_by_seq_id[seq_id]]) for seq_id in keys}
        record.append(dict(now=now, utilities=to_jsonable(utilities)))
        return utilities

    scheduler._build_utilities = build

    def remove():
        del scheduler._build_utilities

    return remove


def native_gates(engine):
    """Read-only native admission/append gate values for each owned request."""
    manager = engine.scheduler.block_manager
    gates = {}
    for seq in engine.scheduler.waiting + engine.scheduler.running:
        if manager.is_allocated(seq):
            gates[str(seq.seq_id)] = dict(
                gate="append", can_append_slot=manager.can_append_slot())
        else:
            gates[str(seq.seq_id)] = dict(
                gate="admission",
                initial_blocks=manager.get_num_initial_blocks(seq),
                can_allocate=manager.can_allocate(seq))
    return dict(watermark_blocks=manager.watermark_blocks,
                free_blocks=manager.get_num_free_gpu_blocks(), requests=gates)


def ops_summary(ops):
    return [[op, seq_id, len(blocks)] for op, seq_id, blocks in ops]


def run_reference(engine, label, observer, central_ops, worker_state, submit,
                  trace):
    """Selected request alone: full prefill, its decodes, then drained."""
    blocks = REFERENCE_BLOCKS[label]
    prompt_len = len(PROMPT_TOKEN_IDS[label])
    num_decodes = REFERENCE_MAX_TOKENS[label]
    submit(label, REFERENCE_MAX_TOKENS[label])
    check(sorted(engine.seq_manager.seq_map) == [0], "expected seq 0")
    expected = [[[0, prompt_len]]] + [[[0, 0]]] * num_decodes
    final_output = None
    for index, scheduled in enumerate(expected):
        tag = f"reference {label} step {index}"
        last = index == len(expected) - 1
        before = central_state(engine)
        worker_before = worker_state()
        check_quiescent(f"{tag} (before)", before, worker_before)
        gates = native_gates(engine)
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
            native_gates=gates, events=events, central_block_ops=step_ops,
            after=after, worker_after=worker_after, request_outputs=outputs))

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
            check(gates["requests"]["0"]["can_allocate"] is True,
                  f"{tag}: admission gate {gates}")
            check(ops_summary(step_ops) == [["allocate", 0, blocks]]
                  and ops_summary(worker_ops) == [["allocate", 0, blocks]],
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
            check(gates["requests"]["0"]["can_append_slot"] is True,
                  f"{tag}: append gate {gates}")
            token = events[1]["sampler_outputs"][0][1]
            previous = before["seqs"]["0"]["output_token_ids"]
            check(len(previous) == index - 1, f"{tag}: prior outputs {previous}")
            if not last:
                check(step_ops == [] and worker_ops == [],
                      f"{tag}: unexpected block operations {step_ops} "
                      f"{worker_ops}")
                check(after["seqs"]["0"]["output_token_ids"]
                      == previous + [token]
                      and after["seqs"]["0"]["status"] == "PAUSED",
                      f"{tag}: after decode {after['seqs']['0']}")
            else:
                check(step_ops == [["free", 0, sched["block_tables"]["0"]]]
                      and ops_summary(worker_ops) == [["free", 0, blocks]],
                      f"{tag}: release {step_ops} {worker_ops}")
                check(len(outputs) == 1 and outputs[0]["finished"] is True
                      and outputs[0]["finish_reason"] == FINISH_REASON
                      and outputs[0]["token_ids"] == previous + [token]
                      and outputs[0]["prompt_token_ids"]
                      == PROMPT_TOKEN_IDS[label],
                      f"{tag}: final output {outputs}")
                final_output = outputs[0]
        print(f"{tag}: scheduled {scheduled} sampler "
              f"{events[1]['sampler_outputs']} central free "
              f"{after['free_blocks']} worker free {worker_after['free_blocks']}",
              flush=True)
    return final_output


def check_decision(tag, expected, before, events, policy_json):
    """Pipeline sequence, mapped inputs and per-request utilities, plan and
    output agreement, and limits."""
    check([e["call"] for e in events] == LP_CALLS,
          f"{tag}: call sequence {[e['call'] for e in events]}")
    mapped, solved, executed, scheduled, worker = events
    check(worker["method"] == "execute_model",
          f"{tag}: worker call {worker['method']}")
    check(mapped["result_type"] == "StateSnapshot",
          f"{tag}: mapping returned {mapped['result_type']}: {mapped['result']}")
    snap = mapped["result"]
    problem = snap["lp_problem"]
    check(mapped["num_running_batches"] == 0
          and snap["num_running_batches"] == 0
          and snap["num_pipeline_stages"] == 1,
          f"{tag}: in-flight or stage state at the mapping boundary")
    check(snap["scheduler_iteration_id"] == before["iteration_id"] + 1
          == mapped["iteration_id"], f"{tag}: mapping iteration")
    check((problem["b_max"], problem["c_max"], problem["s_max"], problem["w"])
          == (B_MAX, C_MAX, S_MAX, MEMORY_RESERVE), f"{tag}: mapped limits")
    check(snap["memory_reserve"] == MEMORY_RESERVE
          and snap["resident_limit"] == LP_MAX_NUM_SEQS
          and snap["decode_memory_policy_id"] == DECODE_POLICY_ID
          and snap["numerical_policy"] == policy_json,
          f"{tag}: mapped policy inputs")
    check(snap["free_physical_blocks"] == problem["m_free"]
          == before["free_blocks"], f"{tag}: mapped free blocks")
    check(sorted(r["raw_seq_id"] for r in snap["requests"])
          == sorted(before["waiting"] + before["running"]),
          f"{tag}: mapped request set")
    labels = {v: k for k, v in LP_SEQ_IDS.items()}
    for r in snap["requests"]:
        check(r["utility"] == REQUEST_UTILITIES[labels[r["raw_seq_id"]]],
              f"{tag}: mapped utility for {r['raw_seq_id']}: {r['utility']}")
    check(solved["result_type"] == "SchedulingSuccess",
          f"{tag}: solve returned {solved['result_type']}: {solved['result']}")
    plan = solved["result"]["plan"]
    raw = {r["request_id"]: r["raw_seq_id"] for r in snap["requests"]}
    order = {r["request_id"]: r["order_key"] for r in snap["requests"]}
    decisions = plan["decisions"]
    victims = sorted((d for d in decisions if d["preempt"]),
                     key=lambda d: order[d["request_id"]])
    actions = sorted((d for d in decisions if d["prefill_tokens"] or d["decode"]),
                     key=lambda d: (0 if d["prefill_tokens"] else 1,
                                    order[d["request_id"]]))
    check(executed["result_type"] == "SchedulerOutputs",
          f"{tag}: executor returned {executed['result_type']}: "
          f"{executed['result']}")
    emitted = scheduled["outputs"]
    check(executed["result"] == emitted,
          f"{tag}: scheduler output differs from executor output")
    check(emitted["id"] == snap["scheduler_iteration_id"], f"{tag}: output id")
    check(emitted["preempted_seq_ids"] == [raw[d["request_id"]] for d in victims]
          and emitted["scheduled"] == [[raw[d["request_id"]], d["prefill_tokens"]]
                                       for d in actions]
          and emitted["ignored_seq_ids"] == [],
          f"{tag}: emitted {emitted} differs from plan {decisions}")
    check(emitted["preempted_seq_ids"] == expected["preempted"]
          and emitted["scheduled"] == expected["scheduled"],
          f"{tag}: emitted preempted {emitted['preempted_seq_ids']} scheduled "
          f"{emitted['scheduled']}; expected {expected['preempted']} "
          f"{expected['scheduled']}")
    tokens = sum(c if c > 0 else 1 for _, c in emitted["scheduled"])
    check(tokens <= B_MAX and 1 <= len(emitted["scheduled"]) <= S_MAX,
          f"{tag}: token/action limits ({tokens}, {len(emitted['scheduled'])})")
    check([s for s, _ in worker["sampler_outputs"]]
          == [s for s, _ in emitted["scheduled"]],
          f"{tag}: sampler outputs {worker['sampler_outputs']} not associated "
          f"with scheduled entries {emitted['scheduled']}")
    return snap, solved["result"]


def check_boundary(tag, snap, solved, gates, sched_state, sched_ops,
                   worker_ops, resets, before, after, worker_after, tol):
    """The preemption decision for A after one generated token, and its
    native execution and replay."""
    problem = snap["lp_problem"]
    reqs = {r["request_id"]: r for r in snap["requests"]}
    a, b = reqs["0"], reqs["1"]
    a_before = before["seqs"]["0"]
    check(problem["legal_preemption_ids"] == ["0"],
          f"{tag}: legal victims {problem['legal_preemption_ids']}")
    check((problem["m_free"], problem["w"]) == (2, 0), f"{tag}: m_free/w")
    check(len(a_before["output_token_ids"]) == 1
          and a_before["prompt_token_ids"] == PROMPT_TOKEN_IDS["A"],
          f"{tag}: A before the decision {a_before}")
    check((a["ownership"], a["status"], a["prompt_len"],
           a["prompt_tokens_processed"], a["prompt_processing_finished"],
           a["physical_block_count"], a["preemption_eligible"],
           a["preemption_recovery"], a["decode_eligible"], a["decode_charge"],
           a["prefill_eligible"])
          == ("running", "PAUSED", 17, 17, True, 2, True, 2, True, 1, False),
          f"{tag}: mapped A {a}")
    check((b["ownership"], b["status"], b["prompt_len"],
           b["physical_block_count"], b["preemption_eligible"],
           b["prefill_fixed_charge"])
          == ("waiting", "WAITING", 48, 0, False, 3), f"{tag}: mapped B {b}")
    # Without A's recovery the native admission gate rejects B.
    check(gates["requests"]["1"] == dict(gate="admission", initial_blocks=3,
                                         can_allocate=False)
          and gates["requests"]["0"]["can_append_slot"] is True,
          f"{tag}: native gates before the decision {gates}")
    relaxed = {d["request_id"]: d for d in solved["relaxed"]["decisions"]}
    for rid, want in EXPECTED_RELAXED.items():
        d = relaxed[rid]
        got = (d["x"], d["y"], d["prefill_indicator"], d["z"])
        check(all(abs(g - w) <= tol for g, w in zip(got, want)),
              f"{tag}: relaxed {rid} {got}; expected {want}")
    check(abs(solved["relaxed"]["normalized_objective"]
              - EXPECTED_RELAXED_OBJECTIVE) <= tol,
          f"{tag}: relaxed objective {solved['relaxed']['normalized_objective']}")
    plan = solved["plan"]
    check([(d["request_id"], d["prefill_tokens"], d["decode"], d["preempt"])
           for d in plan["decisions"]] == [("0", 0, 0, 1), ("1", 16, 0, 0)],
          f"{tag}: integer plan {plan['decisions']}")
    check(plan["dominant_preemption_ids"] == ["0"]
          and plan["safety_preemption_ids"] == [],
          f"{tag}: preemption classification")
    check(abs(plan["objective"] - EXPECTED_INTEGER_OBJECTIVE) <= tol,
          f"{tag}: integer objective {plan['objective']}")

    # Central execution: free A, then admit B; A untouched until replay.
    check(ops_summary(sched_ops) == [["free", 0, 2], ["allocate", 1, 3]]
          and sched_ops[0][2] == before["block_tables"]["0"],
          f"{tag}: central block operations {sched_ops}")
    check(sched_state["waiting"] == [0] and sched_state["running"] == [1]
          and set(sched_state["block_tables"]) == {"1"}
          and sched_state["free_blocks"] == 1,
          f"{tag}: central state after execution {sched_state}")
    check(sched_state["seqs"]["0"] == a_before,
          f"{tag}: A changed before replay: {sched_state['seqs']['0']}")

    # Central replay reset A once; worker replay freed A's local blocks
    # before allocating B's.
    check(len(resets) == 1 and resets[0]["seq_id"] == 0
          and resets[0]["before"]["prompt_token_ids"]
          == a_before["prompt_token_ids"]
          and resets[0]["before"]["output_token_ids"]
          == a_before["output_token_ids"],
          f"{tag}: central reset {resets}")
    check(ops_summary(worker_ops) == [["free", 0, 2], ["allocate", 1, 3]],
          f"{tag}: worker block operations {worker_ops}")

    # Expanded prompt: original prompt then the first generated token, once.
    expanded = a_before["prompt_token_ids"] + a_before["output_token_ids"]
    for label, seqs in (("central", after["seqs"]), ("worker", worker_after["seqs"])):
        state = seqs["0"]
        check(state["status"] == "WAITING"
              and state["prompt_tokens_processed"] == 0
              and state["prompt_processing_finished"] is False
              and state["output_token_ids"] == []
              and state["prompt_token_ids"] == expanded
              and state["logical_blocks"] == 2,
              f"{tag}: {label} A after replay {state}")
    check(after["waiting"] == [0] and after["running"] == [1]
          and "0" not in worker_after["block_tables"],
          f"{tag}: ownership/allocation after replay")


def run_lp(engine, observer, central_ops, worker_state, submit, trace,
           policy_json, tol, resets):
    """The bounded decode-preemption workload."""
    final_outputs = {}
    finish_order = []
    for index, expected in enumerate(LP_EXPECTED_STEPS):
        tag = f"lp step {index} {expected['name']}"
        if index == 0:
            submit("A", LP_MAX_TOKENS["A"])
            check(sorted(engine.seq_manager.seq_map) == [LP_SEQ_IDS["A"]],
                  "A must be seq 0")
        if index == B_SUBMIT_STEP:
            # A has generated one token and is a paused, unfinished resident
            # with two blocks; two blocks are free in each manager.
            state, worker = central_state(engine), worker_state()
            check_quiescent(f"{tag} (before B)", state, worker)
            a = state["seqs"]["0"]
            check(a["status"] == "PAUSED" and len(a["output_token_ids"]) == 1
                  and a["prompt_processing_finished"] is True
                  and len(state["block_tables"]["0"]) == 2
                  and state["free_blocks"] == 2 and worker["free_blocks"] == 2,
                  f"{tag}: A before B is submitted {state} {worker}")
            submit("B", LP_MAX_TOKENS["B"])
            check(sorted(engine.seq_manager.seq_map) == [0, LP_SEQ_IDS["B"]],
                  "B must be seq 1")
        before = central_state(engine)
        worker_before = worker_state()
        check_quiescent(f"{tag} (before)", before, worker_before)
        gates = native_gates(engine)
        observer.events.clear()
        del central_ops[:]
        del resets[:]
        step_outputs = engine.step()
        events = list(observer.events)
        step_ops = list(central_ops)
        step_resets = list(resets)
        after = central_state(engine)
        worker_after = worker_state()
        worker_ops = worker_after["block_ops"]
        outputs = [request_output_record(o) for o in step_outputs]
        trace["steps"].append(dict(
            index=index, name=expected["name"], before=before,
            worker_before=worker_before, native_gates=gates, events=events,
            central_block_ops=step_ops, central_resets=step_resets,
            after=after, worker_after=worker_after, request_outputs=outputs))

        snap, solved = check_decision(tag, expected, before, events,
                                      policy_json)
        sched_state = events[3]["state"]
        sched_ops = events[3]["block_ops"]
        check(after["iteration_id"] == before["iteration_id"] + 1,
              f"{tag}: iteration advance")
        check_quiescent(f"{tag} (after)", after, worker_after)
        check(index == BOUNDARY_STEP or not step_resets,
              f"{tag}: unexpected reset {step_resets}")
        check(index == BOUNDARY_STEP
              or all(d["preempt"] == 0 for d in solved["plan"]["decisions"]),
              f"{tag}: unexpected preemption")
        for seq_id, gate in gates["requests"].items():
            if gate["gate"] == "append" and expected["scheduled"][0] == [
                    int(seq_id), 0]:
                check(gate["can_append_slot"], f"{tag}: append gate {gates}")

        # Progress, outputs, and allocation after replay; prefill samples
        # never enter the output list.
        check(sorted(after["seqs"]) == sorted(expected["after"]),
              f"{tag}: unfinished requests {sorted(after['seqs'])}")
        for seq_id, (processed, num_outputs, blocks) in expected["after"].items():
            seq = after["seqs"][seq_id]
            check(seq["prompt_tokens_processed"] == processed
                  and len(seq["output_token_ids"]) == num_outputs
                  and len(after["block_tables"].get(seq_id, [])) == blocks,
                  f"{tag}: seq {seq_id} after replay {seq} blocks "
                  f"{after['block_tables'].get(seq_id)}")
        expected_ops = LP_BLOCK_OPS.get(index, [])
        check(ops_summary(step_ops) == expected_ops
              and ops_summary(worker_ops) == expected_ops,
              f"{tag}: block operations central {step_ops} worker "
              f"{worker_ops}; expected {expected_ops}")
        if index == BOUNDARY_STEP:
            check_boundary(tag, snap, solved, gates, sched_state, sched_ops,
                           worker_ops, step_resets, before, after,
                           worker_after, tol)
        elif index == READMISSION_STEP:
            a_snap = [r for r in snap["requests"] if r["raw_seq_id"] == 0][0]
            check((a_snap["ownership"], a_snap["status"], a_snap["prompt_len"],
                   a_snap["physical_block_count"],
                   a_snap["prompt_tokens_processed"],
                   a_snap["prefill_fixed_charge"])
                  == ("waiting", "WAITING", 18, 0, 0, 2),
                  f"{tag}: mapped A before readmission {a_snap}")
            check(gates["requests"]["0"] == dict(
                gate="admission", initial_blocks=2, can_allocate=True),
                f"{tag}: readmission gate {gates}")
            check(worker_after["seqs"]["0"]["prompt_tokens_processed"] == 16,
                  f"{tag}: worker A recomputed prompt progress")
        if index in LP_FINISH_STEPS:
            label = LP_FINISH_STEPS[index]
            finished_id = LP_SEQ_IDS[label]
            finished = [o for o in outputs if o["finished"]]
            check([o["seq_id"] for o in finished] == [finished_id]
                  and finished[0]["finish_reason"] == FINISH_REASON,
                  f"{tag}: finished outputs {outputs}")
            check(sched_ops == [], f"{tag}: scheduling block operations "
                  f"{sched_ops}")
            check(str(finished_id) not in after["seqs"]
                  and str(finished_id) not in worker_after["seqs"],
                  f"{tag}: finished request still held")
            final_outputs[label] = finished[0]
            finish_order.append(label)
        print(f"{tag}: preempted {events[3]['outputs']['preempted_seq_ids']} "
              f"scheduled {events[3]['outputs']['scheduled']} central free "
              f"{after['free_blocks']} worker free {worker_after['free_blocks']}"
              f" sampler {events[4]['sampler_outputs']}", flush=True)
    check(finish_order == ["B", "A"], f"finish order {finish_order}")
    return final_outputs


def lp_history(trace, final_outputs):
    """Assemble cumulative histories from the trace's completed decode events
    and cross-check them against native state and final outputs."""
    steps = [dict(index=s["index"],
                  scheduled=s["events"][3]["outputs"]["scheduled"],
                  sampler_outputs=s["events"][4]["sampler_outputs"])
             for s in trace["steps"]]
    assembled = assemble_history(
        steps, {seq_id: label for label, seq_id in LP_SEQ_IDS.items()})
    history = assembled["history"]
    for label, count in CUMULATIVE_TOKENS.items():
        check(len(history[label]) == count,
              f"{label} cumulative history {history[label]}; expected "
              f"{count} tokens")
    a_tokens, b_tokens = history["A"], history["B"]
    # A's first token, captured before the reset.
    pre_reset = trace["steps"][BOUNDARY_STEP]["before"]["seqs"]["0"]
    check(pre_reset["output_token_ids"] == a_tokens[:1],
          f"A's pre-reset output {pre_reset['output_token_ids']} differs "
          f"from its first decode event {a_tokens[:1]}")
    final_a, final_b = final_outputs["A"], final_outputs["B"]
    check(final_a["prompt_token_ids"] == PROMPT_TOKEN_IDS["A"] + a_tokens[:1]
          and final_a["token_ids"] == a_tokens[1:],
          f"A's final output {final_a} is inconsistent with history "
          f"{a_tokens}")
    check(final_b["prompt_token_ids"] == PROMPT_TOKEN_IDS["B"]
          and final_b["token_ids"] == b_tokens,
          f"B's final output {final_b} is inconsistent with history "
          f"{b_tokens}")
    assembled["native_output_semantics"] = dict(
        a_pre_reset_output_token_ids=pre_reset["output_token_ids"],
        a_final_prompt_token_ids=final_a["prompt_token_ids"],
        a_final_output_token_ids=final_a["token_ids"],
        a_expanded_prompt_suffix=final_a["prompt_token_ids"][
            len(PROMPT_TOKEN_IDS["A"]):],
        a_requested_max_tokens=LP_MAX_TOKENS["A"],
        a_cumulative_generated=len(a_tokens),
        note=("The final output IDs hold only the post-reset tokens; the "
              "first token is in the expanded prompt. Three cumulative "
              "tokens for a 2-token cap is the inherited generation-limit "
              "behavior (design 16.1)."))
    return assembled


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
                       comparisons=None, cap_difference=CAP_DIFFERENCE)
    engine = None
    remove_utilities = None
    utility_calls = []
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
                numerical_policy=numerical_policy, **CONFIGURED_UTILITIES,
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
                configured_uniform_utilities=CONFIGURED_UTILITIES,
                request_utilities=REQUEST_UTILITIES,
                numerical_policy=policy_json)
        summary["scheduler_identity"] = identity

        # Capacity in the central scheduler, worker manager, and worker cache.
        (worker_init,) = engine._run_workers(
            WORKER_STATE_METHOD, get_all_outputs=True)
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
              and worker_init["cache_layers"] > 0
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
            special_token_ids=sorted(engine.tokenizer.all_special_ids),
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
        resets = []
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
            # Central replay reset, recorded around the native call.
            seq_manager = engine.seq_manager
            native_preempt_seq = seq_manager._preempt_seq

            def seq_view(seq):
                return dict(status=seq.get_status().name,
                            prompt_token_ids=list(seq.prompt_token_ids),
                            output_token_ids=list(seq.output_token_ids),
                            prompt_tokens_processed=seq.prompt_tokens_processed)

            def preempt_seq(seq_id):
                seq = seq_manager.seq_map[seq_id]
                before = seq_view(seq)
                native_preempt_seq(seq_id)
                resets.append(dict(seq_id=seq_id, before=before,
                                   after=seq_view(seq)))

            seq_manager._preempt_seq = preempt_seq
            remove_utilities = install_fixed_utilities(
                scheduler, {v: k for k, v in LP_SEQ_IDS.items()},
                utility_calls)
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
            (state,) = engine._run_workers(WORKER_STATE_METHOD,
                                           get_all_outputs=True)
            return state

        def submit(label, max_tokens):
            engine.add_request(
                prompt=None,
                sampling_params=SamplingParams(
                    temperature=TEMPERATURE, max_tokens=max_tokens,
                    ignore_eos=IGNORE_EOS, stop=None),
                prompt_token_ids=list(PROMPT_TOKEN_IDS[label]))

        if mode == "vllm":
            summary["sampling"] = sampling_record(mode, args.request)
            final = run_reference(engine, args.request, observer, central_ops,
                                  worker_state, submit, trace)
            assembled = assemble_history(
                [dict(index=s["index"],
                      scheduled=s["events"][0]["outputs"]["scheduled"],
                      sampler_outputs=s["events"][1]["sampler_outputs"])
                 for s in trace["steps"]], {0: args.request})
            tokens = assembled["history"][args.request]
            check(tokens == final["token_ids"],
                  f"decode history {tokens} differs from final output "
                  f"{final['token_ids']}")
        else:
            summary["sampling"] = {label: sampling_record(mode, label)
                                   for label in PROMPT_TOKEN_IDS}
            finals = run_lp(engine, observer, central_ops, worker_state,
                            submit, trace, policy_json,
                            numerical_policy.feasibility_tol, resets)

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
        check_quiescent("drained", drained, worker_drained)

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
        nonempty = len(trace["steps"])
        check(before["iteration_id"] == nonempty - 1
              and after["iteration_id"] == nonempty
              and idle_events[0]["outputs"]["id"] == nonempty
              and {**after, "iteration_id": None}
              == {**before, "iteration_id": None},
              f"idle step changed state: {before} -> {after}")
        print(f"idle: calls {[e['call'] for e in idle_events]}, iteration "
              f"{before['iteration_id']} -> {after['iteration_id']}", flush=True)

        if mode == "vllm":
            summary.update(prompt_token_ids=PROMPT_TOKEN_IDS[args.request],
                           final_output=final, history=assembled,
                           generated_token_ids=list(tokens), passed=True)
            print(f"generated token ids: {tokens}", flush=True)
        else:
            assembled = lp_history(trace, finals)
            summary["execution_passed"] = True
            summary["final_outputs"] = finals
            summary["history"] = assembled
            summary["generated_token_ids"] = assembled["history"]
            summary["utility_calls"] = utility_calls
            summary["decisions"] = [dict(
                name=s["name"],
                preempted=s["events"][3]["outputs"]["preempted_seq_ids"],
                scheduled=s["events"][3]["outputs"]["scheduled"],
                sampler_outputs=s["events"][4]["sampler_outputs"])
                for s in trace["steps"]]
            boundary = trace["steps"][BOUNDARY_STEP]
            summary["boundary"] = dict(
                native_gates=boundary["native_gates"],
                mapped_requests=boundary["events"][0]["result"]["requests"],
                legal_preemption_ids=boundary["events"][0]["result"]
                ["lp_problem"]["legal_preemption_ids"],
                relaxed=boundary["events"][1]["result"]["relaxed"],
                plan=boundary["events"][1]["result"]["plan"],
                emitted=boundary["events"][3]["outputs"],
                central_block_ops=boundary["events"][3]["block_ops"],
                central_resets=boundary["central_resets"],
                worker_block_ops=boundary["worker_after"]["block_ops"],
            )
            print("history:", json.dumps(assembled["history"]), flush=True)
            comparisons, equal = compare_histories(references,
                                                   assembled["history"])
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
        try:
            if remove_utilities is not None:
                remove_utilities()
                cleanup["utility_override_removed"] = (
                    "_build_utilities" not in vars(engine.scheduler))
            # Stop only the Ray runtime this process started.
            if "ray" in sys.modules:
                ray = sys.modules["ray"]
                cleanup["ray_initialized_before_shutdown"] = ray.is_initialized()
                ray.shutdown()
                cleanup["ray_initialized_after_shutdown"] = ray.is_initialized()
        finally:
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
