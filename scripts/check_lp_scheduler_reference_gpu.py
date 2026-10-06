"""One-request real-weight comparison of the LP scheduler with a reference.

Runs one request (prompt IDs 1000-1015, four greedy output tokens) through a
real ``BaseLLMEngine`` with real TinyLlama weights from an existing local
Hugging Face snapshot, one GPU worker, and native completion. Each invocation
runs one scheduler in its own process:

- ``--scheduler vllm``: LPServe's registered ``VLLMScheduler``, which processes
  the full 16-token prompt in one step and then decodes four times.
- ``--scheduler lp``: the registered ``LPScheduler``, which processes the
  prompt in two 8-token chunks and then decodes four times. It reads the
  reference run's ``summary.json``, checks that the reference passed with the
  same inputs and assets, and compares the complete generated token IDs.

Each run asserts its exact schedule, state transitions, memory deltas, native
completion, and one idle call. Agreement covers this one request only; both
paths share model and sampler code, so it cannot rule out a shared defect.

Run from the repository root with the root on PYTHONPATH and offline Hugging
Face settings:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \\
    timeout 300s python -B scripts/check_lp_scheduler_reference_gpu.py \\
        --scheduler vllm --model-path <snapshot> --output-dir <dir>/reference

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \\
    timeout 300s python -B scripts/check_lp_scheduler_reference_gpu.py \\
        --scheduler lp --model-path <snapshot> \\
        --reference-summary <dir>/reference/summary.json --output-dir <dir>/lp

Exit status is zero only if every check passes (for ``lp``, the execution
checks and the exact token comparison).
"""

import argparse
import glob
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Read-only helpers from the single-request check; its main() is not used.
import check_lp_scheduler_gpu as single
from check_lp_scheduler_gpu import (Observer, check, environment_record,
                                    outputs_record, request_output_record,
                                    to_jsonable)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Scoped provisional comparison inputs (not project-wide policy).
MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "auto"
DTYPE = "float16"
ATTENTION_BACKEND = "flash_attention"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 32
BLOCK_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.5
SEED = 42
TRUST_REMOTE_CODE = True
TOKENIZER_MODE = "auto"
PROMPT_TOKEN_IDS = list(range(1000, 1016))
PROMPT_LEN = len(PROMPT_TOKEN_IDS)
TEMPERATURE = 0.0
MAX_TOKENS = 4
IGNORE_EOS = True
METRICS_MODE = "enabled, all optional outputs off, never plotted"
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# Reference scheduler.
VLLM_MAX_NUM_SEQS = 1
VLLM_MAX_NUM_BATCHED_TOKENS = 32

# LP scheduler.
LP_MAX_NUM_SEQS = 1
B_MAX = 32
C_MAX = 8
S_MAX = 1
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

# 16 prompt + 4 generated tokens span at most two 16-token blocks.
PEAK_BLOCKS = 2
FINISHED = "FINISHED_LENGTH_CAPPED"


def _step(name, chunk, sched_physical, prompt_done, generated, final_logical,
          final_physical, finished=False):
    return dict(name=name, chunk=chunk, sched_physical=sched_physical,
                prompt_done=prompt_done, generated=generated,
                final_logical=final_logical, final_physical=final_physical,
                finished=finished)


# Expected nonempty decisions. ``chunk`` is the emitted prompt chunk (0 for a
# decode). Physical-block counts are for the single request; free blocks are
# N minus them, where N is the free count before submission.
DECODE_STEPS = [
    _step("decode_without_allocation", 0, 1, 16, 1, 2, 1),
    _step("decode_with_block_append", 0, 2, 16, 2, 2, 2),
    _step("decode", 0, 2, 16, 3, 2, 2),
    _step("final_decode", 0, 2, 16, 4, 2, 0, finished=True),
]
EXPECTED_STEPS = dict(
    vllm=[_step("admission_full_prefill", 16, 1, 16, 0, 1, 1)] + DECODE_STEPS,
    lp=[_step("admission_prefill", 8, 1, 8, 0, 1, 1),
        _step("resident_prefill", 8, 1, 16, 0, 1, 1)] + DECODE_STEPS,
)
LP_CALLS = ["map_scheduler_state", "solve_and_extract", "execute_plan",
            "schedule", "_run_workers"]
VLLM_CALLS = ["schedule", "_run_workers"]

PROVENANCE_FILES = [
    "scripts/check_lp_scheduler_reference_gpu.py",
    "scripts/check_lp_scheduler_gpu.py",
    "sarathi/core/scheduler/vllm_scheduler.py",
    "sarathi/core/scheduler/base_scheduler.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/core/block_space_manager/base_block_space_manager.py",
    "sarathi/core/block_space_manager/vllm_block_space_manager.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/engine/base_llm_engine.py",
    "sarathi/config.py",
    "sarathi/model_executor/model_loader.py",
    "sarathi/model_executor/weight_utils.py",
    "sarathi/model_executor/models/llama.py",
    "lp_relaxation_scheduler.py",
    "lpserve_state_mapping.py",
    "lpserve_plan_execution.py",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=30).stdout
    except Exception as err:  # provenance only; never affects the check
        return f"unavailable: {err!r}"


def provenance_record():
    """Tested revision, working-tree state, and file hashes."""
    diff = git("diff", "HEAD")
    return dict(
        git_head=git("rev-parse", "HEAD").strip(),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        git_status_short=git("status", "--short", "--untracked-files=all"),
        git_diff_head_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        file_sha256={f: sha256(REPO_ROOT / f) for f in PROVENANCE_FILES},
        executed_script_sha256=sha256(__file__),
        imported_helper_sha256=sha256(single.__file__),
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def file_record(path, with_hash):
    p = Path(path)
    record = dict(name=p.name, is_symlink=p.is_symlink(), exists=p.exists())
    if record["is_symlink"]:
        record["link_target"] = os.readlink(p)
    if record["exists"]:
        record["size"] = p.stat().st_size
        record["resolved"] = str(p.resolve())
        if with_hash:
            record["sha256"] = sha256(p)
    return record


def safetensors_record(path):
    """Header-level completeness check: the file size must equal the header
    plus the furthest tensor end. Tensor data is not read."""
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        check(8 + header_len <= size, f"{path}: truncated safetensors header")
        header = json.loads(f.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    data_end = max((v["data_offsets"][1] for v in tensors.values()), default=0)
    check(size == 8 + header_len + data_end,
          f"{path}: size {size} != header {8 + header_len} + data {data_end}")
    return dict(num_tensors=len(tensors),
                dtypes=sorted({v["dtype"] for v in tensors.values()}),
                tensor_names_sha256=hashlib.sha256(
                    "\n".join(sorted(tensors)).encode()).hexdigest())


def inspect_snapshot(model_path):
    """Check that ``model_path`` is an existing local snapshot of MODEL_REPO
    with config, tokenizer, and complete weights; never downloads.

    Weight selection mirrors ``prepare_hf_model_weights`` for a local
    directory with ``load_format="auto"``: every ``*.safetensors`` file, or,
    if none, every ``*.bin``/``*.pt`` file except ``training_args.bin``."""
    snapshot = Path(model_path)
    check(snapshot.is_absolute(), f"model path {model_path} must be absolute")
    check(snapshot.is_dir(), f"model path {model_path} is not a directory")
    repo_dir = snapshot.parent.parent
    check(snapshot.parent.name == "snapshots"
          and repo_dir.name == "models--" + MODEL_REPO.replace("/", "--"),
          f"{model_path} is not a Hugging Face cache snapshot of {MODEL_REPO}")
    refs_main = repo_dir / "refs" / "main"

    broken = sorted(e.name for e in snapshot.iterdir()
                    if e.is_symlink() and not e.exists())
    check(not broken, f"broken links in snapshot: {broken}")

    check((snapshot / "config.json").exists(), "config.json is missing")
    check((snapshot / "tokenizer_config.json").exists()
          and ((snapshot / "tokenizer.json").exists()
               or (snapshot / "tokenizer.model").exists()),
          "tokenizer assets are missing")
    small = ["config.json", "generation_config.json", "tokenizer_config.json",
             "tokenizer.json", "tokenizer.model", "special_tokens_map.json"]
    small_files = {n: file_record(snapshot / n, True) for n in small
                   if (snapshot / n).exists()}
    config = json.loads((snapshot / "config.json").read_text())

    weights = sorted(glob.glob(str(snapshot / "*.safetensors")))
    weight_format = "safetensors"
    if not weights:
        weight_format = "pt"
        weights = sorted(w for pattern in ("*.bin", "*.pt")
                         for w in glob.glob(str(snapshot / pattern))
                         if not w.endswith("training_args.bin"))
    check(weights, f"no weight files (*.safetensors, *.bin, *.pt) in "
          f"{model_path}; only {sorted(e.name for e in snapshot.iterdir())}")
    weight_files = []
    for w in weights:
        record = file_record(w, False)
        check(record["exists"] and record["size"] > 0,
              f"weight file {w} is missing or empty")
        if weight_format == "safetensors":
            record["safetensors"] = safetensors_record(w)
        weight_files.append(record)

    indexes = {}
    for index_name in ("model.safetensors.index.json",
                       "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.exists():
            continue
        shards = sorted(set(json.loads(index_path.read_text())
                            ["weight_map"].values()))
        missing = [s for s in shards if not (snapshot / s).exists()]
        check(not missing, f"{index_name} names missing shards {missing}")
        if index_name.startswith("model.safetensors") == (
                weight_format == "safetensors"):
            check(set(shards) <= {Path(w).name for w in weights},
                  f"{index_name} shards {shards} not all selected")
        indexes[index_name] = dict(sha256=sha256(index_path), shards=shards)

    return dict(
        repository=MODEL_REPO,
        snapshot_path=str(snapshot),
        snapshot_revision=snapshot.name,
        refs_main=(refs_main.read_text().strip() if refs_main.exists()
                   else None),
        architectures=config.get("architectures"),
        config_torch_dtype=config.get("torch_dtype"),
        config_vocab_size=config.get("vocab_size"),
        small_files=small_files,
        weight_format=weight_format,
        weight_files=weight_files,
        weight_indexes=indexes,
        npcache_dir_present=(snapshot / "np").exists(),
    )


def capture_state(engine, seq):
    """Selected primitive state; holds no live objects. Central block IDs are
    copied for observation only."""
    scheduler = engine.scheduler
    bm = scheduler.block_manager
    request = None
    if seq is not None:
        table = bm.block_tables.get(seq.seq_id)
        request = dict(
            seq_id=seq.seq_id,
            status=seq.get_status().name,
            prompt_tokens_processed=seq.get_num_prompt_tokens_processed(),
            prompt_processing_finished=seq.prompt_processing_finished,
            generated=seq.get_output_len(),
            output_token_ids=list(seq.get_output_token_ids()),
            logical_blocks=len(seq.logical_token_blocks),
            physical_blocks=0 if table is None else len(table),
            block_ids=None if table is None else [b.block_number
                                                  for b in table],
            is_finished=seq.is_finished(),
        )
    return dict(
        iteration_id=scheduler._iteration_id,
        num_running_batches=scheduler.num_running_batches,
        num_pipeline_stages=scheduler.scheduler_config.num_pipeline_stages,
        waiting=[s.seq_id for s in scheduler.waiting],
        running=[s.seq_id for s in scheduler.running],
        free_blocks=bm.get_num_free_gpu_blocks(),
        block_tables={str(k): [b.block_number for b in v]
                      for k, v in bm.block_tables.items()},
        engine_seq_ids=sorted(engine.seq_manager.seq_map),
        num_unfinished=engine.get_num_unfinished_requests(),
        request=request,
    )


def worker_record(args, kwargs, result):
    record = dict(method=args[0])
    if args[0] == "execute_model":
        record["sampler_outputs"] = [[o.seq_id, o.output_token]
                                     for o in result]
    return record


def shared_inputs(asset):
    """Inputs that must be identical in the reference and LP runs."""
    return dict(
        asset=asset, load_format=LOAD_FORMAT, dtype=DTYPE,
        attention_backend=ATTENTION_BACKEND,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=SEED,
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        sampling=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                      ignore_eos=IGNORE_EOS, stop=[]),
        metrics_mode=METRICS_MODE,
        offline_env={k: os.environ.get(k) for k in OFFLINE_ENV},
    )


def validate_prompt(tokenizer, vocab_size):
    special = set(tokenizer.all_special_ids)
    tokens = tokenizer.convert_ids_to_tokens(PROMPT_TOKEN_IDS)
    for token_id, token in zip(PROMPT_TOKEN_IDS, tokens):
        check(0 <= token_id < vocab_size and token_id < len(tokenizer)
              and token_id not in special and token is not None
              and token != tokenizer.unk_token,
              f"prompt token id {token_id} ({token!r}) is not a valid "
              "ordinary vocabulary token")
    return tokens


def load_reference(path, provenance, shared):
    """Validate the reference summary before any LP engine is created."""
    path = Path(path).resolve()
    check(path.is_file(), f"reference summary {path} does not exist")
    record = dict(path=str(path), sha256=sha256(path))
    reference = json.loads(path.read_text())
    check(reference.get("scheduler") == "vllm"
          and reference.get("passed") is True
          and reference.get("failure") is None,
          "reference summary is not a passing reference run")
    ref_prov = reference["provenance"]
    for key in ("git_head", "git_diff_head_sha256", "executed_script_sha256",
                "imported_helper_sha256", "file_sha256"):
        check(ref_prov[key] == provenance[key],
              f"reference provenance {key} differs from this run")
    check(reference["shared"] == to_jsonable(shared),
          "reference shared inputs or assets differ from this run")
    tokens = reference["generated_token_ids"]
    check(isinstance(tokens, list) and len(tokens) == MAX_TOKENS
          and all(isinstance(t, int) for t in tokens),
          f"reference token ids {tokens} malformed")
    record.update(seq_id=reference["seq_id"],
                  generated_token_ids=tokens,
                  shared_after_init=reference["shared_after_init"])
    return record


def compare_tokens(reference_ids, lp_ids):
    first = next((i for i, (r, l) in enumerate(zip(reference_ids, lp_ids))
                  if r != l), None)
    if first is None and len(reference_ids) != len(lp_ids):
        first = min(len(reference_ids), len(lp_ids))
    return dict(
        reference_token_ids=reference_ids, lp_token_ids=lp_ids,
        equal=reference_ids == lp_ids,
        indexing="zero-based position within the generated tokens",
        first_difference_index=first,
        reference_token_at_difference=(
            None if first is None or first >= len(reference_ids)
            else reference_ids[first]),
        lp_token_at_difference=(
            None if first is None or first >= len(lp_ids) else lp_ids[first]),
    )


def check_lp_events(tag, exp, before, events, seq_id, policy_json):
    """Mapper, solver/extraction, and executor checks for one LP step."""
    mapped, solved, executed, scheduled = events[:4]
    check(mapped["result_type"] == "StateSnapshot",
          f"{tag}: mapping returned {mapped['result_type']}: "
          f"{mapped['result']}")
    snap = mapped["result"]
    problem = snap["lp_problem"]
    check((problem["b_max"], problem["c_max"], problem["s_max"]) ==
          (B_MAX, C_MAX, S_MAX), f"{tag}: mapped limits")
    check(snap["memory_reserve"] == MEMORY_RESERVE
          and snap["resident_limit"] == LP_MAX_NUM_SEQS
          and snap["decode_memory_policy_id"] == DECODE_POLICY_ID
          and snap["numerical_policy"] == policy_json
          and snap["num_pipeline_stages"] == 1
          and snap["num_running_batches"] == 0,
          f"{tag}: mapped policy inputs")
    check(snap["free_physical_blocks"] == before["free_blocks"]
          and problem["m_free"] == before["free_blocks"],
          f"{tag}: mapped free blocks")
    check(len(snap["requests"]) == 1
          and snap["requests"][0]["raw_seq_id"] == seq_id
          and [r["request_id"] for r in problem["requests"]]
          == [snap["requests"][0]["request_id"]],
          f"{tag}: mapped request set {snap['requests']}")
    check(snap["requests"][0]["utility"] == dict(
        decode_utility=DECODE_UTILITY,
        prefill_token_utility=PREFILL_TOKEN_UTILITY,
        preemption_penalty=PREEMPTION_PENALTY), f"{tag}: mapped utilities")

    check(solved["result_type"] == "SchedulingSuccess",
          f"{tag}: solve returned {solved['result_type']}: "
          f"{solved['result']}")
    plan = solved["result"]["plan"]
    problem_id = problem["problem_id"]
    check(solved["problem_id_in"] == problem_id
          and solved["result"]["problem_id"] == problem_id
          and plan["problem_id"] == problem_id
          and executed["problem_id_in"] == problem_id
          and executed["snapshot_id_in"] == snap["snapshot_id"],
          f"{tag}: problem/plan/snapshot identities differ")
    check(len(plan["decisions"]) == 1
          and plan["decisions"][0]["request_id"]
          == snap["requests"][0]["request_id"], f"{tag}: plan decisions")
    decision = plan["decisions"][0]
    want = (exp["chunk"], 0 if exp["chunk"] else 1, 0)
    check((decision["prefill_tokens"], decision["decode"],
           decision["preempt"]) == want,
          f"{tag}: plan {decision} differs from expected {want}")
    check(not plan["dominant_preemption_ids"]
          and not plan["safety_preemption_ids"],
          f"{tag}: plan preemption ids")

    check(executed["result_type"] == "SchedulerOutputs",
          f"{tag}: executor returned {executed['result_type']}")
    emitted = scheduled["outputs"]
    check(executed["result"] == emitted,
          f"{tag}: scheduler output differs from executor output")
    check(emitted["id"] == snap["scheduler_iteration_id"],
          f"{tag}: decision id {emitted['id']} differs from snapshot")
    chunk = exp["chunk"]
    check(len(emitted["scheduled"]) <= S_MAX and chunk <= C_MAX
          and (chunk or 1) <= B_MAX
          and len(before["running"]) + len(before["waiting"])
          <= LP_MAX_NUM_SEQS,
          f"{tag}: action/chunk/token/resident limits")
    return dict(snapshot_id=snap["snapshot_id"], problem_id=problem_id,
                plan_decision=decision)


def check_step(tag, mode, exp, n_free, watermark_blocks, before, events,
               after, step_outputs, seq_id, policy_json):
    """Native schedule, state, and memory checks for one nonempty step."""
    calls = [e["call"] for e in events]
    want_calls = LP_CALLS if mode == "lp" else VLLM_CALLS
    check(calls == want_calls, f"{tag}: unexpected call sequence {calls}")
    scheduled, worker = events[-2], events[-1]
    check(worker["method"] == "execute_model",
          f"{tag}: worker call {worker['method']}")

    breq = before["request"]
    check(before["num_pipeline_stages"] == 1
          and before["num_running_batches"] == 0,
          f"{tag}: not a quiescent single-stage boundary")
    admission = breq["status"] == "WAITING"
    if admission:
        check(before["waiting"] == [seq_id] and before["running"] == []
              and breq["block_ids"] is None and before["block_tables"] == {},
              f"{tag}: waiting ownership {before}")
        # Native admission gate (watermark).
        check(before["free_blocks"] - 1 >= watermark_blocks,
              f"{tag}: native admission gate would refuse")
    else:
        check(before["waiting"] == [] and before["running"] == [seq_id]
              and breq["status"] == "PAUSED" and breq["block_ids"],
              f"{tag}: resident ownership {before}")
    check(before["engine_seq_ids"] == [seq_id]
          and before["num_unfinished"] == 1, f"{tag}: engine ownership")

    emitted = scheduled["outputs"]
    check(emitted["id"] == before["iteration_id"] + 1,
          f"{tag}: decision id {emitted['id']}")
    check(emitted["scheduled"] == [[seq_id, exp["chunk"]]],
          f"{tag}: emitted {emitted['scheduled']}, expected "
          f"{[[seq_id, exp['chunk']]]}")
    check(not emitted["ignored_seq_ids"]
          and not emitted["preempted_seq_ids"],
          f"{tag}: ignored/preempted ids emitted")

    # After scheduling: allocation changes, progress does not.
    sched = scheduled["state"]
    sreq = sched["request"]
    check(sched["num_running_batches"] == 1,
          f"{tag}: running batches after scheduling")
    check(sched["waiting"] == [] and sched["running"] == [seq_id]
          and list(sched["block_tables"]) == [str(seq_id)],
          f"{tag}: ownership after scheduling")
    check(sreq["physical_blocks"] == exp["sched_physical"]
          and sched["free_blocks"] == n_free - exp["sched_physical"],
          f"{tag}: blocks after scheduling {sreq['physical_blocks']} "
          f"physical, {sched['free_blocks']} free")
    added = sreq["physical_blocks"] - breq["physical_blocks"]
    if not admission:
        # Existing central block IDs are preserved; the logical/physical gap
        # before the step is exactly the number of appended blocks.
        check(sreq["block_ids"][:len(breq["block_ids"])] == breq["block_ids"],
              f"{tag}: block IDs not preserved")
        gap = breq["logical_blocks"] - breq["physical_blocks"]
        check(added == (gap if exp["chunk"] == 0 else 0) and added in (0, 1),
              f"{tag}: appended {added} blocks with gap {gap}")
        if added:
            check(before["free_blocks"] > 0,
                  f"{tag}: native append gate would refuse")
    else:
        check(added == 1, f"{tag}: admission allocated {added} blocks")
    for key in ("prompt_tokens_processed", "prompt_processing_finished",
                "generated", "output_token_ids", "logical_blocks"):
        check(sreq[key] == breq[key], f"{tag}: {key} changed by scheduling")

    # After completion.
    areq = after["request"]
    check(after["iteration_id"] == before["iteration_id"] + 1,
          f"{tag}: iteration advanced "
          f"{after['iteration_id'] - before['iteration_id']}")
    check(after["num_running_batches"] == 0,
          f"{tag}: running batches after completion")
    check(areq["prompt_tokens_processed"] == exp["prompt_done"]
          and areq["prompt_processing_finished"]
          == (exp["prompt_done"] == PROMPT_LEN)
          and areq["generated"] == exp["generated"]
          and areq["status"] == (FINISHED if exp["finished"] else "PAUSED")
          and areq["logical_blocks"] == exp["final_logical"],
          f"{tag}: request state after completion {areq}")
    if exp["chunk"]:
        check(areq["prompt_tokens_processed"]
              == breq["prompt_tokens_processed"] + exp["chunk"]
              and areq["output_token_ids"] == breq["output_token_ids"],
              f"{tag}: prefill progress or appended token")
    else:
        check(len(areq["output_token_ids"]) == len(breq["output_token_ids"]) + 1
              and areq["output_token_ids"][:-1] == breq["output_token_ids"],
              f"{tag}: decode did not append exactly one token")
    check(after["free_blocks"] == n_free - exp["final_physical"],
          f"{tag}: free blocks after completion {after['free_blocks']}")
    check(len(step_outputs) == 1 and step_outputs[0]["seq_id"] == seq_id
          and step_outputs[0]["finished"] == exp["finished"]
          and step_outputs[0]["token_ids"] == areq["output_token_ids"],
          f"{tag}: request outputs {step_outputs}")
    if exp["finished"]:
        check(after["waiting"] == [] and after["running"] == []
              and after["block_tables"] == {} and after["engine_seq_ids"] == []
              and after["num_unfinished"] == 0 and areq["block_ids"] is None,
              f"{tag}: drained state {after}")
        check(step_outputs[0]["finish_reason"] == "length",
              f"{tag}: finish reason {step_outputs[0]['finish_reason']}")
    else:
        check(after["waiting"] == [] and after["running"] == [seq_id]
              and areq["physical_blocks"] == exp["final_physical"]
              and areq["block_ids"] == sreq["block_ids"]
              and after["engine_seq_ids"] == [seq_id],
              f"{tag}: resident state after completion {after}")
    return dict(emitted=emitted["scheduled"], blocks_added=added,
                free_after_schedule=sched["free_blocks"],
                free_after_completion=after["free_blocks"],
                sampler_outputs=worker["sampler_outputs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheduler", required=True, choices=["vllm", "lp"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-summary")
    args = parser.parse_args()
    mode = args.scheduler
    if (mode == "lp") != (args.reference_summary is not None):
        parser.error("--reference-summary is required for --scheduler lp "
                     "and not accepted for --scheduler vllm")

    # The directory may already exist (holding the console log), but earlier
    # run artifacts are never overwritten.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.json"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")

    trace = dict(submission=None, steps=[], idle=None)
    summary = dict(scheduler=mode, passed=False, failure=None,
                   command=sys.argv, provenance=provenance_record())
    if mode == "lp":
        summary.update(execution_passed=False, comparison_passed=False,
                       comparison=None)
    engine = None
    try:
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import the LP modules")
        check(all(os.environ.get(k) == "1" for k in OFFLINE_ENV),
              f"{OFFLINE_ENV} must both be set to 1")

        # Assets and the reference are checked before any engine exists.
        asset = inspect_snapshot(args.model_path)
        summary["asset"] = asset
        shared = shared_inputs(asset)
        summary["shared"] = to_jsonable(shared)
        print("asset:", json.dumps(asset), flush=True)
        reference = None
        if mode == "lp":
            reference = load_reference(args.reference_summary,
                                       summary["provenance"], shared)
            summary["reference"] = reference
            print("reference:", json.dumps(reference), flush=True)

        import lp_relaxation_scheduler as lrs
        import lpserve_plan_execution as lpe
        import lpserve_state_mapping as lsm
        from sarathi.config import (CacheConfig, LPSchedulerConfig,
                                    MetricsConfig, ModelConfig,
                                    ParallelConfig, VLLMSchedulerConfig)
        from sarathi.core.datatypes.sampling_params import SamplingParams
        from sarathi.core.scheduler.lp_scheduler import LPScheduler
        from sarathi.core.scheduler.vllm_scheduler import VLLMScheduler
        from sarathi.engine.base_llm_engine import BaseLLMEngine

        numerical_policy = lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS)
        policy_json = to_jsonable(numerical_policy)
        # The snapshot directory is both model and tokenizer, so both runs
        # read the same local files through the native loader.
        model_config = ModelConfig(
            model=asset["snapshot_path"], tokenizer=asset["snapshot_path"],
            tokenizer_mode=TOKENIZER_MODE,
            trust_remote_code=TRUST_REMOTE_CODE, download_dir=None,
            load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED, revision=None,
            max_model_len=MAX_MODEL_LEN, attention_backend=ATTENTION_BACKEND,
        )
        check(model_config.load_format == "auto",
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
            scheduler_record = dict(
                type="VLLM", max_num_seqs=VLLM_MAX_NUM_SEQS,
                max_model_len=MAX_MODEL_LEN,
                num_pipeline_stages=PIPELINE_PARALLEL_SIZE,
                max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS)
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
            scheduler_record = dict(
                type="LP", max_num_seqs=LP_MAX_NUM_SEQS,
                max_model_len=MAX_MODEL_LEN,
                num_pipeline_stages=PIPELINE_PARALLEL_SIZE, b_max=B_MAX,
                c_max=C_MAX, s_max=S_MAX, memory_reserve=MEMORY_RESERVE,
                decode_memory_policy_id=DECODE_POLICY_ID,
                utilities=dict(decode=DECODE_UTILITY,
                               prefill_token=PREFILL_TOKEN_UTILITY,
                               preemption_penalty=PREEMPTION_PENALTY),
                numerical_policy=policy_json)
        # Enabled metrics mode with every optional output off; the disabled
        # mode cannot complete engine.step() (see the single-request
        # handoff). plot() is never called.
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
        scheduler_record.update(
            profiling_max_num_batched_tokens=(
                scheduler_config.max_num_batched_tokens),
            profiling_max_num_seqs=scheduler_config.max_num_seqs)
        summary["scheduler_config"] = scheduler_record

        summary["environment"] = environment_record()
        print("environment:", json.dumps(summary["environment"], indent=1),
              flush=True)

        t0 = time.monotonic()
        engine = BaseLLMEngine(model_config, cache_config, parallel_config,
                               scheduler_config, metrics_config)
        init_seconds = time.monotonic() - t0
        scheduler = engine.scheduler
        bm = scheduler.block_manager
        check(type(scheduler) is scheduler_class,
              f"registered scheduler is {type(scheduler).__name__}")

        vocab_size = model_config.hf_config.vocab_size
        prompt_tokens = validate_prompt(engine.tokenizer, vocab_size)
        shared_after_init = dict(
            resolved_dtype=str(model_config.dtype),
            vocab_size=vocab_size, tokenizer_len=len(engine.tokenizer),
            tokenizer_class=type(engine.tokenizer).__name__,
            eos_token_id=engine.tokenizer.eos_token_id,
            prompt_tokens=prompt_tokens,
            max_model_len=model_config.max_model_len,
        )
        summary["shared_after_init"] = shared_after_init
        if reference is not None:
            check(reference["shared_after_init"] == shared_after_init,
                  "reference model/tokenizer facts differ from this run")
        n_free = bm.get_num_free_gpu_blocks()
        summary["pool"] = dict(
            hf_config_commit_hash=getattr(model_config.hf_config,
                                          "_commit_hash", None),
            profiled_num_gpu_blocks=cache_config.num_gpu_blocks,
            block_manager_total_blocks=bm.num_total_gpu_blocks,
            watermark=bm.watermark, watermark_blocks=bm.watermark_blocks,
            free_blocks_before_request=n_free,
            engine_init_seconds=round(init_seconds, 3),
        )
        print("shared_after_init:", json.dumps(shared_after_init), flush=True)
        print("pool:", json.dumps(summary["pool"]), flush=True)

        # Pool adequacy: the two-block peak plus the native watermark (and,
        # for LP, the planning reserve) must fit.
        reserve = MEMORY_RESERVE if mode == "lp" else 0
        check(n_free == bm.num_total_gpu_blocks and not bm.block_tables,
              "block pool must be fully free before the request")
        check(n_free - PEAK_BLOCKS >= bm.watermark_blocks + reserve,
              f"profiled pool {n_free} too small for watermark "
              f"{bm.watermark_blocks} + reserve {reserve} + {PEAK_BLOCKS}")

        # Observation starts after construction, so profiling is excluded.
        observer = Observer()
        if mode == "lp":
            observer.wrap(lsm, "map_scheduler_state", lambda a, k, r: dict(
                result_type=type(r).__name__, result=to_jsonable(r)))
            observer.wrap(lrs, "solve_and_extract", lambda a, k, r: dict(
                problem_id_in=a[0].problem_id,
                result_type=type(r).__name__, result=to_jsonable(r)))
            observer.wrap(lpe, "execute_plan", lambda a, k, r: dict(
                snapshot_id_in=a[1].snapshot_id,
                problem_id_in=a[2].problem_id,
                result_type=type(r).__name__,
                result=(outputs_record(r)
                        if type(r).__name__ == "SchedulerOutputs"
                        else to_jsonable(r))))
        observer.wrap(scheduler, "schedule", lambda a, k, r: dict(
            outputs=outputs_record(r), state=capture_state(engine, seq_ref[0])))
        observer.wrap(engine, "_run_workers", worker_record)

        seq_ref = [None]
        before = capture_state(engine, None)
        observer.events.clear()
        sampling = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                        ignore_eos=IGNORE_EOS, stop=None)
        engine.add_request(prompt=None,
                           sampling_params=SamplingParams(**sampling),
                           prompt_token_ids=list(PROMPT_TOKEN_IDS))
        check(len(engine.seq_manager.seq_map) == 1, "expected one sequence")
        seq_id = next(iter(engine.seq_manager.seq_map))
        seq_ref[0] = engine.seq_manager.seq_map[seq_id]
        after = capture_state(engine, seq_ref[0])
        trace["submission"] = dict(
            seq_id=seq_id, prompt_token_ids=PROMPT_TOKEN_IDS,
            sampling=sampling, before=before,
            events=list(observer.events), after=after)
        check(observer.calls() == ["_run_workers"]
              and observer.events[0]["method"] == "add_seq",
              f"submission calls {observer.events}")
        check(after["waiting"] == [seq_id] and after["running"] == []
              and after["free_blocks"] == n_free
              and after["block_tables"] == {}
              and after["request"]["status"] == "WAITING"
              and after["request"]["output_token_ids"] == []
              and list(seq_ref[0].prompt_token_ids) == PROMPT_TOKEN_IDS,
              f"submitted state {after}")
        print(f"submitted seq {seq_id} prompt {PROMPT_TOKEN_IDS}", flush=True)

        finished_output = None
        decisions = []
        for index, exp in enumerate(EXPECTED_STEPS[mode]):
            tag = f"step {index} {exp['name']}"
            before = capture_state(engine, seq_ref[0])
            observer.events.clear()
            step_outputs = engine.step()
            after = capture_state(engine, seq_ref[0])
            events = list(observer.events)
            outputs = [request_output_record(o) for o in step_outputs]
            record = dict(index=index, name=exp["name"], before=before,
                          events=events, after=after, request_outputs=outputs)
            trace["steps"].append(record)
            if mode == "lp":
                record["lp"] = check_lp_events(tag, exp, before, events,
                                               seq_id, policy_json)
            record["checked"] = check_step(
                tag, mode, exp, n_free, bm.watermark_blocks, before, events,
                after, outputs, seq_id, policy_json)
            decisions.append(dict(name=exp["name"], **record["checked"]))
            if exp["finished"]:
                finished_output = outputs[0]
            print(f"{tag}: emitted {record['checked']['emitted']}, "
                  f"+{record['checked']['blocks_added']} block(s), free "
                  f"{record['checked']['free_after_schedule']} -> "
                  f"{after['free_blocks']}, {after['request']['status']} "
                  f"prompt={after['request']['prompt_tokens_processed']} "
                  f"generated={after['request']['output_token_ids']}",
                  flush=True)

        check(finished_output is not None
              and len(finished_output["token_ids"]) == MAX_TOKENS,
              "expected exactly one finished output with four tokens")
        check(not engine.has_unfinished_requests(), "unfinished requests remain")
        check(bm.get_num_free_gpu_blocks() == n_free,
              "free blocks not restored to N")

        # One ordinary idle call after draining.
        before = capture_state(engine, None)
        observer.events.clear()
        idle_outputs = engine.step()
        after = capture_state(engine, None)
        trace["idle"] = dict(before=before, events=list(observer.events),
                             after=after, request_outputs=[
                                 request_output_record(o) for o in idle_outputs])
        check(idle_outputs == [], f"idle step returned {idle_outputs}")
        check(observer.calls() == ["schedule"],
              f"idle step calls {observer.calls()}")
        check(observer.events[0]["outputs"]["scheduled"] == [],
              "idle step scheduled work")
        check(after["iteration_id"] == before["iteration_id"] + 1,
              "idle step must advance the iteration once")
        check({**after, "iteration_id": None} == {**before, "iteration_id": None},
              f"idle step changed state: {before} -> {after}")
        print(f"idle: no outputs, calls {observer.calls()}, iteration "
              f"{before['iteration_id']} -> {after['iteration_id']}",
              flush=True)

        generated = list(finished_output["token_ids"])
        summary.update(n_free=n_free, seq_id=seq_id,
                       final_output=finished_output,
                       generated_token_ids=generated, decisions=decisions)
        print(f"generated token ids: {generated}", flush=True)
        if mode == "vllm":
            summary["passed"] = True
        else:
            summary["execution_passed"] = True
            comparison = compare_tokens(reference["generated_token_ids"],
                                        generated)
            summary["comparison"] = comparison
            summary["comparison_passed"] = comparison["equal"]
            summary["passed"] = comparison["equal"]
            print("comparison:", json.dumps(comparison), flush=True)
            check(comparison["equal"], "LP generated token ids differ from "
                  f"the reference at index "
                  f"{comparison['first_difference_index']}")
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        # Stop only the Ray runtime this process started (no-op otherwise).
        if "ray" in sys.modules:
            ray = sys.modules["ray"]
            summary["ray_initialized_before_shutdown"] = ray.is_initialized()
            ray.shutdown()
            summary["ray_initialized_after_shutdown"] = ray.is_initialized()
        trace_path.write_text(json.dumps(trace, indent=1))
        summary_path.write_text(json.dumps(summary, indent=1))
    if mode == "lp":
        print("EXECUTION:", "PASS" if summary["execution_passed"] else "FAIL",
              flush=True)
        print("COMPARISON:", "PASS" if summary["comparison_passed"] else
              ("FAIL" if summary["comparison"] else "NOT PERFORMED"),
              flush=True)
    print("RESULT:", "PASS" if summary["passed"] else "FAIL", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
