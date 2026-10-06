"""Three-request real-weight comparison of the LP scheduler with references.

Requests A, B, and C (prompt IDs 1000-1015, 2000-2015, and 3000-3015; four
greedy output tokens each) run through a real ``BaseLLMEngine`` with real
TinyLlama weights from an existing local Hugging Face snapshot, one GPU
worker, and native completion. Each invocation runs one case in its own
process:

- ``--scheduler vllm --request X``: LPServe's registered ``VLLMScheduler``
  serves request X alone (full 16-token prefill, then four decodes).
- ``--scheduler lp``: the registered ``LPScheduler`` serves A, B, and C
  together, all added before the first step, with at most two scheduled
  actions per step (``s_max=2``) and room for three residents. It first
  checks the three reference summaries in ``--references-dir/{A,B,C}``, then
  compares each request's complete generated token IDs with its reference.

The LP schedule is not prescribed. Every nonempty LP step is checked against
invariants derived from the observed pre-step state, and the run must witness
an omitted eligible waiting request, a boundary with three prompt-complete
residents, an omitted resident later selected, and completion of all three.

Agreement covers this workload only; both paths share model and sampler code,
so it cannot rule out a shared defect. It does not establish general fairness,
arbitrary-workload correctness, sampler row association, GPU-worker block
tables, or performance.

Run from the repository root with the root on PYTHONPATH and offline Hugging
Face settings:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \\
    timeout 300s python -B scripts/check_lp_scheduler_contention_reference_gpu.py \\
        --scheduler vllm --request A --model-path <snapshot> \\
        --output-dir <dir>/reference/A

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \\
    timeout 300s python -B scripts/check_lp_scheduler_contention_reference_gpu.py \\
        --scheduler lp --model-path <snapshot> \\
        --references-dir <dir>/reference --output-dir <dir>/lp

Exit status is zero only if every check passes (for ``lp``, the execution
checks, the contention witnesses, and all three exact token comparisons).
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Read-only helpers; none of their main() functions is used and none of their
# constants is changed.
import check_lp_scheduler_contention_gpu as contention
import check_lp_scheduler_gpu as single
import check_lp_scheduler_reference_gpu as reference_helper
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
PROMPTS = dict(A=list(range(1000, 1016)), B=list(range(2000, 2016)),
               C=list(range(3000, 3016)))
PROMPT_LEN = 16
TEMPERATURE = 0.0
MAX_TOKENS = 4
IGNORE_EOS = True
METRICS_MODE = "enabled, all optional outputs off, never plotted"
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
FINISHED = "FINISHED_LENGTH_CAPPED"

# Reference scheduler.
VLLM_MAX_NUM_SEQS = 1
VLLM_MAX_NUM_BATCHED_TOKENS = 32
# 16 prompt + 4 generated tokens span at most two 16-token blocks.
REFERENCE_PEAK_BLOCKS = 2

# LP scheduler.
LP_MAX_NUM_SEQS = 3
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
# Two blocks per request at its peak, three requests.
LP_PEAK_BLOCKS_IN_USE = 6
# Total prompt-token work plus requested decode-token work: a termination
# guard, not an expected schedule length.
MAX_NONEMPTY_STEPS = len(PROMPTS) * (PROMPT_LEN + MAX_TOKENS)

# Helper constants this script relies on through the imported functions.
HELPER_CONSTANTS = [
    (reference_helper, "MODEL_REPO", MODEL_REPO),
    (reference_helper, "PROMPT_LEN", PROMPT_LEN),
    (reference_helper, "MAX_TOKENS", MAX_TOKENS),
    (reference_helper, "FINISHED", FINISHED),
    (contention, "PROMPT_LEN", PROMPT_LEN),
    (contention, "MAX_TOKENS", MAX_TOKENS),
    (contention, "FINISHED", FINISHED),
    (contention, "B_MAX", B_MAX),
    (contention, "C_MAX", C_MAX),
    (contention, "S_MAX", S_MAX),
    (contention, "MAX_NUM_SEQS", LP_MAX_NUM_SEQS),
    (contention, "MEMORY_RESERVE", MEMORY_RESERVE),
    (contention, "DECODE_POLICY_ID", DECODE_POLICY_ID),
    (contention, "DECODE_UTILITY", DECODE_UTILITY),
    (contention, "PREFILL_TOKEN_UTILITY", PREFILL_TOKEN_UTILITY),
    (contention, "PREEMPTION_PENALTY", PREEMPTION_PENALTY),
    (contention, "PEAK_BLOCKS_IN_USE", LP_PEAK_BLOCKS_IN_USE),
]

HELPER_FILES = {
    "scripts/check_lp_scheduler_gpu.py": single,
    "scripts/check_lp_scheduler_reference_gpu.py": reference_helper,
    "scripts/check_lp_scheduler_contention_gpu.py": contention,
}
PROVENANCE_FILES = [
    "scripts/check_lp_scheduler_contention_reference_gpu.py",
    *HELPER_FILES,
    "sarathi/core/scheduler/vllm_scheduler.py",
    "sarathi/core/scheduler/base_scheduler.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/core/block_space_manager/base_block_space_manager.py",
    "sarathi/core/block_space_manager/vllm_block_space_manager.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/engine/base_llm_engine.py",
    "sarathi/config.py",
    "sarathi/model_executor/model_runner.py",
    "sarathi/model_executor/model_loader.py",
    "sarathi/model_executor/weight_utils.py",
    "sarathi/model_executor/models/llama.py",
    "sarathi/model_executor/layers/sampler.py",
    "lp_relaxation_scheduler.py",
    "lpserve_state_mapping.py",
    "lpserve_plan_execution.py",
]
sha256 = reference_helper.sha256
git = reference_helper.git


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
        imported_helper_sha256={f: sha256(m.__file__)
                                for f, m in HELPER_FILES.items()},
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def shared_inputs(asset):
    """Inputs that must be identical in all four runs."""
    return dict(
        asset=asset, load_format=LOAD_FORMAT, dtype=DTYPE,
        attention_backend=ATTENTION_BACKEND,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=SEED,
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        prompt_token_ids=PROMPTS,
        sampling=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                      ignore_eos=IGNORE_EOS, stop=[]),
        metrics_mode=METRICS_MODE,
        offline_env={k: os.environ.get(k) for k in OFFLINE_ENV},
    )


def validate_prompts(tokenizer, vocab_size):
    """All prompt IDs are valid ordinary tokens; the lists are disjoint."""
    special = set(tokenizer.all_special_ids)
    tokens = {}
    for label, ids in PROMPTS.items():
        check(len(ids) == PROMPT_LEN, f"prompt {label} length {len(ids)}")
        tokens[label] = tokenizer.convert_ids_to_tokens(ids)
        for token_id, token in zip(ids, tokens[label]):
            check(0 <= token_id < vocab_size and token_id < len(tokenizer)
                  and token_id not in special and token is not None
                  and token != tokenizer.unk_token,
                  f"prompt {label} token id {token_id} ({token!r}) is not a "
                  "valid ordinary vocabulary token")
    all_ids = [i for ids in PROMPTS.values() for i in ids]
    check(len(all_ids) == len(set(all_ids)),
          "prompt token lists must be distinct and disjoint")
    return tokens


def load_references(references_dir, provenance, shared):
    """Validate the A/B/C reference summaries before any LP engine exists."""
    records = {}
    for label in PROMPTS:
        path = (Path(references_dir) / label / "summary.json").resolve()
        check(path.is_file(), f"reference summary {path} does not exist")
        data = json.loads(path.read_text())
        check(data.get("scheduler") == "vllm" and data.get("request") == label
              and data.get("passed") is True and data.get("failure") is None,
              f"reference {label} is not a passing reference run")
        check(data.get("request_prompt_token_ids") == PROMPTS[label],
              f"reference {label} prompt ids differ")
        ref_prov = data["provenance"]
        for key in ("git_head", "git_diff_head_sha256",
                    "executed_script_sha256", "imported_helper_sha256",
                    "file_sha256"):
            check(ref_prov[key] == provenance[key],
                  f"reference {label} provenance {key} differs from this run")
        check(data["shared"] == to_jsonable(shared),
              f"reference {label} shared inputs or assets differ")
        tokens = data["generated_token_ids"]
        check(isinstance(tokens, list) and len(tokens) == MAX_TOKENS
              and all(isinstance(t, int) for t in tokens),
              f"reference {label} token ids {tokens} malformed")
        records[label] = dict(
            path=str(path), sha256=sha256(path), seq_id=data["seq_id"],
            generated_token_ids=tokens,
            shared_after_init=data["shared_after_init"],
            environment=dict(hostname=data["environment"]["hostname"],
                             slurm=data["environment"]["slurm"]))
    return records


def build_engine(mode, asset, out_dir, summary):
    import lp_relaxation_scheduler as lrs
    from sarathi.config import (CacheConfig, LPSchedulerConfig,
                                MetricsConfig, ModelConfig, ParallelConfig,
                                VLLMSchedulerConfig)
    from sarathi.core.scheduler.lp_scheduler import LPScheduler
    from sarathi.core.scheduler.vllm_scheduler import VLLMScheduler
    from sarathi.engine.base_llm_engine import BaseLLMEngine

    numerical_policy = lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS)
    # The snapshot directory is both model and tokenizer in every run.
    model_config = ModelConfig(
        model=asset["snapshot_path"], tokenizer=asset["snapshot_path"],
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        download_dir=None, load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED,
        revision=None, max_model_len=MAX_MODEL_LEN,
        attention_backend=ATTENTION_BACKEND,
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
            numerical_policy=to_jsonable(numerical_policy))
        # The generic profiler builds max_num_seqs prompts sharing
        # max_num_batched_tokens: 96 // 3 = three 32-token prompts
        # (source-confirmed; the worker-side batch is not observed here).
        check(scheduler_config.max_num_batched_tokens == B_MAX
              and B_MAX // LP_MAX_NUM_SEQS == MAX_MODEL_LEN
              and B_MAX % LP_MAX_NUM_SEQS == 0,
              "profiling inputs must give three max-length prompts")
    # Enabled metrics mode with every optional output off; the disabled mode
    # cannot complete engine.step() (see the single-request handoff).
    # plot() is never called.
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
        profiling_max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
        profiling_max_num_seqs=scheduler_config.max_num_seqs)
    summary["scheduler_config"] = scheduler_record

    summary["environment"] = environment_record()
    print("environment:", json.dumps(summary["environment"], indent=1),
          flush=True)
    t0 = time.monotonic()
    engine = BaseLLMEngine(model_config, cache_config, parallel_config,
                           scheduler_config, metrics_config)
    init_seconds = time.monotonic() - t0
    check(type(engine.scheduler) is scheduler_class,
          f"registered scheduler is {type(engine.scheduler).__name__}")

    vocab_size = model_config.hf_config.vocab_size
    shared_after_init = dict(
        resolved_dtype=str(model_config.dtype),
        vocab_size=vocab_size, tokenizer_len=len(engine.tokenizer),
        tokenizer_class=type(engine.tokenizer).__name__,
        eos_token_id=engine.tokenizer.eos_token_id,
        prompt_tokens=validate_prompts(engine.tokenizer, vocab_size),
        max_model_len=model_config.max_model_len,
        hf_config_commit_hash=getattr(model_config.hf_config,
                                      "_commit_hash", None),
    )
    summary["shared_after_init"] = shared_after_init
    bm = engine.scheduler.block_manager
    summary["pool"] = dict(
        profiled_num_gpu_blocks=cache_config.num_gpu_blocks,
        block_manager_total_blocks=bm.num_total_gpu_blocks,
        watermark=bm.watermark, watermark_blocks=bm.watermark_blocks,
        free_blocks_before_requests=bm.get_num_free_gpu_blocks(),
        engine_init_seconds=round(init_seconds, 3),
    )
    print("shared_after_init:", json.dumps(shared_after_init), flush=True)
    print("pool:", json.dumps(summary["pool"]), flush=True)
    return engine, to_jsonable(numerical_policy)


def run_idle(engine, observer, capture, trace):
    """One ordinary idle call after draining."""
    before = capture()
    events = observer.events = []
    idle_outputs = engine.step()
    after = capture()
    trace["idle"] = dict(before=before, events=list(events), after=after,
                         request_outputs=[request_output_record(o)
                                          for o in idle_outputs])
    check(idle_outputs == [], f"idle step returned {idle_outputs}")
    check([e["call"] for e in events] == ["schedule"],
          f"idle step calls {[e['call'] for e in events]}")
    check(events[0]["outputs"]["scheduled"] == [], "idle step scheduled work")
    check(after["iteration_id"] == before["iteration_id"] + 1,
          "idle step must advance the iteration once")
    check({**after, "iteration_id": None} == {**before, "iteration_id": None},
          f"idle step changed state: {before} -> {after}")
    print(f"idle: no outputs, calls {[e['call'] for e in events]}, "
          f"iteration {before['iteration_id']} -> {after['iteration_id']}",
          flush=True)


def run_reference(engine, label, sampling_params, trace, summary):
    """Serve one request alone with the VLLM scheduler; return its tokens."""
    scheduler = engine.scheduler
    bm = scheduler.block_manager
    n_free = bm.get_num_free_gpu_blocks()
    check(n_free == bm.num_total_gpu_blocks and not bm.block_tables,
          "block pool must be fully free before the request")
    check(n_free - REFERENCE_PEAK_BLOCKS >= bm.watermark_blocks,
          f"profiled pool {n_free} too small for watermark "
          f"{bm.watermark_blocks} + {REFERENCE_PEAK_BLOCKS}")

    seq_ref = [None]
    capture = lambda: reference_helper.capture_state(engine, seq_ref[0])
    observer = Observer()
    observer.wrap(scheduler, "schedule", lambda a, k, r: dict(
        outputs=outputs_record(r), state=capture()))
    observer.wrap(engine, "_run_workers", reference_helper.worker_record)

    before = capture()
    observer.events = []
    engine.add_request(prompt=None, sampling_params=sampling_params,
                       prompt_token_ids=list(PROMPTS[label]))
    check(len(engine.seq_manager.seq_map) == 1, "expected one sequence")
    seq_id = next(iter(engine.seq_manager.seq_map))
    seq_ref[0] = engine.seq_manager.seq_map[seq_id]
    after = capture()
    trace["submission"] = dict(label=label, seq_id=seq_id,
                               prompt_token_ids=PROMPTS[label], before=before,
                               events=list(observer.events), after=after)
    check(observer.calls() == ["_run_workers"]
          and observer.events[0]["method"] == "add_seq",
          f"submission calls {observer.events}")
    check(after["waiting"] == [seq_id] and after["running"] == []
          and after["free_blocks"] == n_free and after["block_tables"] == {}
          and after["request"]["status"] == "WAITING"
          and after["request"]["output_token_ids"] == []
          and list(seq_ref[0].prompt_token_ids) == PROMPTS[label],
          f"submitted state {after}")
    print(f"submitted {label}: seq {seq_id} prompt {PROMPTS[label]}",
          flush=True)

    finished_output = None
    decisions = []
    for index, exp in enumerate(reference_helper.EXPECTED_STEPS["vllm"]):
        tag = f"{label} step {index} {exp['name']}"
        before = capture()
        observer.events = []
        step_outputs = engine.step()
        after = capture()
        events = list(observer.events)
        outputs = [request_output_record(o) for o in step_outputs]
        record = dict(index=index, name=exp["name"], before=before,
                      events=events, after=after, request_outputs=outputs)
        trace["steps"].append(record)
        record["checked"] = reference_helper.check_step(
            tag, "vllm", exp, n_free, bm.watermark_blocks, before, events,
            after, outputs, seq_id, None)
        decisions.append(dict(name=exp["name"], **record["checked"]))
        if exp["finished"]:
            finished_output = outputs[0]
        print(f"{tag}: emitted {record['checked']['emitted']}, "
              f"+{record['checked']['blocks_added']} block(s), free "
              f"{record['checked']['free_after_schedule']} -> "
              f"{after['free_blocks']}, {after['request']['status']} "
              f"prompt={after['request']['prompt_tokens_processed']} "
              f"generated={after['request']['output_token_ids']}", flush=True)

    check(finished_output is not None
          and len(finished_output["token_ids"]) == MAX_TOKENS
          and finished_output["finish_reason"] == "length",
          "expected exactly one finished four-token length-capped output")
    check(not engine.has_unfinished_requests(), "unfinished requests remain")
    check(bm.get_num_free_gpu_blocks() == n_free and not bm.block_tables,
          "central memory not restored")
    run_idle(engine, observer, lambda: reference_helper.capture_state(
        engine, None), trace)
    summary.update(n_free=n_free, seq_id=seq_id, final_output=finished_output,
                   decisions=decisions)
    return list(finished_output["token_ids"])


def run_lp(engine, policy_json, sampling_params, trace, summary):
    """Serve A, B, and C together with the LP scheduler; return the
    generated token IDs by request label."""
    import lp_relaxation_scheduler as lrs
    import lpserve_plan_execution as lpe
    import lpserve_state_mapping as lsm

    scheduler = engine.scheduler
    bm = scheduler.block_manager
    n_free = bm.get_num_free_gpu_blocks()
    # Pool adequacy: all requests resident at their two-block peak, the
    # planning reserve, conservative decode charges (at most S_MAX), and the
    # native watermark must all fit.
    check(n_free == bm.num_total_gpu_blocks and not bm.block_tables,
          "block pool must be fully free before the requests")
    check(n_free - LP_PEAK_BLOCKS_IN_USE - S_MAX
          >= bm.watermark_blocks + MEMORY_RESERVE,
          f"profiled pool {n_free} too small for {LP_PEAK_BLOCKS_IN_USE} "
          f"blocks + {S_MAX} decode charges + reserve {MEMORY_RESERVE} "
          f"+ watermark {bm.watermark_blocks}")

    seqs = {}
    capture = lambda: contention.capture_state(engine, seqs)
    observer = Observer()
    observer.wrap(lsm, "map_scheduler_state", lambda a, k, r: dict(
        result_type=type(r).__name__, result=to_jsonable(r)))
    observer.wrap(lrs, "solve_and_extract", lambda a, k, r: dict(
        problem_id_in=a[0].problem_id,
        result_type=type(r).__name__, result=to_jsonable(r)))
    observer.wrap(lpe, "execute_plan", lambda a, k, r: dict(
        snapshot_id_in=a[1].snapshot_id, problem_id_in=a[2].problem_id,
        result_type=type(r).__name__,
        result=(outputs_record(r) if type(r).__name__ == "SchedulerOutputs"
                else to_jsonable(r))))
    observer.wrap(scheduler, "schedule", lambda a, k, r: dict(
        outputs=contention.outputs_with_counts(r), state=capture()))
    observer.wrap(engine, "_run_workers", contention.worker_record)

    ids = {}
    for label in PROMPTS:
        before = capture()
        events = observer.events = []
        known = set(engine.seq_manager.seq_map)
        engine.add_request(prompt=None, sampling_params=sampling_params,
                           prompt_token_ids=list(PROMPTS[label]))
        new = set(engine.seq_manager.seq_map) - known
        check(len(new) == 1, f"add {label}: new sequences {new}")
        ids[label] = new.pop()
        seqs[label] = engine.seq_manager.seq_map[ids[label]]
        after = capture()
        trace["add_requests"].append(dict(
            label=label, seq_id=ids[label], prompt_token_ids=PROMPTS[label],
            before=before, events=list(events), after=after))
        check([(e["call"], e["method"]) for e in events]
              == [("_run_workers", "add_seq")],
              f"add {label}: unexpected calls {events}")
        check(after["waiting"] == before["waiting"] + [ids[label]]
              and after["running"] == []
              and after["iteration_id"] == before["iteration_id"]
              and after["free_blocks"] == n_free
              and after["requests"][label]["block_ids"] is None
              and after["requests"][label]["status"] == "WAITING"
              and list(seqs[label].prompt_token_ids) == PROMPTS[label],
              f"add {label}: state after submission {after}")
        print(f"added {label}: seq_id {ids[label]}, waiting "
              f"{after['waiting']}", flush=True)
    summary["seq_ids"] = ids

    finished_outputs = []
    witness = dict(waiting_omitted=[], all_resident_boundaries=[],
                   later_selected=[])
    summary["witnesses"] = witness
    step = 0
    while engine.has_unfinished_requests():
        check(step < MAX_NONEMPTY_STEPS,
              f"termination guard of {MAX_NONEMPTY_STEPS} steps reached")
        step += 1
        tag = f"step {step}"
        before = capture()
        record = dict(step=step, before=before)
        trace["steps"].append(record)
        events = record["events"] = observer.events = []
        step_outputs = engine.step()
        after = capture()
        record.update(after=after, request_outputs=[
            request_output_record(o) for o in step_outputs])
        result = contention.check_step(tag, ids, before, events, after,
                                       step_outputs, policy_json)
        record["checked"] = result
        finished_outputs.extend(o for o in step_outputs if o.finished)

        if result["omitted_waiting"]:
            witness["waiting_omitted"].append(
                dict(step=step, omitted=result["omitted_waiting"],
                     selected=result["selected"]))
        earlier = {n for b in witness["all_resident_boundaries"]
                   for n in b["omitted"]}
        if earlier & set(result["selected"]):
            witness["later_selected"].append(
                dict(step=step,
                     requests=sorted(earlier & set(result["selected"]))))
        if result["all_prompt_complete_residents"]:
            check(result["omitted_resident"],
                  f"{tag}: no resident omitted at a three-resident boundary")
            witness["all_resident_boundaries"].append(
                dict(step=step, selected=result["selected"],
                     omitted=result["omitted_resident"]))
        print(f"{tag}: plan {result['plan']}, emitted {result['emitted']}, "
              f"omitted {result['omitted']}, sampler "
              f"{result['sampler_outputs']}, free {result['free']}, work "
              f"{result['work']}, finished {result['finished']}", flush=True)

    check(witness["waiting_omitted"],
          "no eligible waiting request was omitted")
    check(witness["all_resident_boundaries"],
          "no boundary had three prompt-complete allocated residents")
    check(witness["later_selected"],
          "no resident omitted at such a boundary was later selected")

    final = capture()
    check(len(finished_outputs) == len(PROMPTS)
          and sorted(o.seq_id for o in finished_outputs)
          == sorted(ids.values())
          and all(len(o.token_ids) == MAX_TOKENS
                  and o.finish_reason == "length" for o in finished_outputs),
          "expected one finished four-token output per request")
    check(all(r["status"] == FINISHED and r["generated"] == MAX_TOKENS
              for r in final["requests"].values()),
          f"final request states {final['requests']}")
    check(not engine.has_unfinished_requests(), "unfinished requests remain")
    check(final["waiting"] == [] and final["running"] == []
          and final["block_tables"] == {} and final["engine_seq_ids"] == []
          and final["free_blocks"] == n_free, f"drained state {final}")
    run_idle(engine, observer, capture, trace)

    labels = {seq_id: label for label, seq_id in ids.items()}
    by_label = {labels[o.seq_id]: request_output_record(o)
                for o in finished_outputs}
    summary.update(
        n_free=n_free, nonempty_steps=step,
        finished_outputs=by_label,
        decisions=[dict(step=s["step"], **{k: s["checked"][k] for k in (
            "selected", "omitted", "plan", "emitted", "sampler_outputs",
            "free", "work", "finished")}) for s in trace["steps"]],
    )
    return {label: list(by_label[label]["token_ids"]) for label in PROMPTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheduler", required=True, choices=["vllm", "lp"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request", choices=sorted(PROMPTS))
    parser.add_argument("--references-dir")
    args = parser.parse_args()
    mode = args.scheduler
    if (mode == "vllm") != (args.request is not None):
        parser.error("--request is required for --scheduler vllm and not "
                     "accepted for --scheduler lp")
    if (mode == "lp") != (args.references_dir is not None):
        parser.error("--references-dir is required for --scheduler lp and "
                     "not accepted for --scheduler vllm")

    # The directory may already exist (holding the console log), but earlier
    # run artifacts are never overwritten.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.json"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")

    summary = dict(scheduler=mode, request=args.request, passed=False,
                   failure=None, command=sys.argv,
                   provenance=provenance_record())
    if mode == "vllm":
        summary["request_prompt_token_ids"] = PROMPTS[args.request]
        trace = dict(submission=None, steps=[], idle=None)
    else:
        summary.update(execution_passed=False, comparison_passed=False,
                       comparisons=None, witnesses=None)
        trace = dict(add_requests=[], steps=[], idle=None)
    engine = None
    try:
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import the LP modules")
        check(all(os.environ.get(k) == "1" for k in OFFLINE_ENV),
              f"{OFFLINE_ENV} must both be set to 1")
        for module, name, value in HELPER_CONSTANTS:
            check(getattr(module, name) == value,
                  f"helper {module.__name__}.{name} is "
                  f"{getattr(module, name)!r}, expected {value!r}")

        # Assets and references are checked before any engine exists.
        asset = reference_helper.inspect_snapshot(args.model_path)
        summary["asset"] = asset
        shared = shared_inputs(asset)
        summary["shared"] = to_jsonable(shared)
        print("asset:", json.dumps(asset), flush=True)
        references = None
        if mode == "lp":
            references = load_references(args.references_dir,
                                         summary["provenance"], shared)
            summary["references"] = references
            print("references:", json.dumps(references), flush=True)

        engine, policy_json = build_engine(mode, asset, out_dir, summary)
        if references is not None:
            for label, record in references.items():
                check(record["shared_after_init"]
                      == summary["shared_after_init"],
                      f"reference {label} model/tokenizer facts differ")

        from sarathi.core.datatypes.sampling_params import (SamplingParams,
                                                            SamplingType)
        sampling_params = SamplingParams(temperature=TEMPERATURE,
                                         max_tokens=MAX_TOKENS,
                                         ignore_eos=IGNORE_EOS, stop=None)
        check(sampling_params.sampling_type == SamplingType.GREEDY,
              f"sampling type {sampling_params.sampling_type}")

        if mode == "vllm":
            generated = run_reference(engine, args.request, sampling_params,
                                      trace, summary)
            summary.update(generated_token_ids=generated, passed=True)
            print(f"{args.request} generated token ids: {generated}",
                  flush=True)
        else:
            generated = run_lp(engine, policy_json, sampling_params, trace,
                               summary)
            summary["generated_token_ids"] = generated
            summary["execution_passed"] = True
            comparisons = {label: reference_helper.compare_tokens(
                references[label]["generated_token_ids"], generated[label])
                for label in PROMPTS}
            summary["comparisons"] = comparisons
            summary["comparison_passed"] = all(c["equal"]
                                               for c in comparisons.values())
            summary["passed"] = summary["comparison_passed"]
            for label, c in comparisons.items():
                print(f"comparison {label}:", json.dumps(c), flush=True)
            check(summary["comparison_passed"],
                  "LP generated token ids differ from the reference for "
                  + ", ".join(f"{label} at index "
                              f"{c['first_difference_index']}"
                              for label, c in comparisons.items()
                              if not c["equal"]))
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        # The engine is never reused after a failure.
        engine = None
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
              ("FAIL" if summary["comparisons"] else "NOT PERFORMED"),
              flush=True)
    print("RESULT:", "PASS" if summary["passed"] else "FAIL", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
