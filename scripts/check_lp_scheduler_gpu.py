"""Bounded GPU correctness check for the live LP scheduler.

Runs one request through a real ``BaseLLMEngine`` with the registered
``LPScheduler``, one GPU worker, dummy TinyLlama weights, and native
completion. It asserts the exact expected schedule and state transitions for
four nonempty decisions (admission prefill, resident prefill, decode without
allocation, decode with one block append) and one idle call after draining.

It does not establish generation quality, numerical equivalence, mixed-batch
correctness, central/worker block-table equality, or performance.

Run from the repository root with the root on PYTHONPATH (the Ray worker must
unpickle the LP numerical policy from the root-level module):

    PYTHONPATH="$PWD" timeout 300s python -B scripts/check_lp_scheduler_gpu.py \\
        --output-dir validation_output/lp_scheduler_gpu/<new timestamp>

Exit status is zero only if every assertion passes.
"""

import argparse
import dataclasses
import enum
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Approved scoped test inputs (provisional, not project-wide policy).
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "dummy"
DTYPE = "float16"  # existing benchmark-runner convention
ATTENTION_BACKEND = "flash_attention"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 32
BLOCK_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.5
SEED = 42
TRUST_REMOTE_CODE = True  # existing benchmark-runner convention
TOKENIZER_MODE = "auto"
PROMPT_LEN = 16
PROMPT_ID_SEARCH_START = 1000
TEMPERATURE = 0.0
MAX_TOKENS = 2
IGNORE_EOS = True
B_MAX = 32
C_MAX = 8
S_MAX = 1
MAX_NUM_SEQS = 1
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

# Expected nonempty decisions. ``free_delta`` is relative to N, the central
# free-block count before the request is added.
EXPECTED_STEPS = [
    dict(name="admission_prefill", prefill=8, decode=0, sched_physical=1,
         sched_free_delta=-1, prompt_done=8, prompt_finished=False,
         generated=0, status="PAUSED", final_free_delta=-1,
         final_physical=1, final_logical=1, finished=False),
    dict(name="resident_prefill", prefill=8, decode=0, sched_physical=1,
         sched_free_delta=-1, prompt_done=16, prompt_finished=True,
         generated=0, status="PAUSED", final_free_delta=-1,
         final_physical=1, final_logical=1, finished=False),
    dict(name="decode_without_allocation", prefill=0, decode=1,
         sched_physical=1, sched_free_delta=-1, prompt_done=16,
         prompt_finished=True, generated=1, status="PAUSED",
         final_free_delta=-1, final_physical=1, final_logical=2,
         finished=False),
    dict(name="decode_with_block_append", prefill=0, decode=1,
         sched_physical=2, sched_free_delta=-2, prompt_done=16,
         prompt_finished=True, generated=2,
         status="FINISHED_LENGTH_CAPPED", final_free_delta=0,
         final_physical=None, final_logical=2, finished=True),
]


class CheckFailure(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise CheckFailure(message)


def to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, (frozenset, set)):
        return sorted((to_jsonable(x) for x in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def outputs_record(outputs):
    return dict(
        id=outputs.id,
        scheduled=[[m.seq_id, m.prompt_chunk_len]
                   for m in outputs.scheduled_seq_metadata_list],
        ignored_seq_ids=list(outputs.ignored_seq_ids),
        preempted_seq_ids=list(outputs.preempted_seq_ids),
    )


def request_output_record(ro):
    return dict(seq_id=ro.seq_id, prompt_token_ids=list(ro.prompt_token_ids),
                token_ids=list(ro.token_ids), text=ro.text,
                finished=ro.finished, finish_reason=ro.finish_reason)


def capture_state(engine, seq):
    """Selected primitive state; holds no live objects."""
    scheduler = engine.scheduler
    bm = scheduler.block_manager
    return dict(
        iteration_id=scheduler._iteration_id,
        num_running_batches=scheduler.num_running_batches,
        waiting=[s.seq_id for s in scheduler.waiting],
        running=[s.seq_id for s in scheduler.running],
        free_blocks=bm.get_num_free_gpu_blocks(),
        block_tables={str(k): len(v) for k, v in bm.block_tables.items()},
        engine_seq_ids=sorted(engine.seq_manager.seq_map),
        num_unfinished=engine.get_num_unfinished_requests(),
        request=None if seq is None else dict(
            seq_id=seq.seq_id,
            status=seq.get_status().name,
            prompt_tokens_processed=seq.get_num_prompt_tokens_processed(),
            prompt_processing_finished=seq.prompt_processing_finished,
            generated=seq.get_output_len(),
            output_token_ids=list(seq.get_output_token_ids()),
            logical_blocks=len(seq.logical_token_blocks),
            physical_blocks=(len(bm.block_tables[seq.seq_id])
                             if seq.seq_id in bm.block_tables else 0),
            is_finished=seq.is_finished(),
        ),
    )


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=30).stdout.strip()
    except Exception as err:  # provenance only; never affects the check
        return f"unavailable: {err!r}"


def environment_record():
    versions = {}
    for dist in ["torch", "transformers", "ray", "numpy", "scipy",
                 "flashinfer", "vllm-flash-attn", "sarathi", "nvidia-ml-py",
                 "huggingface-hub", "tokenizers"]:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    import torch
    return dict(
        hostname=socket.gethostname(),
        slurm={k: os.environ.get(k) for k in [
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
            "SLURM_STEP_ID"]},
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,"
                 "memory.total", "--format=csv,noheader"]),
        torch_cuda=torch.version.cuda,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        virtual_env=os.environ.get("VIRTUAL_ENV"),
        pythonpath=os.environ.get("PYTHONPATH"),
        versions=versions,
        git_head=run(["git", "rev-parse", "HEAD"]),
        git_status_short=run(["git", "status", "--short",
                              "--untracked-files=all"]),
        cwd=os.getcwd(),
    )


def select_prompt_ids(tokenizer, vocab_size):
    special = set(tokenizer.all_special_ids)
    ids = []
    candidate = PROMPT_ID_SEARCH_START
    while len(ids) < PROMPT_LEN:
        token = tokenizer.convert_ids_to_tokens(candidate)
        if (candidate < vocab_size and candidate < len(tokenizer)
                and candidate not in special and token is not None
                and token != tokenizer.unk_token):
            ids.append(candidate)
        candidate += 1
    return ids, tokenizer.convert_ids_to_tokens(ids)


class Observer:
    """Records calls and immutable results; delegates exactly once and
    returns the original result unchanged."""

    def __init__(self):
        self.events = []

    def wrap(self, owner, name, record):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            self.events.append(dict(call=name, **record(args, kwargs, result)))
            return result

        setattr(owner, name, wrapper)

    def calls(self):
        return [e["call"] for e in self.events]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # The directory may already exist (e.g. holding the console log), but
    # earlier run artifacts are never overwritten.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.json"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")

    trace = dict(steps=[], idle=None)
    summary = dict(passed=False, failure=None)
    engine = None
    try:
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import the LP modules")

        import lp_relaxation_scheduler as lrs
        import lpserve_plan_execution as lpe
        import lpserve_state_mapping as lsm
        from sarathi.config import (CacheConfig, LPSchedulerConfig,
                                    MetricsConfig, ModelConfig,
                                    ParallelConfig)
        from sarathi.core.datatypes.sampling_params import SamplingParams
        from sarathi.core.scheduler.lp_scheduler import LPScheduler
        from sarathi.engine.base_llm_engine import BaseLLMEngine

        numerical_policy = lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS)
        model_config = ModelConfig(
            model=MODEL, tokenizer=MODEL, tokenizer_mode=TOKENIZER_MODE,
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
            num_pipeline_stages=PIPELINE_PARALLEL_SIZE, b_max=B_MAX,
            c_max=C_MAX, s_max=S_MAX, memory_reserve=MEMORY_RESERVE,
            decode_memory_policy_id=DECODE_POLICY_ID,
            decode_utility=DECODE_UTILITY,
            prefill_token_utility=PREFILL_TOKEN_UTILITY,
            preemption_penalty=PREEMPTION_PENALTY,
            numerical_policy=numerical_policy,
        )
        # The disabled MetricsStore mode cannot be used: BaseLLMEngine calls
        # record_block_util/record_active_gpu_seqs on every step, and those
        # read attributes that only the enabled mode initializes. Use the
        # enabled mode with every optional output off; nothing is plotted or
        # written because plot() is never called.
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
        check(scheduler_config.max_num_batched_tokens == B_MAX,
              "profiling token count must equal b_max")

        summary["environment"] = environment_record()
        print("environment:", json.dumps(summary["environment"], indent=1),
              flush=True)

        t0 = time.monotonic()
        engine = BaseLLMEngine(model_config, cache_config, parallel_config,
                               scheduler_config, metrics_config)
        init_seconds = time.monotonic() - t0
        scheduler = engine.scheduler
        bm = scheduler.block_manager
        check(type(scheduler) is LPScheduler,
              f"registered scheduler is {type(scheduler).__name__}")

        vocab_size = model_config.hf_config.vocab_size
        prompt_ids, prompt_tokens = select_prompt_ids(engine.tokenizer,
                                                      vocab_size)
        n_free = bm.get_num_free_gpu_blocks()
        config_record = dict(
            model=MODEL, tokenizer=MODEL, load_format=LOAD_FORMAT,
            requested_dtype=DTYPE, resolved_dtype=str(model_config.dtype),
            hf_config_torch_dtype=str(getattr(model_config.hf_config,
                                              "torch_dtype", None)),
            hf_config_commit_hash=getattr(model_config.hf_config,
                                          "_commit_hash", None),
            attention_backend=ATTENTION_BACKEND,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
            max_model_len=model_config.max_model_len, block_size=BLOCK_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=SEED,
            trust_remote_code=TRUST_REMOTE_CODE, tokenizer_mode=TOKENIZER_MODE,
            vocab_size=vocab_size, tokenizer_len=len(engine.tokenizer),
            eos_token_id=engine.tokenizer.eos_token_id,
            prompt_token_ids=prompt_ids, prompt_tokens=prompt_tokens,
            sampling=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                          ignore_eos=IGNORE_EOS, stop=[]),
            b_max=B_MAX, c_max=C_MAX, s_max=S_MAX, max_num_seqs=MAX_NUM_SEQS,
            memory_reserve=MEMORY_RESERVE,
            decode_memory_policy_id=DECODE_POLICY_ID,
            utilities=dict(decode=DECODE_UTILITY,
                           prefill_token=PREFILL_TOKEN_UTILITY,
                           preemption_penalty=PREEMPTION_PENALTY),
            numerical_policy=to_jsonable(numerical_policy),
            profiling_max_num_batched_tokens=(
                scheduler_config.max_num_batched_tokens),
            profiling_max_num_seqs=scheduler_config.max_num_seqs,
            metrics_mode="enabled, all optional outputs off, never plotted",
            profiled_num_gpu_blocks=cache_config.num_gpu_blocks,
            block_manager_total_blocks=bm.num_total_gpu_blocks,
            watermark=bm.watermark, watermark_blocks=bm.watermark_blocks,
            free_blocks_before_request=n_free,
            engine_init_seconds=round(init_seconds, 3),
        )
        summary["config"] = config_record
        print("config:", json.dumps(config_record), flush=True)

        # Pool adequacy: two blocks in use at the peak, the planning reserve,
        # and the native watermark must all fit.
        check(n_free == bm.num_total_gpu_blocks and not bm.block_tables,
              "block pool must be fully free before the request")
        check(n_free - 2 >= bm.watermark_blocks + MEMORY_RESERVE,
              f"profiled pool {n_free} too small for watermark "
              f"{bm.watermark_blocks} + reserve {MEMORY_RESERVE} + 2 blocks")
        check(len(prompt_ids) == PROMPT_LEN, "prompt length mismatch")

        observer = Observer()
        observer.wrap(lsm, "map_scheduler_state", lambda a, k, r: dict(
            result_type=type(r).__name__, result=to_jsonable(r)))
        observer.wrap(lrs, "solve_and_extract", lambda a, k, r: dict(
            result_type=type(r).__name__, result=to_jsonable(r)))
        observer.wrap(lpe, "execute_plan", lambda a, k, r: dict(
            result_type=type(r).__name__,
            result=(outputs_record(r) if type(r).__name__ == "SchedulerOutputs"
                    else to_jsonable(r))))
        observer.wrap(scheduler, "schedule", lambda a, k, r: dict(
            outputs=outputs_record(r), state=capture_state(engine, seq_ref[0])))
        observer.wrap(engine, "_run_workers", lambda a, k, r: dict(
            method=a[0]))

        seq_ref = [None]
        engine.add_request(
            prompt=None,
            sampling_params=SamplingParams(temperature=TEMPERATURE,
                                           max_tokens=MAX_TOKENS,
                                           ignore_eos=IGNORE_EOS, stop=None),
            prompt_token_ids=list(prompt_ids),
        )
        check(len(engine.seq_manager.seq_map) == 1, "expected one sequence")
        seq_id = next(iter(engine.seq_manager.seq_map))
        seq_ref[0] = engine.seq_manager.seq_map[seq_id]
        observer.events.clear()  # drop the add_seq worker call
        finished_outputs = []

        for index, exp in enumerate(EXPECTED_STEPS):
            name = exp["name"]
            before = capture_state(engine, seq_ref[0])
            check(before["num_running_batches"] == 0,
                  f"{name}: running batches before scheduling")
            observer.events.clear()
            step_outputs = engine.step()
            after = capture_state(engine, seq_ref[0])
            events = list(observer.events)
            record = dict(index=index, name=name, before=before,
                          events=events, after=after,
                          request_outputs=[request_output_record(o)
                                           for o in step_outputs])
            trace["steps"].append(record)

            check(observer.calls() == [
                "map_scheduler_state", "solve_and_extract", "execute_plan",
                "schedule", "_run_workers"],
                f"{name}: unexpected call sequence {observer.calls()}")
            check(events[4]["method"] == "execute_model",
                  f"{name}: worker call {events[4]['method']}")
            snap = events[0]
            check(snap["result_type"] == "StateSnapshot",
                  f"{name}: mapping returned {snap['result_type']}")
            snap = snap["result"]
            problem = snap["lp_problem"]
            check((problem["b_max"], problem["c_max"], problem["s_max"]) ==
                  (B_MAX, C_MAX, S_MAX), f"{name}: mapped limits")
            check(snap["memory_reserve"] == MEMORY_RESERVE
                  and snap["resident_limit"] == MAX_NUM_SEQS
                  and snap["decode_memory_policy_id"] == DECODE_POLICY_ID
                  and snap["numerical_policy"] == to_jsonable(numerical_policy),
                  f"{name}: mapped policy inputs")
            check(snap["free_physical_blocks"] == before["free_blocks"],
                  f"{name}: mapped free blocks")
            check(len(snap["requests"]) == 1
                  and snap["requests"][0]["raw_seq_id"] == seq_id,
                  f"{name}: mapped request set")
            check(snap["requests"][0]["utility"] == dict(
                decode_utility=DECODE_UTILITY,
                prefill_token_utility=PREFILL_TOKEN_UTILITY,
                preemption_penalty=PREEMPTION_PENALTY),
                f"{name}: mapped utilities")

            solved = events[1]
            check(solved["result_type"] == "SchedulingSuccess",
                  f"{name}: solve returned {solved['result_type']}: "
                  f"{solved['result']}")
            plan = solved["result"]["plan"]
            check(len(plan["decisions"]) == 1, f"{name}: plan decision count")
            decision = plan["decisions"][0]
            check(decision["request_id"] == snap["requests"][0]["request_id"],
                  f"{name}: plan request id")
            check((decision["prefill_tokens"], decision["decode"],
                   decision["preempt"]) == (exp["prefill"], exp["decode"], 0),
                  f"{name}: plan {decision} differs from expected "
                  f"prefill {exp['prefill']} decode {exp['decode']}")
            check(not plan["dominant_preemption_ids"]
                  and not plan["safety_preemption_ids"],
                  f"{name}: plan preemption ids")

            executed = events[2]
            check(executed["result_type"] == "SchedulerOutputs",
                  f"{name}: executor returned {executed['result_type']}")
            emitted = events[3]["outputs"]
            check(executed["result"] == emitted,
                  f"{name}: scheduler output differs from executor output")
            check(emitted["id"] == before["iteration_id"] + 1
                  and emitted["id"] == snap["scheduler_iteration_id"],
                  f"{name}: decision id {emitted['id']}")
            check(emitted["scheduled"] == [[seq_id, exp["prefill"]]],
                  f"{name}: emitted {emitted['scheduled']} differs from plan")
            check(not emitted["ignored_seq_ids"]
                  and not emitted["preempted_seq_ids"],
                  f"{name}: ignored/preempted ids emitted")

            sched = events[3]["state"]
            req = sched["request"]
            check(sched["num_running_batches"] == 1,
                  f"{name}: running batches after scheduling")
            check(sched["waiting"] == [] and sched["running"] == [seq_id],
                  f"{name}: ownership after scheduling")
            check(req["physical_blocks"] == exp["sched_physical"]
                  and sched["free_blocks"] == n_free + exp["sched_free_delta"],
                  f"{name}: blocks after scheduling {req['physical_blocks']} "
                  f"physical, {sched['free_blocks']} free")
            check(req["prompt_tokens_processed"] == before["request"][
                "prompt_tokens_processed"]
                and req["generated"] == before["request"]["generated"],
                f"{name}: progress changed before completion")

            req = after["request"]
            check(after["iteration_id"] == before["iteration_id"] + 1,
                  f"{name}: iteration advanced "
                  f"{after['iteration_id'] - before['iteration_id']}")
            check(after["num_running_batches"] == 0,
                  f"{name}: running batches after completion")
            check(req["prompt_tokens_processed"] == exp["prompt_done"]
                  and req["prompt_processing_finished"]
                  == exp["prompt_finished"]
                  and req["generated"] == exp["generated"]
                  and req["status"] == exp["status"]
                  and req["logical_blocks"] == exp["final_logical"],
                  f"{name}: request state after completion {req}")
            check(after["free_blocks"] == n_free + exp["final_free_delta"],
                  f"{name}: free blocks after completion "
                  f"{after['free_blocks']}")
            check(len(step_outputs) == 1
                  and step_outputs[0].seq_id == seq_id
                  and step_outputs[0].finished == exp["finished"]
                  and len(step_outputs[0].token_ids) == exp["generated"],
                  f"{name}: request outputs {record['request_outputs']}")
            if exp["finished"]:
                check(after["waiting"] == [] and after["running"] == []
                      and after["block_tables"] == {}
                      and after["engine_seq_ids"] == []
                      and after["num_unfinished"] == 0,
                      f"{name}: drained state {after}")
                check(step_outputs[0].finish_reason == "length",
                      f"{name}: finish reason {step_outputs[0].finish_reason}")
                finished_outputs.extend(step_outputs)
            else:
                check(after["waiting"] == [] and after["running"] == [seq_id]
                      and req["physical_blocks"] == exp["final_physical"]
                      and after["engine_seq_ids"] == [seq_id],
                      f"{name}: resident state after completion {after}")
            print(f"step {index} {name}: emitted {emitted['scheduled']}, "
                  f"plan prefill={decision['prefill_tokens']} "
                  f"decode={decision['decode']}, scheduled "
                  f"{sched['request']['physical_blocks']} physical / "
                  f"{sched['free_blocks']} free, after {req['status']} "
                  f"prompt={req['prompt_tokens_processed']} "
                  f"generated={req['generated']} free={after['free_blocks']}",
                  flush=True)

        check(len(finished_outputs) == 1
              and len(finished_outputs[0].token_ids) == MAX_TOKENS,
              "expected exactly one finished output with two tokens")
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

        summary.update(
            passed=True,
            n_free=n_free,
            seq_id=seq_id,
            final_output=request_output_record(finished_outputs[0]),
            decisions=[dict(name=s["name"],
                            emitted=s["events"][3]["outputs"]["scheduled"])
                       for s in trace["steps"]],
        )
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        trace_path.write_text(json.dumps(trace, indent=1))
        summary_path.write_text(json.dumps(summary, indent=1))
        # Stop only the Ray runtime this process started (no-op otherwise).
        if "ray" in sys.modules:
            sys.modules["ray"].shutdown()
    print("RESULT:", "PASS" if summary["passed"] else "FAIL", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
