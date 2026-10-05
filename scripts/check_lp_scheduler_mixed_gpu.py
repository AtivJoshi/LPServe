"""Bounded two-request mixed-batch GPU correctness check for the LP scheduler.

Runs requests A and B through a real ``BaseLLMEngine`` with the registered
``LPScheduler``, one GPU worker, dummy TinyLlama weights, greedy sampling for
both requests, and native completion. A is added first and prefilled in two
chunks; B is added only after the second step returns. The next two steps
must emit B's prefill chunk before A's decode (prompt-first, even though A has
the smaller order key), then B decodes alone. The script asserts the exact
schedule, ownership, progress, and central block changes for six nonempty
steps and one idle call.

It does not establish generation quality, numerical agreement with a
reference, mixed sampling methods, arbitrary workloads, central/worker
block-table equality, preemption, pipeline execution, or performance.

Run from the repository root with the root on PYTHONPATH (the Ray worker must
unpickle the LP numerical policy from the root-level module):

    PYTHONPATH="$PWD" timeout 300s python -B \\
        scripts/check_lp_scheduler_mixed_gpu.py \\
        --output-dir validation_output/lp_scheduler_mixed_gpu/<new timestamp>

Exit status is zero only if every assertion passes.
"""

import argparse
import hashlib
import json
import os
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

# Approved scoped test inputs (provisional, not project-wide policy).
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "dummy"
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
PROMPT_LEN = 16
# Distinct search starts give disjoint prompt-token lists for A and B.
PROMPT_ID_SEARCH_START = dict(A=1000, B=2000)
TEMPERATURE = 0.0
MAX_TOKENS = 2
IGNORE_EOS = True
# b_max=64 with max_num_seqs=2 makes the generic profiler build two 32-token
# prompts; actual prefill chunks stay bounded by c_max.
B_MAX = 64
C_MAX = 8
S_MAX = 2
MAX_NUM_SEQS = 2
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
ADD_B_AFTER_STEP = 2
PEAK_BLOCKS_IN_USE = 3


def req(prompt, generated, status, logical, physical):
    return dict(prompt=prompt, prompt_finished=prompt == PROMPT_LEN,
                generated=generated, status=status, logical=logical,
                physical=physical)


FINISHED = "FINISHED_LENGTH_CAPPED"

# Expected nonempty steps, numbered from 1. Requests are named "A"/"B";
# ``free_delta`` values are relative to N, the free-block count before A is
# added. ``plan`` maps request -> (prefill_tokens, decode); preempt is 0.
# ``outputs`` lists (request, generated tokens, finished) in returned order.
EXPECTED_STEPS = [
    dict(step=1, name="a_admission_prefill", plan=dict(A=(8, 0)),
         emitted=[("A", 8)], sched_running=["A"], sched_physical=dict(A=1),
         sched_free_delta=-1, running=["A"], free_delta=-1,
         requests=dict(A=req(8, 0, "PAUSED", 1, 1)),
         outputs=[("A", 0, False)]),
    dict(step=2, name="a_resident_prefill", plan=dict(A=(8, 0)),
         emitted=[("A", 8)], sched_running=["A"], sched_physical=dict(A=1),
         sched_free_delta=-1, running=["A"], free_delta=-1,
         requests=dict(A=req(16, 0, "PAUSED", 1, 1)),
         outputs=[("A", 0, False)]),
    dict(step=3, name="b_admission_prefill_with_a_decode",
         plan=dict(A=(0, 1), B=(8, 0)), emitted=[("B", 8), ("A", 0)],
         sched_running=["A", "B"], sched_physical=dict(A=1, B=1),
         sched_free_delta=-2, running=["A", "B"], free_delta=-2,
         requests=dict(A=req(16, 1, "PAUSED", 2, 1),
                       B=req(8, 0, "PAUSED", 1, 1)),
         outputs=[("B", 0, False), ("A", 1, False)]),
    dict(step=4, name="b_resident_prefill_with_a_append_decode",
         plan=dict(A=(0, 1), B=(8, 0)), emitted=[("B", 8), ("A", 0)],
         sched_running=["A", "B"], sched_physical=dict(A=2, B=1),
         sched_free_delta=-3, running=["B"], free_delta=-1,
         requests=dict(A=req(16, 2, FINISHED, 2, 0),
                       B=req(16, 0, "PAUSED", 1, 1)),
         outputs=[("B", 0, False), ("A", 2, True)]),
    dict(step=5, name="b_decode_without_allocation", plan=dict(B=(0, 1)),
         emitted=[("B", 0)], sched_running=["B"], sched_physical=dict(B=1),
         sched_free_delta=-1, running=["B"], free_delta=-1,
         requests=dict(A=req(16, 2, FINISHED, 2, 0),
                       B=req(16, 1, "PAUSED", 2, 1)),
         outputs=[("B", 1, False)]),
    dict(step=6, name="b_decode_with_block_append", plan=dict(B=(0, 1)),
         emitted=[("B", 0)], sched_running=["B"], sched_physical=dict(B=2),
         sched_free_delta=-2, running=[], free_delta=0,
         requests=dict(A=req(16, 2, FINISHED, 2, 0),
                       B=req(16, 2, FINISHED, 2, 0)),
         outputs=[("B", 2, True)]),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_prompt_ids(tokenizer, vocab_size, start):
    special = set(tokenizer.all_special_ids)
    ids = []
    candidate = start
    while len(ids) < PROMPT_LEN:
        token = tokenizer.convert_ids_to_tokens(candidate)
        if (candidate < vocab_size and candidate < len(tokenizer)
                and candidate not in special and token is not None
                and token != tokenizer.unk_token):
            ids.append(candidate)
        candidate += 1
    return ids, tokenizer.convert_ids_to_tokens(ids)


def capture_state(engine, seqs):
    """Selected primitive state for every added request; no live objects."""
    scheduler = engine.scheduler
    bm = scheduler.block_manager
    requests = {}
    for name, seq in seqs.items():
        requests[name] = dict(
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
        )
    return dict(
        iteration_id=scheduler._iteration_id,
        num_running_batches=scheduler.num_running_batches,
        waiting=[s.seq_id for s in scheduler.waiting],
        running=[s.seq_id for s in scheduler.running],
        free_blocks=bm.get_num_free_gpu_blocks(),
        block_tables={str(k): len(v) for k, v in bm.block_tables.items()},
        engine_seq_ids=sorted(engine.seq_manager.seq_map),
        num_unfinished=engine.get_num_unfinished_requests(),
        requests=requests,
    )


def worker_record(args, kwargs, result):
    record = dict(method=args[0])
    if args[0] == "execute_model":
        record["sampler_outputs"] = [[o.seq_id, o.output_token]
                                     for o in result]
    return record


def check_step(exp, ids, n_free, before, events, sched, after, step_outputs,
               policy_json):
    tag = f"step {exp['step']} {exp['name']}"
    calls = [e["call"] for e in events]
    check(calls == ["map_scheduler_state", "solve_and_extract",
                    "execute_plan", "schedule", "_run_workers"],
          f"{tag}: unexpected call sequence {calls}")
    mapped, solved, executed, scheduled, worker = events
    check(worker["method"] == "execute_model",
          f"{tag}: worker call {worker['method']}")
    check(before["num_running_batches"] == 0,
          f"{tag}: running batches before scheduling")

    # Mapping.
    check(mapped["result_type"] == "StateSnapshot",
          f"{tag}: mapping returned {mapped['result_type']}: "
          f"{mapped['result']}")
    snap = mapped["result"]
    problem = snap["lp_problem"]
    check((problem["b_max"], problem["c_max"], problem["s_max"]) ==
          (B_MAX, C_MAX, S_MAX), f"{tag}: mapped limits")
    check(snap["memory_reserve"] == MEMORY_RESERVE
          and snap["resident_limit"] == MAX_NUM_SEQS
          and snap["decode_memory_policy_id"] == DECODE_POLICY_ID
          and snap["numerical_policy"] == policy_json,
          f"{tag}: mapped policy inputs")
    check(snap["free_physical_blocks"] == before["free_blocks"],
          f"{tag}: mapped free blocks")
    present = sorted(exp["plan"], key=lambda n: ids[n])
    check([r["raw_seq_id"] for r in snap["requests"]]
          == [ids[n] for n in present],
          f"{tag}: mapped request set {snap['requests']}")
    check([r["request_id"] for r in problem["requests"]]
          == [r["request_id"] for r in snap["requests"]],
          f"{tag}: problem request order")
    for r in snap["requests"]:
        check(r["utility"] == dict(decode_utility=DECODE_UTILITY,
                                   prefill_token_utility=PREFILL_TOKEN_UTILITY,
                                   preemption_penalty=PREEMPTION_PENALTY),
              f"{tag}: mapped utilities")
    rid = {r["raw_seq_id"]: r["request_id"] for r in snap["requests"]}
    if len(present) == 2:
        keys = {r["raw_seq_id"]: r["order_key"] for r in snap["requests"]}
        check(keys[ids["A"]] < keys[ids["B"]],
              f"{tag}: A must have the smaller order key, got {keys}")

    # Solve/extraction and identities.
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
    got = {d["request_id"]: (d["prefill_tokens"], d["decode"], d["preempt"])
           for d in plan["decisions"]}
    want = {rid[ids[n]]: (*exp["plan"][n], 0) for n in present}
    check(got == want, f"{tag}: plan {got} differs from expected {want}")
    check(not plan["dominant_preemption_ids"]
          and not plan["safety_preemption_ids"],
          f"{tag}: plan preemption ids")
    selected = [v for v in got.values() if v[0] > 0 or v[1]]
    check(sum(p + d for p, d, _ in got.values()) <= B_MAX
          and all(p <= C_MAX for p, _, _ in got.values())
          and len(selected) <= S_MAX,
          f"{tag}: token/chunk/action limits exceeded by {got}")

    # Emitted outputs.
    check(executed["result_type"] == "SchedulerOutputs",
          f"{tag}: executor returned {executed['result_type']}")
    emitted = scheduled["outputs"]
    check(executed["result"] == emitted,
          f"{tag}: scheduler output differs from executor output")
    check(emitted["id"] == before["iteration_id"] + 1
          and emitted["id"] == snap["scheduler_iteration_id"],
          f"{tag}: decision id {emitted['id']}")
    want_emitted = [[ids[n], c] for n, c in exp["emitted"]]
    check(emitted["scheduled"] == want_emitted,
          f"{tag}: emitted {emitted['scheduled']}, expected {want_emitted}")
    check(not emitted["ignored_seq_ids"]
          and not emitted["preempted_seq_ids"],
          f"{tag}: ignored/preempted ids emitted")
    check([s for s, _ in worker["sampler_outputs"]]
          == [s for s, _ in want_emitted],
          f"{tag}: sampler output ids {worker['sampler_outputs']}")

    # After scheduling, before replay/completion.
    check(sched["num_running_batches"] == 1,
          f"{tag}: running batches after scheduling")
    check(sched["waiting"] == []
          and sorted(sched["running"]) ==
          sorted(ids[n] for n in exp["sched_running"])
          and len(sched["running"]) <= MAX_NUM_SEQS,
          f"{tag}: ownership after scheduling {sched}")
    for n, physical in exp["sched_physical"].items():
        check(sched["requests"][n]["physical_blocks"] == physical,
              f"{tag}: {n} physical blocks after scheduling "
              f"{sched['requests'][n]['physical_blocks']}")
    check(sched["free_blocks"] == n_free + exp["sched_free_delta"],
          f"{tag}: free blocks after scheduling {sched['free_blocks']}")
    for n in before["requests"]:
        for field in ("prompt_tokens_processed", "generated"):
            check(sched["requests"][n][field] == before["requests"][n][field],
                  f"{tag}: {n} {field} changed during scheduling")

    # After engine completion.
    check(after["iteration_id"] == before["iteration_id"] + 1,
          f"{tag}: iteration advanced "
          f"{after['iteration_id'] - before['iteration_id']}")
    check(after["num_running_batches"] == 0,
          f"{tag}: running batches after completion")
    running = sorted(ids[n] for n in exp["running"])
    check(after["waiting"] == [] and sorted(after["running"]) == running
          and after["engine_seq_ids"] == running
          and after["num_unfinished"] == len(running),
          f"{tag}: ownership after completion {after}")
    check(after["block_tables"] ==
          {str(ids[n]): exp["requests"][n]["physical"]
           for n in exp["running"]},
          f"{tag}: block tables after completion {after['block_tables']}")
    check(after["free_blocks"] == n_free + exp["free_delta"],
          f"{tag}: free blocks after completion {after['free_blocks']}")
    for n, r in exp["requests"].items():
        state = after["requests"][n]
        check((state["prompt_tokens_processed"],
               state["prompt_processing_finished"], state["generated"],
               state["status"], state["logical_blocks"],
               state["physical_blocks"]) ==
              (r["prompt"], r["prompt_finished"], r["generated"],
               r["status"], r["logical"], r["physical"]),
              f"{tag}: {n} state after completion {state}, expected {r}")
    check([(o.seq_id, len(o.token_ids), o.finished) for o in step_outputs]
          == [(ids[n], g, f) for n, g, f in exp["outputs"]],
          f"{tag}: request outputs "
          f"{[request_output_record(o) for o in step_outputs]}")
    for o in step_outputs:
        if o.finished:
            check(o.finish_reason == "length",
                  f"{tag}: finish reason {o.finish_reason}")
    return dict(emitted=emitted["scheduled"],
                plan={n: list(got[rid[ids[n]]]) for n in present},
                sampler_outputs=worker["sampler_outputs"],
                sched_free=sched["free_blocks"],
                after_free=after["free_blocks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # The directory may already exist (holding the console log), but earlier
    # run artifacts are never overwritten.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.json"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")

    trace = dict(add_requests=[], steps=[], idle=None)
    summary = dict(passed=False, failure=None,
                   script_sha256=sha256(__file__),
                   reused_helper=dict(path="scripts/check_lp_scheduler_gpu.py",
                                      sha256=sha256(single.__file__)))
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
        from sarathi.core.datatypes.sampling_params import (SamplingParams,
                                                            SamplingType)
        from sarathi.core.scheduler.lp_scheduler import LPScheduler
        from sarathi.engine.base_llm_engine import BaseLLMEngine

        numerical_policy = lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS)
        policy_json = to_jsonable(numerical_policy)
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
        # The generic profiler builds max_num_seqs prompts that share
        # max_num_batched_tokens: 64 // 2 = two 32-token prompts.
        check(scheduler_config.max_num_batched_tokens == B_MAX
              and scheduler_config.max_num_seqs == MAX_NUM_SEQS
              and B_MAX // MAX_NUM_SEQS == MAX_MODEL_LEN,
              "profiling inputs must give two max-length prompts")

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
        prompts = {n: select_prompt_ids(engine.tokenizer, vocab_size, start)
                   for n, start in PROMPT_ID_SEARCH_START.items()}
        prompt_ids = {n: ids for n, (ids, _) in prompts.items()}
        for n, ids in prompt_ids.items():
            check(len(ids) == PROMPT_LEN and all(
                0 <= i < vocab_size for i in ids), f"prompt {n} invalid")
        check(not set(prompt_ids["A"]) & set(prompt_ids["B"]),
              "prompt token lists must be distinct")
        sampling_params = SamplingParams(temperature=TEMPERATURE,
                                         max_tokens=MAX_TOKENS,
                                         ignore_eos=IGNORE_EOS, stop=None)
        check(sampling_params.sampling_type == SamplingType.GREEDY,
              f"sampling type {sampling_params.sampling_type}")

        n_free = bm.get_num_free_gpu_blocks()
        config_record = dict(
            model=MODEL, tokenizer=MODEL, load_format=LOAD_FORMAT,
            requested_dtype=DTYPE, resolved_dtype=str(model_config.dtype),
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
            prompt_token_ids=prompt_ids,
            prompt_tokens={n: toks for n, (_, toks) in prompts.items()},
            sampling=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                          ignore_eos=IGNORE_EOS, stop=[],
                          sampling_type=sampling_params.sampling_type.name,
                          same_params_for_both=True),
            add_b_after_step=ADD_B_AFTER_STEP,
            b_max=B_MAX, c_max=C_MAX, s_max=S_MAX, max_num_seqs=MAX_NUM_SEQS,
            memory_reserve=MEMORY_RESERVE,
            decode_memory_policy_id=DECODE_POLICY_ID,
            utilities=dict(decode=DECODE_UTILITY,
                           prefill_token=PREFILL_TOKEN_UTILITY,
                           preemption_penalty=PREEMPTION_PENALTY),
            numerical_policy=policy_json,
            profiling_max_num_batched_tokens=(
                scheduler_config.max_num_batched_tokens),
            profiling_max_num_seqs=scheduler_config.max_num_seqs,
            metrics_mode="enabled, all optional outputs off, never plotted",
            profiled_num_gpu_blocks=cache_config.num_gpu_blocks,
            block_manager_total_blocks=bm.num_total_gpu_blocks,
            watermark=bm.watermark, watermark_blocks=bm.watermark_blocks,
            free_blocks_before_requests=n_free,
            engine_init_seconds=round(init_seconds, 3),
        )
        summary["config"] = config_record
        print("config:", json.dumps(config_record), flush=True)

        # Pool adequacy: three blocks in use at the peak, the planning
        # reserve, and the native watermark must all fit.
        check(n_free == bm.num_total_gpu_blocks and not bm.block_tables,
              "block pool must be fully free before the requests")
        check(n_free - PEAK_BLOCKS_IN_USE >= bm.watermark_blocks
              + MEMORY_RESERVE,
              f"profiled pool {n_free} too small for watermark "
              f"{bm.watermark_blocks} + reserve {MEMORY_RESERVE} + "
              f"{PEAK_BLOCKS_IN_USE} blocks")

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
            outputs=outputs_record(r), state=capture_state(engine, seqs)))
        observer.wrap(engine, "_run_workers", worker_record)

        seqs = {}
        ids = {}

        def add_request(name):
            before = capture_state(engine, seqs)
            events = observer.events = []
            known = set(engine.seq_manager.seq_map)
            engine.add_request(prompt=None, sampling_params=sampling_params,
                               prompt_token_ids=list(prompt_ids[name]))
            new = set(engine.seq_manager.seq_map) - known
            check(len(new) == 1, f"add {name}: new sequences {new}")
            ids[name] = new.pop()
            seqs[name] = engine.seq_manager.seq_map[ids[name]]
            after = capture_state(engine, seqs)
            trace["add_requests"].append(dict(
                name=name, seq_id=ids[name], before=before,
                events=list(events), after=after))
            check([(e["call"], e["method"]) for e in events]
                  == [("_run_workers", "add_seq")],
                  f"add {name}: unexpected calls {events}")
            check(after["waiting"][-1] == ids[name]
                  and after["iteration_id"] == before["iteration_id"]
                  and after["free_blocks"] == before["free_blocks"]
                  and after["requests"][name]["physical_blocks"] == 0
                  and after["requests"][name]["status"] == "WAITING",
                  f"add {name}: state after admission {after}")
            print(f"added {name}: seq_id {ids[name]}, waiting "
                  f"{after['waiting']}, running {after['running']}",
                  flush=True)

        add_request("A")
        finished_outputs = []

        for exp in EXPECTED_STEPS:
            if exp["step"] == ADD_B_AFTER_STEP + 1:
                add_request("B")
                check(ids["A"] < ids["B"], f"request ids {ids}")
            before = capture_state(engine, seqs)
            record = dict(step=exp["step"], name=exp["name"], before=before)
            trace["steps"].append(record)
            events = record["events"] = observer.events = []
            step_outputs = engine.step()
            after = capture_state(engine, seqs)
            record.update(after=after, request_outputs=[
                request_output_record(o) for o in step_outputs])
            sched = events[3]["state"] if len(events) > 3 else None
            result = check_step(exp, ids, n_free, before, events, sched,
                                after, step_outputs, policy_json)
            record["checked"] = result
            finished_outputs.extend(o for o in step_outputs if o.finished)
            print(f"step {exp['step']} {exp['name']}: plan {result['plan']}, "
                  f"emitted {result['emitted']}, sampler "
                  f"{result['sampler_outputs']}, free after scheduling "
                  f"{result['sched_free']} (N{result['sched_free'] - n_free:+d})"
                  f", after completion {result['after_free']} "
                  f"(N{result['after_free'] - n_free:+d}), outputs "
                  f"{[(o.seq_id, list(o.token_ids), o.finished) for o in step_outputs]}",
                  flush=True)

        final = capture_state(engine, seqs)
        check(len(finished_outputs) == 2
              and sorted(o.seq_id for o in finished_outputs)
              == sorted(ids.values())
              and all(len(o.token_ids) == MAX_TOKENS
                      and o.finish_reason == "length"
                      for o in finished_outputs),
              "expected one finished two-token output per request")
        check(not engine.has_unfinished_requests(),
              "unfinished requests remain")
        check(final["waiting"] == [] and final["running"] == []
              and final["block_tables"] == {}
              and final["engine_seq_ids"] == []
              and final["free_blocks"] == n_free,
              f"drained state {final}")

        # One ordinary idle call after draining.
        before = capture_state(engine, seqs)
        events = observer.events = []
        idle_outputs = engine.step()
        after = capture_state(engine, seqs)
        trace["idle"] = dict(before=before, events=list(events), after=after,
                             request_outputs=[request_output_record(o)
                                              for o in idle_outputs])
        check(idle_outputs == [], f"idle step returned {idle_outputs}")
        check([e["call"] for e in events] == ["schedule"],
              f"idle step calls {[e['call'] for e in events]}")
        check(events[0]["outputs"]["scheduled"] == [],
              "idle step scheduled work")
        check(after["iteration_id"] == before["iteration_id"] + 1,
              "idle step must advance the iteration once")
        check({**after, "iteration_id": None} ==
              {**before, "iteration_id": None},
              f"idle step changed state: {before} -> {after}")
        print(f"idle: no outputs, calls {[e['call'] for e in events]}, "
              f"iteration {before['iteration_id']} -> "
              f"{after['iteration_id']}", flush=True)

        summary.update(
            passed=True,
            n_free=n_free,
            seq_ids=ids,
            finished_outputs=[request_output_record(o)
                              for o in finished_outputs],
            decisions=[dict(step=s["step"], name=s["name"],
                            plan=s["checked"]["plan"],
                            emitted=s["checked"]["emitted"],
                            sampler_outputs=s["checked"]["sampler_outputs"])
                       for s in trace["steps"]],
        )
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        # The engine is never reused after a failure.
        engine = None
        trace_path.write_text(json.dumps(trace, indent=1))
        summary_path.write_text(json.dumps(summary, indent=1))
        # Stop only the Ray runtime this process started (no-op otherwise).
        if "ray" in sys.modules:
            sys.modules["ray"].shutdown()
    print("RESULT:", "PASS" if summary["passed"] else "FAIL", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
