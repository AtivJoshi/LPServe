"""Bounded three-request contention GPU check for the live LP scheduler.

Adds requests A, B, and C (16 prompt tokens, four greedy output tokens each)
to a real ``BaseLLMEngine`` before the first step, with at most two scheduled
actions per step (``s_max=2``) and room for three residents. It then steps the
engine until all three finish, using the registered ``LPScheduler``, one GPU
worker, dummy TinyLlama weights, and native completion.

The schedule is not prescribed: which tied request the solver selects is
recorded, not asserted. Every nonempty step is checked against invariants
derived from the observed pre-step state, and the run must witness:

1. an eligible waiting request omitted from a decision, left waiting and
   unallocated;
2. a decision boundary with all three requests prompt-complete, allocated,
   and unfinished;
3. a resident omitted at such a boundary, unchanged through that step;
4. a later selection of that omitted resident; and
5. completion of all three requests.

It does not establish general fairness, arbitrary-workload correctness,
sampler correctness, numerical agreement with a reference, central/worker
block-table equality, preemption, pipeline execution, or performance.

Run from the repository root with the root on PYTHONPATH (the Ray worker must
unpickle the LP numerical policy from the root-level module):

    PYTHONPATH="$PWD" timeout 300s python -B \\
        scripts/check_lp_scheduler_contention_gpu.py \\
        --output-dir validation_output/lp_scheduler_contention/<new timestamp>

Exit status is zero only if every assertion passes.
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
# Distinct search starts give disjoint prompt-token lists.
PROMPT_ID_SEARCH_START = dict(A=1000, B=2000, C=3000)
TEMPERATURE = 0.0
MAX_TOKENS = 4
IGNORE_EOS = True
# b_max=96 with max_num_seqs=3 makes the generic profiler build three
# 32-token prompts; actual prefill chunks stay bounded by c_max.
B_MAX = 96
C_MAX = 8
S_MAX = 2
MAX_NUM_SEQS = 3
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
# 16 prompt + 4 generated tokens span at most two 16-token blocks per request.
PEAK_BLOCKS_IN_USE = 6
# Total prompt-token work plus requested decode-token work: a termination
# guard, not an expected schedule length.
MAX_NONEMPTY_STEPS = 3 * (PROMPT_LEN + MAX_TOKENS)
FINISHED = "FINISHED_LENGTH_CAPPED"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=30).stdout
    except Exception as err:  # provenance only; never affects the check
        return f"unavailable: {err!r}"


def provenance_record():
    """Tested revision, working-tree state, and file hashes."""
    files = ["scripts/check_lp_scheduler_contention_gpu.py",
             "scripts/check_lp_scheduler_gpu.py",
             "tests/test_lp_scheduler.py",
             "sarathi/core/scheduler/lp_scheduler.py",
             "lp_relaxation_scheduler.py", "lpserve_state_mapping.py",
             "lpserve_plan_execution.py"]
    diff = git("diff", "HEAD")
    return dict(
        git_head=git("rev-parse", "HEAD").strip(),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        # tests/ matches an ignore rule; tracked files there still appear.
        git_status_short=git("status", "--short", "--untracked-files=all"),
        git_status_ignored_in_scope=git(
            "status", "--short", "--ignored", "--", "tests/test_lp_scheduler.py",
            "scripts/check_lp_scheduler_contention_gpu.py"),
        git_diff_head_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        git_diff_head_stat=git("diff", "HEAD", "--stat"),
        file_sha256={f: sha256(REPO_ROOT / f) for f in files},
        executed_script_sha256=sha256(__file__),
        imported_helper_sha256=sha256(single.__file__),
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def model_asset_record(model_config):
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        repo = Path(HF_HUB_CACHE) / ("models--" + MODEL.replace("/", "--"))
        refs_main = (repo / "refs" / "main").read_text().strip()
    except Exception as err:  # provenance only
        repo, refs_main = None, f"unavailable: {err!r}"
    return dict(cache_repo=str(repo), refs_main=refs_main,
                hf_config_commit_hash=getattr(model_config.hf_config,
                                              "_commit_hash", None))


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
    """Selected primitive state for every added request; no live objects.
    Central block IDs are copied for observation only."""
    scheduler = engine.scheduler
    bm = scheduler.block_manager
    requests = {}
    for name, seq in seqs.items():
        table = bm.block_tables.get(seq.seq_id)
        requests[name] = dict(
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
        requests=requests,
    )


def outputs_with_counts(outputs):
    return dict(outputs_record(outputs),
                num_batched_prompt_tokens=outputs.num_batched_prompt_tokens,
                num_batched_output_tokens=outputs.num_batched_output_tokens,
                num_batched_tokens=outputs.num_batched_tokens)


def worker_record(args, kwargs, result):
    record = dict(method=args[0])
    if args[0] == "execute_model":
        record["sampler_outputs"] = [[o.seq_id, o.output_token]
                                     for o in result]
    return record


def remaining_work(state):
    return sum(PROMPT_LEN - r["prompt_tokens_processed"]
               + MAX_TOKENS - r["generated"]
               for r in state["requests"].values())


def check_step(tag, ids, before, events, after, step_outputs, policy_json):
    """Check one nonempty step from observed state; return its record."""
    names = {seq_id: n for n, seq_id in ids.items()}
    unfinished = sorted(ids[n] for n, r in before["requests"].items()
                        if not r["is_finished"])

    # Quiescent decision boundary with exact ownership.
    check(before["num_pipeline_stages"] == 1
          and before["num_running_batches"] == 0,
          f"{tag}: not a quiescent single-stage boundary")
    check(sorted(before["waiting"] + before["running"]) == unfinished
          and before["engine_seq_ids"] == unfinished
          and before["num_unfinished"] == len(unfinished),
          f"{tag}: ownership {before}")
    for seq_id in before["waiting"]:
        r = before["requests"][names[seq_id]]
        check(r["status"] == "WAITING" and r["block_ids"] is None,
              f"{tag}: waiting request {seq_id} state {r}")
    for seq_id in before["running"]:
        r = before["requests"][names[seq_id]]
        check(r["status"] == "PAUSED" and r["block_ids"],
              f"{tag}: resident request {seq_id} state {r}")

    calls = [e["call"] for e in events]
    check(calls == ["map_scheduler_state", "solve_and_extract",
                    "execute_plan", "schedule", "_run_workers"],
          f"{tag}: unexpected call sequence {calls}")
    mapped, solved, executed, scheduled, worker = events
    check(worker["method"] == "execute_model",
          f"{tag}: worker call {worker['method']}")

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
          and snap["numerical_policy"] == policy_json
          and snap["num_pipeline_stages"] == 1
          and snap["num_running_batches"] == 0,
          f"{tag}: mapped policy inputs")
    check(snap["free_physical_blocks"] == before["free_blocks"]
          and problem["m_free"] == before["free_blocks"],
          f"{tag}: mapped free blocks")
    check(sorted(r["raw_seq_id"] for r in snap["requests"]) == unfinished,
          f"{tag}: mapped request set {snap['requests']}")
    check([r["request_id"] for r in problem["requests"]]
          == [r["request_id"] for r in snap["requests"]],
          f"{tag}: problem request order")
    for r in snap["requests"]:
        check(r["utility"] == dict(decode_utility=DECODE_UTILITY,
                                   prefill_token_utility=PREFILL_TOKEN_UTILITY,
                                   preemption_penalty=PREEMPTION_PENALTY),
              f"{tag}: mapped utilities")
    raw = {r["request_id"]: r["raw_seq_id"] for r in snap["requests"]}

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
    check(not plan["dominant_preemption_ids"]
          and not plan["safety_preemption_ids"],
          f"{tag}: plan preemption ids")
    selected = []
    for d in plan["decisions"]:
        check(d["preempt"] == 0 and not (d["prefill_tokens"] > 0
                                         and d["decode"]),
              f"{tag}: plan decision {d}")
        if d["prefill_tokens"] > 0 or d["decode"]:
            selected.append(d)
    selected.sort(key=lambda d: (0 if d["prefill_tokens"] > 0 else 1,
                                 tuple(d["order_key"])))
    want_emitted = [[raw[d["request_id"]], d["prefill_tokens"]]
                    for d in selected]

    # Emitted outputs.
    check(executed["result_type"] == "SchedulerOutputs",
          f"{tag}: executor returned {executed['result_type']}")
    emitted = scheduled["outputs"]
    check(executed["result"] == outputs_record_only(emitted),
          f"{tag}: scheduler output differs from executor output")
    check(emitted["id"] == before["iteration_id"] + 1
          and emitted["id"] == snap["scheduler_iteration_id"],
          f"{tag}: decision id {emitted['id']}")
    check(emitted["scheduled"] == want_emitted,
          f"{tag}: emitted {emitted['scheduled']}, plan gives {want_emitted}")
    check(not emitted["ignored_seq_ids"]
          and not emitted["preempted_seq_ids"],
          f"{tag}: ignored/preempted ids emitted")
    sel_ids = [s for s, _ in emitted["scheduled"]]
    check(len(sel_ids) == len(set(sel_ids)) and 1 <= len(sel_ids) <= S_MAX,
          f"{tag}: selected ids {sel_ids}")
    chunks = dict(emitted["scheduled"])
    prefill_ids = {s for s, c in chunks.items() if c > 0}
    decode_ids = {s for s, c in chunks.items() if c == 0}
    for seq_id, chunk in chunks.items():
        r = before["requests"][names[seq_id]]
        remainder = PROMPT_LEN - r["prompt_tokens_processed"]
        if chunk > 0:
            check(chunk <= min(remainder, C_MAX),
                  f"{tag}: chunk {chunk} for {seq_id}, remainder {remainder}")
        else:
            check(remainder == 0 and r["prompt_processing_finished"]
                  and seq_id in before["running"],
                  f"{tag}: decode for ineligible request {seq_id}: {r}")
    tokens = sum(chunks.values()) + len(decode_ids)
    check(emitted["num_batched_prompt_tokens"] == sum(chunks.values())
          and emitted["num_batched_output_tokens"] == len(decode_ids)
          and emitted["num_batched_tokens"] == tokens and tokens <= B_MAX,
          f"{tag}: output counts {emitted}")
    check([s for s, _ in worker["sampler_outputs"]] == sel_ids,
          f"{tag}: sampler output ids {worker['sampler_outputs']}")

    # After scheduling, before replay/completion.
    sched = scheduled["state"]
    check(sched["num_running_batches"] == 1,
          f"{tag}: running batches after scheduling")
    check(len(sched["running"]) <= MAX_NUM_SEQS,
          f"{tag}: residents {sched['running']}")
    expected_free = before["free_blocks"]
    for seq_id in unfinished:
        n = names[seq_id]
        b, s = before["requests"][n], sched["requests"][n]
        check((s["prompt_tokens_processed"], s["generated"])
              == (b["prompt_tokens_processed"], b["generated"]),
              f"{tag}: {n} progress changed during scheduling")
        if seq_id in prefill_ids and seq_id in before["waiting"]:
            check(seq_id in sched["running"]
                  and s["physical_blocks"] == s["logical_blocks"],
                  f"{tag}: {n} admission state {s}")
            expected_free -= s["physical_blocks"]
        elif seq_id in decode_ids:
            gap = b["logical_blocks"] - b["physical_blocks"]
            check(gap in (0, 1)
                  and s["block_ids"][:b["physical_blocks"]] == b["block_ids"]
                  and s["physical_blocks"] == b["physical_blocks"] + gap,
                  f"{tag}: {n} decode append {b} -> {s}")
            expected_free -= gap
        else:
            check(s["block_ids"] == b["block_ids"],
                  f"{tag}: {n} blocks changed without allocation")
    check(sched["free_blocks"] == expected_free,
          f"{tag}: free blocks after scheduling {sched['free_blocks']}, "
          f"expected {expected_free}")

    # After engine completion: only selected requests advance.
    check(after["iteration_id"] == before["iteration_id"] + 1
          and after["num_running_batches"] == 0,
          f"{tag}: iteration/batch state after completion")
    freed = 0
    finished_now = []
    for seq_id in unfinished:
        n = names[seq_id]
        b, a = before["requests"][n], after["requests"][n]
        if seq_id in prefill_ids:
            done = b["prompt_tokens_processed"] + chunks[seq_id]
            check(a["prompt_tokens_processed"] == done
                  and a["prompt_processing_finished"] == (done == PROMPT_LEN)
                  and a["output_token_ids"] == b["output_token_ids"],
                  f"{tag}: {n} prefill completion {b} -> {a}")
        elif seq_id in decode_ids:
            check(a["prompt_tokens_processed"] == b["prompt_tokens_processed"]
                  and a["generated"] == b["generated"] + 1
                  and a["output_token_ids"][:-1] == b["output_token_ids"],
                  f"{tag}: {n} decode completion {b} -> {a}")
        else:
            same = ("status", "prompt_tokens_processed",
                    "prompt_processing_finished", "output_token_ids",
                    "generated", "logical_blocks", "block_ids")
            check(all(a[k] == b[k] for k in same)
                  and (seq_id in after["waiting"])
                  == (seq_id in before["waiting"])
                  and (seq_id in after["running"])
                  == (seq_id in before["running"]),
                  f"{tag}: unselected {n} changed {b} -> {a}")
        if a["is_finished"]:
            check(seq_id in decode_ids and a["status"] == FINISHED
                  and a["generated"] == MAX_TOKENS
                  and a["block_ids"] is None
                  and seq_id not in after["waiting"] + after["running"]
                  and seq_id not in after["engine_seq_ids"],
                  f"{tag}: {n} finish state {a}")
            freed += sched["requests"][n]["physical_blocks"]
            finished_now.append(seq_id)
        else:
            check(a["status"] == ("WAITING" if seq_id in after["waiting"]
                                  else "PAUSED")
                  and (seq_id in after["running"]) == bool(a["block_ids"]),
                  f"{tag}: {n} ownership after completion {a}")
    check(after["free_blocks"] == sched["free_blocks"] + freed,
          f"{tag}: free blocks after completion {after['free_blocks']}")
    reduced = remaining_work(before) - remaining_work(after)
    check(reduced == tokens and reduced >= 1,
          f"{tag}: remaining work reduced by {reduced}, expected {tokens}")
    returned = [(o.seq_id, len(o.token_ids), o.finished)
                for o in step_outputs]
    check(returned == [(s, after["requests"][names[s]]["generated"],
                        s in finished_now) for s in sel_ids],
          f"{tag}: request outputs "
          f"{[request_output_record(o) for o in step_outputs]}")
    for o in step_outputs:
        if o.finished:
            check(o.finish_reason == "length",
                  f"{tag}: finish reason {o.finish_reason}")

    omitted = [s for s in unfinished if s not in sel_ids]
    return dict(
        selected=[names[s] for s in sel_ids],
        omitted=[names[s] for s in omitted],
        omitted_waiting=[names[s] for s in omitted if s in before["waiting"]],
        omitted_resident=[names[s] for s in omitted
                          if s in before["running"]],
        all_prompt_complete_residents=(
            len(before["running"]) == 3 and all(
                before["requests"][names[s]]["prompt_processing_finished"]
                for s in before["running"])),
        emitted=emitted["scheduled"],
        plan={names[raw[d["request_id"]]]: [d["prefill_tokens"], d["decode"],
                                            d["preempt"]]
              for d in plan["decisions"]},
        objective=plan["objective"],
        sampler_outputs=worker["sampler_outputs"],
        free=dict(before=before["free_blocks"], sched=sched["free_blocks"],
                  after=after["free_blocks"]),
        work=dict(before=remaining_work(before), after=remaining_work(after)),
        finished=[names[s] for s in finished_now],
    )


def outputs_record_only(record):
    return {k: record[k] for k in ("id", "scheduled", "ignored_seq_ids",
                                   "preempted_seq_ids")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # The directory may already exist (holding the console log and CPU
    # check output), but earlier run artifacts are never overwritten.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.json"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")

    trace = dict(add_requests=[], steps=[], idle=None)
    summary = dict(passed=False, failure=None, witnesses=None,
                   command=sys.argv, provenance=provenance_record())
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
        # max_num_batched_tokens: 96 // 3 = three 32-token prompts
        # (source-confirmed; the worker-side batch is not observed here).
        check(scheduler_config.max_num_batched_tokens == B_MAX
              and scheduler_config.max_num_seqs == MAX_NUM_SEQS
              and B_MAX // MAX_NUM_SEQS == MAX_MODEL_LEN
              and B_MAX % MAX_NUM_SEQS == 0,
              "profiling inputs must give three max-length prompts")

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
        special = set(engine.tokenizer.all_special_ids)
        for n, ids in prompt_ids.items():
            check(len(ids) == PROMPT_LEN and all(
                0 <= i < vocab_size and i not in special for i in ids),
                f"prompt {n} invalid")
        all_ids = [i for ids in prompt_ids.values() for i in ids]
        check(len(all_ids) == len(set(all_ids)),
              "prompt token lists must be distinct and disjoint")
        sampling_params = SamplingParams(temperature=TEMPERATURE,
                                         max_tokens=MAX_TOKENS,
                                         ignore_eos=IGNORE_EOS, stop=None)
        check(sampling_params.sampling_type == SamplingType.GREEDY,
              f"sampling type {sampling_params.sampling_type}")

        n_free = bm.get_num_free_gpu_blocks()
        config_record = dict(
            model=MODEL, tokenizer=MODEL, load_format=LOAD_FORMAT,
            requested_dtype=DTYPE, resolved_dtype=str(model_config.dtype),
            model_assets=model_asset_record(model_config),
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
                          same_params_for_all=True),
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
            peak_blocks_in_use_bound=PEAK_BLOCKS_IN_USE,
            max_nonempty_steps=MAX_NONEMPTY_STEPS,
            engine_init_seconds=round(init_seconds, 3),
        )
        summary["config"] = config_record
        print("config:", json.dumps(config_record), flush=True)

        # Pool adequacy: all requests resident at their two-block peak, the
        # planning reserve, conservative decode charges (at most S_MAX), and
        # the native watermark must all fit.
        check(n_free == bm.num_total_gpu_blocks and not bm.block_tables,
              "block pool must be fully free before the requests")
        check(n_free - PEAK_BLOCKS_IN_USE - S_MAX >= bm.watermark_blocks
              + MEMORY_RESERVE,
              f"profiled pool {n_free} too small for {PEAK_BLOCKS_IN_USE} "
              f"blocks + {S_MAX} decode charges + reserve {MEMORY_RESERVE} "
              f"+ watermark {bm.watermark_blocks}")

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
            outputs=outputs_with_counts(r), state=capture_state(engine, seqs)))
        observer.wrap(engine, "_run_workers", worker_record)

        seqs = {}
        ids = {}
        for name in PROMPT_ID_SEARCH_START:
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
            check(after["waiting"] == before["waiting"] + [ids[name]]
                  and after["running"] == []
                  and after["iteration_id"] == before["iteration_id"]
                  and after["free_blocks"] == n_free
                  and after["requests"][name]["block_ids"] is None
                  and after["requests"][name]["status"] == "WAITING",
                  f"add {name}: state after admission {after}")
            print(f"added {name}: seq_id {ids[name]}, waiting "
                  f"{after['waiting']}", flush=True)
        summary["seq_ids"] = ids

        finished_outputs = []
        witness = dict(waiting_omitted=[], all_resident_boundaries=[],
                       later_selected=[])
        step = 0
        while engine.has_unfinished_requests():
            check(step < MAX_NONEMPTY_STEPS,
                  f"termination guard of {MAX_NONEMPTY_STEPS} steps reached")
            step += 1
            tag = f"step {step}"
            before = capture_state(engine, seqs)
            record = dict(step=step, before=before)
            trace["steps"].append(record)
            events = record["events"] = observer.events = []
            step_outputs = engine.step()
            after = capture_state(engine, seqs)
            record.update(after=after, request_outputs=[
                request_output_record(o) for o in step_outputs])
            result = check_step(tag, ids, before, events, after, step_outputs,
                                policy_json)
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
                      f"{tag}: no resident omitted at a three-resident "
                      "boundary")
                witness["all_resident_boundaries"].append(
                    dict(step=step, selected=result["selected"],
                         omitted=result["omitted_resident"]))
            print(f"{tag}: plan {result['plan']}, emitted "
                  f"{result['emitted']}, omitted {result['omitted']}, "
                  f"sampler {result['sampler_outputs']}, free "
                  f"{result['free']}, work {result['work']}, finished "
                  f"{result['finished']}", flush=True)

        summary["witnesses"] = witness
        check(witness["waiting_omitted"],
              "no eligible waiting request was omitted")
        check(witness["all_resident_boundaries"],
              "no boundary had three prompt-complete allocated residents")
        check(witness["later_selected"],
              "no resident omitted at such a boundary was later selected")

        final = capture_state(engine, seqs)
        check(len(finished_outputs) == 3
              and sorted(o.seq_id for o in finished_outputs)
              == sorted(ids.values())
              and all(len(o.token_ids) == MAX_TOKENS
                      and o.finish_reason == "length"
                      for o in finished_outputs),
              "expected one finished four-token output per request")
        check(all(r["status"] == FINISHED and r["generated"] == MAX_TOKENS
                  for r in final["requests"].values()),
              f"final request states {final['requests']}")
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
            nonempty_steps=step,
            finished_outputs=[request_output_record(o)
                              for o in finished_outputs],
            decisions=[dict(step=s["step"], **{k: s["checked"][k] for k in (
                "selected", "omitted", "plan", "emitted", "sampler_outputs",
                "finished")}) for s in trace["steps"]],
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
