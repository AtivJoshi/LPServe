"""Bounded dummy-weight GPU comparison of the adapted vLLM reference and two
LP coefficient policies on one two-request workload.

One process runs one (policy, resident limit) pair through a real
``BaseLLMEngine`` built from native ``EngineArgs``, one GPU worker (tensor
and pipeline parallelism 1), dummy TinyLlama weights, greedy sampling with
EOS ignored, and native replay and completion. Policies:

- ``adapted_vllm``: the registered ``AdaptedVLLMScheduler``, budget 8, with
  no LP chunk or action cap.
- ``lp_uniform``: the registered ``LPScheduler`` with the configured uniform
  coefficients 1/1/1.
- ``lp_rank_based``: the registered ``LPScheduler`` with
  ``lpserve_state_mapping.rank_based_utility`` (D = P = G = 1) selected
  through the existing ``_coefficient_policy`` hook; the configured triple
  is then unused.

Both LP policies use ``b_max`` 8, ``c_max`` 8, ``s_max`` 1, and numerical
policy ``lp_relaxation_mvp_v1``. The action cap is a deliberate difference
from the reference.

Workload: A (5 prompt tokens) then B (8), two output tokens each, both
submitted before the first decision; block size 16, 8 initialized blocks,
``max_model_len`` 32, resident limit 2 or 1.

The decision sequence is not prescribed. Each nonempty decision is checked
against its policy's own contract (mapped or proposed inputs, plan or
proposal validity, emitted order, central and worker block operations,
replay, sampler association, progress, completion and release) and recorded
with the before/after state. ``MAX_DECISIONS`` bounds the observation: work
still unfinished at the bound is reported as ``bounded_unfinished``, not
forced to finish. A scheduler, mapping, solver, execution, worker, or check
error is a ``failure``. Generated-token events count emitted decode entries;
the native output counter resets on recomputation and is reported
separately.

Not established: physical-ID equality between managers, cache-content or
attention correctness, sampler generality, generation quality, policy
equivalence or recovery, starvation, fairness, or performance.

Run from the repository root with the root on PYTHONPATH, one pair per
process, on an allocated GPU node:

    PYTHONPATH="$PWD" timeout 600s python -B scripts/compare_scheduler_gpu.py \\
        --policy lp_uniform --resident-limit 2 \\
        --model-path <local TinyLlama snapshot directory> \\
        --output-dir validation_output/scheduler_comparison_gpu/<UTC>/<run>

Exit status is zero only if every check passes; the workload outcome is
reported separately.
"""

import argparse
import functools
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Read-only helpers from the earlier bounded checks; their main() is unused.
import check_adapted_vllm_gpu as adapted
import check_lp_scheduler_gpu as single
import check_lp_scheduler_preemption_gpu as preemption
from check_adapted_vllm_gpu import (EngineRuntime, build_engine_class,
                                    central_state, check_quiescent,
                                    instrument_block_manager)
from check_lp_scheduler_gpu import (CheckFailure, check, environment_record,
                                    outputs_record, request_output_record,
                                    to_jsonable)
from check_lp_scheduler_preemption_gpu import (CacheCapacitySelection,
                                               check_declared_order, git,
                                               select_prompt_ids, sha256)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Approved scoped comparison inputs (provisional, not project-wide policy).
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "dummy"
DTYPE = "float16"
ATTENTION_BACKEND = "flash_attention"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 8
MAX_MODEL_LEN = 32
GPU_MEMORY_UTILIZATION = 0.5
SEED = 42
TRUST_REMOTE_CODE = True
TOKENIZER_MODE = "auto"
TEMPERATURE = 0.0
IGNORE_EOS = True
MAX_TOKENS = 2
# Explicit disjoint prompt tokens, submitted A then B (seq 0 then seq 1).
# They equal the accepted mixed-case selection from search starts 1000/2000.
PROMPTS = dict(A=list(range(1000, 1005)), B=list(range(2000, 2008)))
NAMES = {0: "A", 1: "B"}
TOKEN_BUDGET = 8
RESIDENT_LIMITS = (2, 1)
LP_B_MAX = 8
LP_C_MAX = 8
LP_S_MAX = 1
UNIFORM = dict(decode_utility=1.0, prefill_token_utility=1.0,
               preemption_penalty=1.0)
RANK_SCALES = dict(decode_scale=1.0, prefill_scale=1.0, penalty_scale=1.0)
NUMERICAL_POLICY_FIELDS = dict(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)
POLICIES = ("adapted_vllm", "lp_uniform", "lp_rank_based")
MAX_DECISIONS = 8  # observation bound per run, not a scheduler restart cap
PROCESS_TIMEOUT_SECONDS = 600
PROFILE_METHOD = preemption.PROFILE_METHOD
INIT_CACHE_METHOD = preemption.INIT_CACHE_METHOD

DRAINED, BOUNDED_UNFINISHED, FAILURE = "drained", "bounded_unfinished", "failure"
ADMISSION, PREFILL, DECODE = "admission", "resident_prefill", "decode"
LP_CALLS = ["map_scheduler_state", "solve_and_extract", "execute_plan"]


def run_parameters(policy, resident_limit):
    """The provisional inputs of one run, as recorded and checked."""
    check(policy in POLICIES and resident_limit in RESIDENT_LIMITS,
          f"unknown run {policy} {resident_limit}")
    params = dict(
        policy=policy, resident_limit=resident_limit, block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_GPU_BLOCKS, max_model_len=MAX_MODEL_LEN,
        native_watermark_blocks=int(0.01 * NUM_GPU_BLOCKS),
        prompts=PROMPTS, max_tokens=MAX_TOKENS, token_budget=TOKEN_BUDGET,
        max_decisions=MAX_DECISIONS)
    if policy == "adapted_vllm":
        params.update(scheduler_type="adapted_vllm",
                      max_num_batched_tokens=TOKEN_BUDGET,
                      chunk_cap=None, action_cap=None)
    else:
        params.update(scheduler_type="lp", b_max=LP_B_MAX, c_max=LP_C_MAX,
                      s_max=LP_S_MAX, numerical_policy=NUMERICAL_POLICY_FIELDS,
                      configured_triple=UNIFORM,
                      coefficient_policy=(
                          "configured uniform triple" if policy == "lp_uniform"
                          else dict(rank_based_utility=RANK_SCALES,
                                    note="configured triple unused")))
    return params


def engine_args(policy, resident_limit, model_path, metrics_dir):
    """Native ``EngineArgs`` for one run."""
    from sarathi.engine.arg_utils import EngineArgs

    common = dict(
        model=str(model_path), tokenizer=str(model_path),
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN, max_num_seqs=resident_limit,
        # Enabled metrics with optional outputs off, as in the earlier
        # bounded checks; nothing is plotted.
        write_metrics=True, output_dir=str(metrics_dir),
        enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
        enable_chrome_trace=False, enable_request_outputs=False,
        keep_individual_batch_metrics=True,
        attention_backend=ATTENTION_BACKEND)
    if policy == "adapted_vllm":
        return EngineArgs(scheduler_type="adapted_vllm",
                          max_num_batched_tokens=TOKEN_BUDGET, **common)
    from lp_relaxation_scheduler import NumericalPolicy
    return EngineArgs(
        scheduler_type="lp", lp_b_max=LP_B_MAX, lp_c_max=LP_C_MAX,
        lp_s_max=LP_S_MAX, lp_decode_utility=UNIFORM["decode_utility"],
        lp_prefill_token_utility=UNIFORM["prefill_token_utility"],
        lp_preemption_penalty=UNIFORM["preemption_penalty"],
        lp_numerical_policy=NumericalPolicy(**NUMERICAL_POLICY_FIELDS),
        **common)


def select_coefficient_policy(scheduler, policy):
    """Rank-based runs override the live hook on this instance only."""
    if policy == "lp_rank_based":
        import lpserve_state_mapping as lsm
        coefficients = functools.partial(lsm.rank_based_utility, **RANK_SCALES)
        scheduler._coefficient_policy = lambda: coefficients


def register_worker_helpers_by_value():
    """The validation worker class reuses observation functions from
    ``check_adapted_vllm_gpu``; pickle them by value so the Ray worker does
    not need to import that script module."""
    from ray import cloudpickle
    cloudpickle.register_pickle_by_value(adapted)


# ---------------------------------------------------------------------------
# Observation (driver process only)
# ---------------------------------------------------------------------------


class DecisionObserver:
    """Validation-only wrappers that record each native result once and
    return it unchanged: block operations, the reference proposal or the LP
    mapping/solve/execution results, and the emitted output with the
    central state at the end of scheduling. ``close`` restores the module
    functions it wrapped."""

    def __init__(self, scheduler, seq_map, policy):
        self.ops, self.proposals, self.lp_events, self.schedules = [], [], [], []
        self._restore = []
        instrument_block_manager(scheduler.block_manager, self.ops)
        if policy == "adapted_vllm":
            self._wrap(scheduler, "_propose", lambda r: self.proposals.append(
                to_jsonable(r)))
        else:
            import lp_relaxation_scheduler as lrs
            import lpserve_plan_execution as lpe
            import lpserve_state_mapping as lsm
            for module, name in ((lsm, LP_CALLS[0]), (lrs, LP_CALLS[1]),
                                 (lpe, LP_CALLS[2])):
                self._wrap(module, name, functools.partial(self._lp, name))
        self._wrap(scheduler, "schedule", lambda r: self.schedules.append(dict(
            outputs=outputs_record(r), num_ops=len(self.ops),
            state=central_state(scheduler, seq_map))))

    def _lp(self, name, result):
        record = (outputs_record(result)
                  if type(result).__name__ == "SchedulerOutputs"
                  else to_jsonable(result))
        self.lp_events.append(dict(call=name, result_type=type(result).__name__,
                                   result=record))

    def _wrap(self, owner, name, record):
        native = getattr(owner, name)

        def wrapper(*args, **kwargs):
            result = native(*args, **kwargs)
            record(result)
            return result
        setattr(owner, name, wrapper)
        self._restore.append((owner, name, native))

    def take(self):
        record = dict(proposals=list(self.proposals),
                      lp_events=list(self.lp_events),
                      schedules=list(self.schedules), central_ops=list(self.ops))
        del self.ops[:], self.proposals[:], self.lp_events[:], self.schedules[:]
        return record

    def close(self):
        for owner, name, native in reversed(self._restore):
            setattr(owner, name, native)
        del self._restore[:]


# ---------------------------------------------------------------------------
# Policy-specific checks. Each returns (victims, layout): victim seq IDs in
# preemption order and [seq_id, kind, chunk, block_demand] in emitted order.
# ---------------------------------------------------------------------------


def _remaining(seq):
    return seq["prompt_len"] - seq["prompt_tokens_processed"]


def _gap(before, sid):
    return (before["seqs"][str(sid)]["logical_blocks"]
            - len(before["block_tables"][str(sid)]))


def check_reference_decision(tag, before, record, resident_limit):
    check(len(record["proposals"]) == 1 and record["lp_events"] == [],
          f"{tag}: {len(record['proposals'])} reference proposals")
    p = record["proposals"][0]
    check(p["running_order"] == before["running"]
          and p["arrived_waiting_order"] == before["waiting"]
          and p["free_blocks"] == before["free_blocks"]
          and p["watermark_blocks"] == before["watermark_blocks"]
          and p["token_budget"] == TOKEN_BUDGET,
          f"{tag}: proposal inputs {p}")
    actions, victims = p["actions"], list(p["victims"])
    selected = [a["seq_id"] for a in actions]
    check(len(set(selected)) == len(selected) and set(victims) <= set(before["running"])
          and not set(victims) & set(selected) and (actions or not victims),
          f"{tag}: victims {victims} / actions {selected}")
    for a in actions:
        sid, seq = a["seq_id"], before["seqs"][str(a["seq_id"])]
        if a["kind"] == ADMISSION:
            ok = (sid in before["waiting"]
                  and 0 < a["num_tokens"] <= _remaining(seq)
                  and a["block_demand"] == seq["logical_blocks"])
        elif a["kind"] == PREFILL:
            ok = (sid in before["running"] and not seq["prompt_processing_finished"]
                  and 0 < a["num_tokens"] <= _remaining(seq)
                  and a["block_demand"] == 0)
        else:
            ok = (a["kind"] == DECODE and sid in before["running"]
                  and seq["prompt_processing_finished"] and a["num_tokens"] == 1
                  and a["block_demand"] == _gap(before, sid))
        check(ok, f"{tag}: ineligible or malformed action {a}")
    check(sum(a["num_tokens"] for a in actions) <= TOKEN_BUDGET,
          f"{tag}: selected tokens exceed the budget")
    admitted = sum(a["kind"] == ADMISSION for a in actions)
    check(len(before["running"]) - len(victims) + admitted <= resident_limit,
          f"{tag}: resident limit exceeded")
    # Emitted: victims, then the stable prompt-first partition of selection.
    ordered = ([a for a in actions if a["kind"] != DECODE]
               + [a for a in actions if a["kind"] == DECODE])
    layout = [[a["seq_id"], a["kind"], a["num_tokens"] if a["kind"] != DECODE
               else 0, a["block_demand"]] for a in ordered]
    return victims, layout


def expected_coefficients(policy, snap, request):
    """The coefficient triple the run's policy specifies for ``request``."""
    if policy == "lp_uniform":
        return UNIFORM
    order = list(snap["running_order"]) + list(snap["arrived_waiting_order"])
    r = order.index(request["raw_seq_id"])
    penalty = 0.0
    if request["preemption_eligible"]:
        s = list(snap["running_order"]).index(request["raw_seq_id"])
        penalty = RANK_SCALES["penalty_scale"] / (1 + s)
    return dict(decode_utility=RANK_SCALES["decode_scale"] / (1 + r),
                prefill_token_utility=RANK_SCALES["prefill_scale"] / (1 + r),
                preemption_penalty=penalty)


def check_lp_decision(tag, before, record, policy, resident_limit):
    events = record["lp_events"]
    check([e["call"] for e in events] == LP_CALLS and not record["proposals"],
          f"{tag}: LP calls {[e['call'] for e in events]}; expected one "
          "mapping, one solve, and one execution")
    mapped, solved, executed = events
    check(mapped["result_type"] == "StateSnapshot",
          f"{tag}: mapping returned {mapped['result']}")
    check(solved["result_type"] == "SchedulingSuccess",
          f"{tag}: solve returned {solved['result']}")
    check(executed["result_type"] == "SchedulerOutputs",
          f"{tag}: executor returned {executed['result']}")
    snap, result = mapped["result"], solved["result"]
    problem, plan = snap["lp_problem"], result["plan"]

    # Mapped inputs equal the observed before-state and the run inputs.
    check(snap["running_order"] == before["running"]
          and snap["arrived_waiting_order"] == before["waiting"]
          and snap["free_physical_blocks"] == problem["m_free"] == before["free_blocks"]
          and snap["watermark_blocks"] == problem["admission_watermark"]
          == before["watermark_blocks"]
          and snap["scheduler_iteration_id"] == before["iteration_id"] + 1
          and snap["num_running_batches"] == 0 and snap["num_pipeline_stages"] == 1,
          f"{tag}: mapped state differs from the observed state")
    check((problem["b_max"], problem["c_max"], problem["s_max"],
           problem["resident_limit"], problem["resident_count"])
          == (LP_B_MAX, LP_C_MAX, LP_S_MAX, resident_limit, len(before["running"]))
          and snap["numerical_policy"] == NUMERICAL_POLICY_FIELDS
          and problem["numerical_policy"] == NUMERICAL_POLICY_FIELDS,
          f"{tag}: mapped limits or numerical policy")
    facts = {r["request_id"]: r for r in snap["requests"]}
    inputs = {r["request_id"]: r for r in problem["requests"]}
    check(sorted(r["raw_seq_id"] for r in facts.values())
          == sorted(before["waiting"] + before["running"])
          and set(facts) == set(inputs), f"{tag}: mapped request set")
    residents = len(snap["running_order"])
    for rid, fact in facts.items():
        raw, lp_in = fact["raw_seq_id"], inputs[rid]
        want = expected_coefficients(policy, snap, fact)
        check(fact["utility"] == want
              and (lp_in["decode_utility"], lp_in["prefill_token_utility"],
                   lp_in["preemption_penalty"])
              == (want["decode_utility"], want["prefill_token_utility"],
                  want["preemption_penalty"]),
              f"{tag}: coefficients of {raw} {fact['utility']}; expected {want}")
        if raw in snap["running_order"]:
            pos = snap["running_order"].index(raw)
            ranks = (pos, residents - 1 - pos)
        else:
            ranks = (residents + snap["arrived_waiting_order"].index(raw), None)
        check((fact["service_rank"], fact["victim_rank"]) == ranks,
              f"{tag}: ranks of {raw}")

    # Integer-plan validity against the mapped problem (independent sums).
    check(plan["problem_id"] == problem["problem_id"]
          == result["problem_id"], f"{tag}: plan/problem identity")
    prefill = decode = actions = admitted = preempted = 0
    memory = problem["m_free"]
    objective = 0.0
    for d in plan["decisions"]:
        r = inputs[d["request_id"]]
        x, y, z = d["prefill_tokens"], d["decode"], d["preempt"]
        check((x > 0) + y + z <= 1 and y in (0, 1) and z in (0, 1) and x >= 0,
              f"{tag}: conflicting actions {d}")
        check(x == 0 or (r["prefill_eligible"] and x <= r["prefill_upper_bound"]
                         <= min(LP_C_MAX, r["prompt_tokens_remaining"])),
              f"{tag}: ineligible prefill {d}")
        check(not y or r["decode_eligible"], f"{tag}: ineligible decode {d}")
        check(not z or (r["preemption_eligible"]
                        and d["request_id"] in problem["legal_preemption_ids"]),
              f"{tag}: ineligible preemption {d}")
        prefill, decode = prefill + x, decode + y
        actions += (x > 0) + y
        admitted += x > 0 and not r["resident"]
        preempted += z
        memory += z * r["preemption_recovery"] - (
            r["prefill_fixed_charge"] if x > 0 and not r["resident"] else 0)
        objective += (x * r["prefill_token_utility"] + y * r["decode_utility"]
                      - z * r["preemption_penalty"])
    check(prefill + decode == plan["token_use"] <= LP_B_MAX
          and 1 <= actions == plan["action_count"] <= LP_S_MAX,
          f"{tag}: token or action limits")
    check(problem["resident_count"] - preempted + admitted
          == plan["resident_count_after"] <= resident_limit,
          f"{tag}: resident count after the plan")
    check(memory >= (problem["admission_watermark"] if admitted else 0)
          and memory == plan["free_blocks_before_decodes"]
          and memory - plan["decode_demand"] >= 0,
          f"{tag}: plan memory {memory} {plan['free_blocks_before_decodes']}")
    tol = NUMERICAL_POLICY_FIELDS["feasibility_tol"]
    check(abs(objective - plan["objective"]) <= tol * max(1.0, abs(objective)),
          f"{tag}: plan objective {plan['objective']} vs {objective}")

    # Emitted order is the derived declared order, and the executor's own
    # output is what the scheduler emitted.
    emitted = record["schedules"][0]["outputs"]
    check(executed["result"] == emitted, f"{tag}: executor output differs")
    victims, ordered = check_declared_order(tag, snap, plan, emitted)
    raw = {rid: f["raw_seq_id"] for rid, f in facts.items()}
    layout = []
    for d in ordered:
        r = inputs[d["request_id"]]
        if d["prefill_tokens"] > 0:
            kind = PREFILL if r["resident"] else ADMISSION
            layout.append([raw[d["request_id"]], kind, d["prefill_tokens"],
                           0 if r["resident"] else r["prefill_fixed_charge"]])
        else:
            layout.append([raw[d["request_id"]], DECODE, 0, r["decode_demand"]])
    return [raw[v] for v in victims], layout


# ---------------------------------------------------------------------------
# Policy-independent execution and replay checks
# ---------------------------------------------------------------------------


def check_execution(tag, before, record, victims, layout, resident_limit):
    """Central calls, emitted output, worker replay, sampler association,
    progress, completion, and release for one completed step."""
    check(len(record["schedules"]) == 1, f"{tag}: schedule calls")
    scheduled = record["schedules"][0]
    emitted = scheduled["outputs"]
    check(emitted["id"] == before["iteration_id"] + 1 and not emitted["ignored_seq_ids"],
          f"{tag}: output id or ignored ids")
    check(emitted["preempted_seq_ids"] == victims
          and emitted["scheduled"] == [[sid, chunk] for sid, _, chunk, _ in layout],
          f"{tag}: emitted {emitted} differs from the policy's order "
          f"{victims} {layout}")

    # Central native calls at scheduling, then completion frees.
    ops = record["central_ops"]
    sched_ops, completion_ops = ops[:scheduled["num_ops"]], ops[scheduled["num_ops"]:]
    want = ([["free", v] for v in victims]
            + [["allocate" if kind == ADMISSION else "append", sid]
               for sid, kind, _, _ in layout if kind != PREFILL])
    check([op[:2] for op in sched_ops] == want,
          f"{tag}: central calls {[op[:2] for op in sched_ops]}; expected {want}")
    demand = {sid: d for sid, _, _, d in layout}
    for op, sid, table_before, table_after in sched_ops:
        check(table_before == before["block_tables"].get(str(sid), []),
              f"{tag}: {op} {sid} saw table {table_before}")
        if op == "allocate":
            check(len(table_after) == demand[sid], f"{tag}: allocation of {sid}")
        elif op == "append":
            check(len(table_after) - len(table_before) == demand[sid],
                  f"{tag}: append growth of {sid}")
    sched_state = scheduled["state"]
    recovered = sum(len(before["block_tables"][str(v)]) for v in victims)
    check(sched_state["free_blocks"]
          == before["free_blocks"] + recovered - sum(demand.values())
          and sched_state["seqs"] == before["seqs"]
          and sched_state["num_running_batches"] == 1
          and len(sched_state["running"]) <= resident_limit,
          f"{tag}: central state at the end of scheduling")

    after, worker_after = record["after"], record["worker_after"]
    check(after["iteration_id"] == before["iteration_id"] + 1,
          f"{tag}: iteration advanced {after['iteration_id'] - before['iteration_id']}")
    check_quiescent(f"{tag} (after)", dict(num_gpu_blocks=NUM_GPU_BLOCKS,
                                           max_num_seqs=resident_limit),
                    after, worker_after)

    # Worker replay: preemption frees, then each emitted entry (allocation
    # if not yet allocated, otherwise append), then completion frees.
    finished = [o["seq_id"] for o in record["request_outputs"] if o["finished"]]
    worker_ops = [op[:2] for op in worker_after["block_ops"]]
    head = ([["free", v] for v in victims]
            + [["allocate" if kind == ADMISSION else "append", sid]
               for sid, kind, _, _ in layout])
    check(worker_ops[:len(head)] == head
          and sorted(worker_ops[len(head):]) == sorted(["free", f] for f in finished),
          f"{tag}: worker calls {worker_ops}; expected {head} then frees of "
          f"{finished}")
    check(sorted(op[:2] for op in completion_ops)
          == sorted(["free", f] for f in finished),
          f"{tag}: central completion calls {completion_ops}")

    # Samples and request outputs belong to the emitted entries, in order.
    sampler = record["sampler_outputs"]
    check(sampler is not None
          and [s for s, _ in sampler] == [s for s, _ in emitted["scheduled"]],
          f"{tag}: sampler outputs {sampler} vs {emitted['scheduled']}")
    check(sorted(o["seq_id"] for o in record["request_outputs"])
          == sorted(s for s, _ in emitted["scheduled"]),
          f"{tag}: request outputs {record['request_outputs']}")

    # Progress: prefill advances by its chunk, decode appends one token,
    # victims are reset for recomputation, everything else is unchanged.
    chunks = {sid: chunk for sid, _, chunk, _ in layout}
    for key, seq in before["seqs"].items():
        sid = int(key)
        if sid in finished:
            out = [o for o in record["request_outputs"] if o["seq_id"] == sid][0]
            check(chunks.get(sid) == 0 and out["finish_reason"] == "length"
                  and len(seq["output_token_ids"]) + 1 == MAX_TOKENS
                  and out["token_ids"] == seq["output_token_ids"] + [out["token_ids"][-1]]
                  and key not in after["seqs"] and key not in worker_after["seqs"]
                  and key not in after["block_tables"]
                  and key not in worker_after["block_tables"],
                  f"{tag}: finished request {sid} {out}")
            continue
        now = after["seqs"][key]
        if sid in victims:
            ok = (now["status"] == "WAITING" and now["prompt_tokens_processed"] == 0
                  and not now["prompt_processing_finished"]
                  and now["output_token_ids"] == []
                  and now["prompt_token_ids"]
                  == seq["prompt_token_ids"] + seq["output_token_ids"])
        elif chunks.get(sid, 0) > 0:
            done = seq["prompt_tokens_processed"] + chunks[sid]
            ok = (now["prompt_tokens_processed"] == done
                  and now["prompt_processing_finished"] == (done == seq["prompt_len"])
                  and now["output_token_ids"] == seq["output_token_ids"])
        elif sid in chunks:
            ok = (now["output_token_ids"][:-1] == seq["output_token_ids"]
                  and len(now["output_token_ids"]) == len(seq["output_token_ids"]) + 1
                  and now["prompt_tokens_processed"] == seq["prompt_tokens_processed"])
        else:
            ok = now == seq
        check(ok, f"{tag}: request {sid} progress {seq} -> {now}")
    return finished


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def _accounting():
    return {n: dict(first_prefill=None, first_decode=None, completion=None,
                    generated_token_events=0, preemptions=0,
                    discarded_prompt_tokens=0, discarded_generated_tokens=0,
                    prefill_tokens_executed=0, omitted_decisions=[])
            for n in NAMES.values()}


def _update_accounting(acc, index, before, victims, layout, finished):
    served = {sid for sid, _, _, _ in layout}
    for sid in victims:
        seq, a = before["seqs"][str(sid)], acc[NAMES[sid]]
        a["preemptions"] += 1
        a["discarded_prompt_tokens"] += seq["prompt_tokens_processed"]
        a["discarded_generated_tokens"] += len(seq["output_token_ids"])
    for sid, _, chunk, _ in layout:
        a = acc[NAMES[sid]]
        if chunk > 0:
            a["prefill_tokens_executed"] += chunk
            a["first_prefill"] = (index if a["first_prefill"] is None
                                  else a["first_prefill"])
        else:
            a["generated_token_events"] += 1
            a["first_decode"] = index if a["first_decode"] is None else a["first_decode"]
    for sid in before["waiting"] + before["running"]:
        if sid not in served and sid not in victims:
            acc[NAMES[sid]]["omitted_decisions"].append(index)
    for sid in finished:
        acc[NAMES[sid]]["completion"] = index


def run_comparison(policy, resident_limit, runtime, trace):
    """Submit the workload, observe at most MAX_DECISIONS nonempty decisions,
    and check each against its policy's contract, filling ``trace`` as it
    goes so a failure keeps the partial record. Returns the workload outcome
    (``drained`` or ``bounded_unfinished``); raises on any failure."""
    from sarathi.core.datatypes.sampling_params import SamplingParams

    scheduler = runtime.scheduler
    config = scheduler.scheduler_config
    expected_type = ("AdaptedVLLMScheduler" if policy == "adapted_vllm"
                     else "LPScheduler")
    check(type(scheduler).__name__ == expected_type,
          f"registered scheduler is {type(scheduler).__name__}")
    limits = (config.max_num_seqs, config.max_model_len, config.num_pipeline_stages,
              config.max_num_batched_tokens)
    check(limits == (resident_limit, MAX_MODEL_LEN, 1, TOKEN_BUDGET),
          f"scheduler configuration {limits}")
    if policy != "adapted_vllm":
        check((config.b_max, config.c_max, config.s_max)
              == (LP_B_MAX, LP_C_MAX, LP_S_MAX)
              and to_jsonable(config.numerical_policy) == NUMERICAL_POLICY_FIELDS
              and (config.decode_utility, config.prefill_token_utility,
                   config.preemption_penalty) == tuple(UNIFORM.values()),
              "LP scheduler configuration")
    manager = scheduler.block_manager
    check(manager.num_total_gpu_blocks == NUM_GPU_BLOCKS
          and manager.get_num_free_gpu_blocks() == NUM_GPU_BLOCKS
          and manager.watermark_blocks == 0 and not manager.block_tables
          and manager.block_size == BLOCK_SIZE, "central pool before requests")

    select_coefficient_policy(scheduler, policy)
    observer = DecisionObserver(scheduler, runtime.seq_map, policy)
    trace.update(policy=policy, resident_limit=resident_limit, prompts={},
                 arrival_times={}, decisions=[], final=None, idle=None)
    acc = _accounting()
    try:
        for sid, name in NAMES.items():
            runtime.submit(list(PROMPTS[name]), SamplingParams(
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                ignore_eos=IGNORE_EOS, stop=None))
            check(sorted(runtime.seq_map) == list(range(sid + 1)),
                  f"request {name} must be seq {sid}")
            trace["prompts"][name] = list(PROMPTS[name])
            trace["arrival_times"][name] = runtime.seq_map[sid].arrival_time
        runtime.worker_state()  # drop registration-time worker operations

        for index in range(MAX_DECISIONS):
            if not scheduler.has_unfinished_seqs():
                break
            tag = f"{policy} limit {resident_limit} decision {index}"
            before = central_state(scheduler, runtime.seq_map)
            before["arrival_times"] = {str(s): runtime.seq_map[s].arrival_time
                                       for s in runtime.seq_map}
            worker_before = runtime.worker_state()
            check_quiescent(f"{tag} (before)",
                            dict(num_gpu_blocks=NUM_GPU_BLOCKS,
                                 max_num_seqs=resident_limit),
                            before, worker_before)
            check(worker_before["block_ops"] == [],
                  f"{tag}: worker operations outside a step")
            observer.take()
            record = dict(index=index, before=before, worker_before=worker_before)
            trace["decisions"].append(record)
            try:
                request_outputs, sampler_outputs = runtime.step()
            except BaseException:
                record.update(observer.take(), central_at_failure=central_state(
                    scheduler, runtime.seq_map))
                raise
            record.update(observer.take())
            record.update(
                after=central_state(scheduler, runtime.seq_map),
                worker_after=runtime.worker_state(),
                sampler_outputs=sampler_outputs,
                request_outputs=[request_output_record(o) for o in request_outputs])
            check(record["schedules"]
                  and record["schedules"][0]["outputs"]["scheduled"],
                  f"{tag}: empty decision with unfinished arrived work")
            if index == 0:
                decided = (record["proposals"][0]["decision_time"]
                           if policy == "adapted_vllm"
                           else record["lp_events"][0]["result"]["snapshot_time"])
                record["decision_time"] = decided
                check(all(a <= decided for a in trace["arrival_times"].values()),
                      f"{tag}: requests not yet arrived at the first decision")
            if policy == "adapted_vllm":
                victims, layout = check_reference_decision(
                    tag, before, record, resident_limit)
            else:
                victims, layout = check_lp_decision(
                    tag, before, record, policy, resident_limit)
            finished = check_execution(tag, before, record, victims, layout,
                                       resident_limit)
            _update_accounting(acc, index, before, victims, layout, finished)
            record["accounting"] = json.loads(json.dumps(acc))
            print(f"{tag}: preempted {victims} emitted "
                  f"{record['schedules'][0]['outputs']['scheduled']} free "
                  f"{record['after']['free_blocks']}/"
                  f"{record['worker_after']['free_blocks']}", flush=True)

        final = central_state(scheduler, runtime.seq_map)
        worker_final = runtime.worker_state()
        unfinished = sorted(NAMES[int(k)] for k in final["seqs"])
        trace["final"] = dict(central=final, worker=worker_final,
                              unfinished=unfinished, accounting=acc)
        if scheduler.has_unfinished_seqs():
            # Bounded: retain the final native state; nothing is forced.
            check(unfinished and len(trace["decisions"]) == MAX_DECISIONS,
                  "unfinished work without reaching the decision bound")
            check_quiescent("final", dict(num_gpu_blocks=NUM_GPU_BLOCKS,
                                          max_num_seqs=resident_limit),
                            final, worker_final)
            return BOUNDED_UNFINISHED

        # Drained: full release, then one ordinary idle call.
        check(final["waiting"] == [] and final["running"] == []
              and final["seqs"] == {} and final["block_tables"] == {}
              and final["free_blocks"] == NUM_GPU_BLOCKS
              and worker_final["seqs"] == {} and worker_final["block_tables"] == {}
              and worker_final["free_blocks"] == NUM_GPU_BLOCKS
              and all(a["completion"] is not None for a in acc.values()),
              f"not drained {final} {worker_final}")
        observer.take()
        idle_outputs, idle_sampler = runtime.step()
        idle = observer.take()
        after = central_state(scheduler, runtime.seq_map)
        trace["idle"] = dict(before=final, after=after, worker_after=runtime.worker_state(),
                             **idle)
        expected_proposals = 1 if policy == "adapted_vllm" else 0
        check(list(idle_outputs) == [] and idle_sampler is None
              and len(idle["schedules"]) == 1
              and idle["schedules"][0]["outputs"]["scheduled"] == []
              and idle["schedules"][0]["outputs"]["preempted_seq_ids"] == []
              and idle["central_ops"] == [] and idle["lp_events"] == []
              and trace["idle"]["worker_after"]["block_ops"] == []
              and len(idle["proposals"]) == expected_proposals
              and all(p["actions"] == [] and p["victims"] == []
                      for p in idle["proposals"]),
              "idle step was not an ordinary empty decision")
        check({**after, "iteration_id": None} == {**final, "iteration_id": None}
              and after["iteration_id"] == final["iteration_id"] + 1,
              "idle step changed state")
        return DRAINED
    finally:
        observer.close()


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

PROVENANCE_FILES = [
    "scripts/compare_scheduler_gpu.py", "scripts/check_adapted_vllm_gpu.py",
    "scripts/check_lp_scheduler_gpu.py",
    "scripts/check_lp_scheduler_preemption_gpu.py",
    "tests/test_compare_scheduler_gpu.py",
    "sarathi/core/scheduler/adapted_vllm_scheduler.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/core/scheduler/base_scheduler.py", "sarathi/config.py",
    "sarathi/engine/arg_utils.py", "sarathi/engine/base_llm_engine.py",
    "sarathi/core/block_space_manager/base_block_space_manager.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/core/sequence_manager/worker_sequence_manager.py",
    "sarathi/worker/base_worker.py", "sarathi/model_executor/model_runner.py",
    "lp_relaxation_scheduler.py", "lpserve_state_mapping.py",
    "lpserve_plan_execution.py",
]


def provenance_record():
    return dict(
        git_head=git("rev-parse", "HEAD").strip(),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        git_status_short=git("status", "--short", "--untracked-files=all"),
        git_diff_head_sha256=hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        file_sha256={f: (sha256(REPO_ROOT / f) if (REPO_ROOT / f).exists()
                         else None) for f in PROVENANCE_FILES},
        imported_helper_sha256={m.__name__: sha256(m.__file__)
                                for m in (adapted, single, preemption)},
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--resident-limit", required=True, type=int,
                        choices=RESIDENT_LIMITS)
    parser.add_argument("--model-path", required=True,
                        help="existing local directory with the TinyLlama "
                        "config and tokenizer")
    parser.add_argument("--output-dir", required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / f).exists() for f in ("decision_trace.json", "summary.json")):
        parser.error(f"{out_dir} already holds run artifacts")
    model_path = Path(args.model_path).resolve()

    # Local assets only; never download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    trace = {}
    summary = dict(validation_passed=False, workload_outcome=FAILURE, failure=None,
                   run=run_parameters(args.policy, args.resident_limit),
                   process_timeout_seconds=PROCESS_TIMEOUT_SECONDS,
                   provenance=provenance_record(), argv=sys.argv,
                   model_path=str(model_path))
    engine = None
    try:
        check((model_path / "config.json").is_file()
              and (model_path / "tokenizer.json").is_file(),
              f"model path {model_path} lacks config.json/tokenizer.json")
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import sarathi and the LP modules")

        args_record = engine_args(args.policy, args.resident_limit, model_path,
                                  out_dir / "metrics_store_unused")
        configs = args_record.create_engine_configs()
        model_config, cache_config, parallel_config, scheduler_config, _ = configs
        check((parallel_config.tensor_parallel_size,
               parallel_config.pipeline_parallel_size)
              == (TENSOR_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE)
              and cache_config.block_size == BLOCK_SIZE
              and model_config.max_model_len == MAX_MODEL_LEN
              and scheduler_config.max_num_seqs == args.resident_limit,
              "native configuration differs from the run")

        summary["environment"] = environment_record()
        print("environment:", json.dumps(summary["environment"], indent=1), flush=True)
        register_worker_helpers_by_value()
        selection = CacheCapacitySelection(chosen=NUM_GPU_BLOCKS)
        t0 = time.monotonic()
        engine = build_engine_class(selection)(*configs)
        init_seconds = time.monotonic() - t0
        runtime = EngineRuntime(engine, engine.tokenizer,
                                model_config.hf_config.vocab_size)
        worker_init = runtime.worker_state()
        summary["cache_capacity"] = dict(
            **selection.record(),
            engine_cache_config_num_gpu_blocks=cache_config.num_gpu_blocks,
            central_manager_total_blocks=(
                engine.scheduler.block_manager.num_total_gpu_blocks),
            worker={k: v for k, v in worker_init.items() if k != "seqs"})
        print("cache capacity:", json.dumps(to_jsonable(summary["cache_capacity"])),
              flush=True)
        adapted.check_capacity(dict(num_gpu_blocks=NUM_GPU_BLOCKS), selection,
                               cache_config, engine.scheduler.block_manager,
                               worker_init)
        for start, name in ((1000, "A"), (2000, "B")):
            check(select_prompt_ids(engine.tokenizer, runtime.vocab_size, start,
                                    len(PROMPTS[name]))[0] == PROMPTS[name],
                  f"prompt {name} contains a special or out-of-vocabulary token")
        summary["config"] = dict(
            model=MODEL_NAME, engine_args=to_jsonable(vars(args_record)),
            resolved_dtype=str(model_config.dtype),
            hf_config_commit_hash=getattr(model_config.hf_config, "_commit_hash",
                                          None),
            scheduler_class=type(engine.scheduler).__name__,
            scheduler_config=to_jsonable(vars(scheduler_config)),
            vocab_size=runtime.vocab_size,
            eos_token_id=engine.tokenizer.eos_token_id,
            engine_init_seconds=round(init_seconds, 3))
        print("config:", json.dumps(summary["config"]), flush=True)

        outcome = run_comparison(args.policy, args.resident_limit, runtime, trace)
        summary.update(validation_passed=True, workload_outcome=outcome,
                       decisions=len(trace["decisions"]),
                       unfinished=trace["final"]["unfinished"],
                       accounting=trace["final"]["accounting"])
    except BaseException as err:
        failure = dict(type=type(err).__name__, message=str(err),
                       traceback=traceback.format_exc())
        for name in ("stage", "category", "reason"):
            if hasattr(err, name):
                failure[name] = getattr(err, name)
        summary["failure"] = failure
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        cleanup = dict(engine_constructed=engine is not None)
        try:
            (out_dir / "decision_trace.json").write_text(
                json.dumps(to_jsonable(trace), indent=1))
            (out_dir / "summary.json").write_text(
                json.dumps(to_jsonable(summary), indent=1))
        finally:
            # Stop only the Ray runtime this process started.
            if "ray" in sys.modules:
                ray = sys.modules["ray"]
                cleanup["ray_initialized_before_shutdown"] = ray.is_initialized()
                ray.shutdown()
                cleanup["ray_initialized_after_shutdown"] = ray.is_initialized()
            print("cleanup:", json.dumps(cleanup), flush=True)
            (out_dir / "cleanup.json").write_text(json.dumps(cleanup, indent=1))
    print("RESULT:", "PASS" if summary["validation_passed"] else "FAIL",
          "workload:", summary["workload_outcome"], flush=True)
    return 0 if summary["validation_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
