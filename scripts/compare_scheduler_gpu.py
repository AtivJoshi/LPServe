"""Bounded dummy-weight GPU comparison of the adapted vLLM reference and
LP coefficient policies on one of three small workloads.

One process runs one (case, policy, resident limit) through a real
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
- ``lp_coordinated``: the registered ``LPScheduler`` with
  ``lpserve_state_mapping.coordinated_rank_utility`` (rank base 1/4, bound to
  the case's ``c_max`` and ``b_max``) selected through the same hook.
- ``lp_reciprocal_square``: the registered ``LPScheduler`` with
  ``lpserve_state_mapping.reciprocal_square_rank_utility`` (weight
  ``1 / (1 + r) ** 2``, bound to the case's ``c_max`` and ``b_max``)
  selected through the same hook.

The LP policies use numerical policy ``lp_relaxation_mvp_v1``.

CPU-only additions (``run_comparison`` with a CPU runtime; never accepted on
the command line): the methods ``ilp_coordinated`` and
``ilp_reciprocal_square``, which take each decision from the integer optimum
of the same coordinated or reciprocal-square problem
(``exact_integer_plan.integer_decision``); the representative cases
``REPRESENTATIVE_CASES`` (six requests, optional per-request output limits,
and virtual arrival times; see ``case_inputs``); and the queue-growth cases
``QUEUE_GROWTH_CASES`` other than ``queue48_burst_pool12``.

Cases (``--case``):

- ``two_request`` (default): A (5 prompt tokens) then B (8), two output
  tokens each, both submitted before the first decision; block size 16, 8
  initialized blocks, ``max_model_len`` 32, budget 8, resident limit 2 or 1,
  LP ``b_max`` 8, ``c_max`` 8, ``s_max`` 1 (the action cap is a deliberate
  difference from the reference), bound 8 nonempty decisions.
- ``memory_pressure``: the reference check's ``preemption`` workload. A, B,
  C, D (16 prompt tokens each, explicit ranges from 1000/2000/3000/4000),
  three output tokens each, all submitted before the first decision; block
  size 16, 4 initialized blocks, ``max_model_len`` 64, budget 64, resident
  limit 4, LP ``b_max`` 64, ``c_max`` 64, ``s_max`` 4 (no restriction beyond
  the budget and resident limit), bound 16 nonempty decisions. The four
  prefills can fill the pool while decode demand is still zero: the
  reference runs zero-demand decodes with zero free blocks, while the LP's
  decode prerequisite requires at least one free block before any decode.
  Both contracts are kept; each decision records which constraint, if any,
  required its victims for the selected batch.
- ``queue48_burst_pool12``: 48 requests at virtual time 0 (see
  ``QUEUE_GROWTH_INPUTS``), 12 initialized blocks, ``max_model_len`` 256,
  budget 128, resident limit 8, LP ``b_max`` 128, ``c_max`` 128, ``s_max``
  8, bound 384 nonempty decisions. For the reference, each decision's
  pre-decision state is also mapped read-only into a reciprocal-square
  diagnostic problem (``capture_reference_problems``) for an offline CPU
  integer comparison; the LP runs retain their own mapped problems.

The decision sequence is not prescribed. Each nonempty decision is checked
against its policy's own contract (mapped or proposed inputs, plan or
proposal validity, emitted order, central and worker block operations,
replay, sampler association, progress, completion and release) and recorded
with the before/after state. The case's decision bound limits the
observation: work still unfinished at the bound is reported as
``bounded_unfinished``, not forced to finish. A scheduler, mapping, solver,
execution, worker, or check error is a ``failure``. Generated-token events count emitted decode entries
cumulatively; the native output counter resets on recomputation, so a
restarted request can produce more events than ``max_tokens`` (inherited,
reported as overgeneration). Executed prefill tokens are split into
first-time original-prompt positions, replayed positions, and positions of
generated tokens moved into the prompt (expanded context).

Not established: physical-ID equality between managers, cache-content or
attention correctness, sampler generality, generation quality, policy
equivalence or recovery, starvation, fairness, or performance.

Run from the repository root with the root on PYTHONPATH, one pair per
process, on an allocated GPU node:

    PYTHONPATH="$PWD" timeout 600s python -B scripts/compare_scheduler_gpu.py \\
        --policy lp_uniform --resident-limit 2 \\
        --model-path <local TinyLlama snapshot directory> \\
        --output-dir validation_output/scheduler_comparison_gpu/<UTC>/<run>

The memory-pressure case adds ``--case memory_pressure`` (resident limit 4,
the default for that case).

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
# Coordinated policy: w = RANK_BASE ** r (provisional, not tuned).
COORDINATED = dict(rank_base=0.25)
# Reciprocal-square coordinated policy: w = 1 / (1 + r) ** 2 (provisional;
# the exponent is fixed at two and was not tuned).
RECIPROCAL_SQUARE_POLICY = "lp_reciprocal_square"
POLICIES = ("adapted_vllm", "lp_uniform", "lp_rank_based", "lp_coordinated",
            RECIPROCAL_SQUARE_POLICY)
INTEGER_POLICY = "ilp_coordinated"  # CPU comparison only
RECIPROCAL_SQUARE_INTEGER_POLICY = "ilp_reciprocal_square"  # CPU only
INTEGER_POLICIES = (INTEGER_POLICY, RECIPROCAL_SQUARE_INTEGER_POLICY)
CPU_POLICIES = POLICIES + INTEGER_POLICIES
COORDINATED_POLICIES = ("lp_coordinated", INTEGER_POLICY)
RECIPROCAL_SQUARE_POLICIES = (RECIPROCAL_SQUARE_POLICY,
                              RECIPROCAL_SQUARE_INTEGER_POLICY)
MAX_DECISIONS = 8  # observation bound per run, not a scheduler restart cap
TWO_REQUEST, MEMORY_PRESSURE = "two_request", "memory_pressure"
GPU_CASES = (TWO_REQUEST, MEMORY_PRESSURE, "queue48_burst_pool12")
# The reference check's ``preemption`` workload (same prompt IDs, outputs,
# pool, length, budget, and resident limit). LP caps equal the budget and
# resident limit, so they add no restriction in this case.
MEMORY_PRESSURE_INPUTS = dict(
    prompts=dict(A=list(range(1000, 1016)), B=list(range(2000, 2016)),
                 C=list(range(3000, 3016)), D=list(range(4000, 4016))),
    max_tokens=3, token_budget=64, num_gpu_blocks=4, max_model_len=64,
    resident_limits=(4,), b_max=64, c_max=64, s_max=4, max_decisions=16)
# Representative CPU workloads (provisional inputs): six requests A..F with
# disjoint prompt IDs from 1000 * (k + 1), block size 16, max_model_len 64,
# resident limit 4, budget/b_max/c_max 32, s_max 4, bound 64 nonempty
# decisions. Virtual arrivals: the clock is the number of nonempty decisions
# executed; a request is submitted before the first decision at or after its
# arrival time, and when nothing is unfinished the clock jumps to the next
# arrival without a scheduler call. Staggered: request k arrives at 2k.
EQUAL_LENGTHS, MIXED_LENGTHS = [16] * 6, [5, 8, 16, 17, 31, 32]
MIXED_OUTPUTS = [2, 3, 2, 4, 3, 4]
STAGGERED_ARRIVALS = [2 * k for k in range(6)]


def _representative(lengths, outputs, arrivals, pool):
    names = "ABCDEF"
    return dict(
        prompts={n: list(range(1000 * (k + 1), 1000 * (k + 1) + lengths[k]))
                 for k, n in enumerate(names)},
        max_tokens=(outputs if isinstance(outputs, int)
                    else dict(zip(names, outputs))),
        arrivals=dict(zip(names, arrivals)), token_budget=32,
        num_gpu_blocks=pool, max_model_len=64, resident_limits=(4,), b_max=32,
        c_max=32, s_max=4, max_decisions=64)


REPRESENTATIVE_INPUTS = {
    "equal_burst_ample": _representative(EQUAL_LENGTHS, 3, [0] * 6, 16),
    "mixed_burst_ample": _representative(MIXED_LENGTHS, MIXED_OUTPUTS,
                                         [0] * 6, 16),
    "mixed_staggered_ample": _representative(MIXED_LENGTHS, MIXED_OUTPUTS,
                                             STAGGERED_ARRIVALS, 16),
    "mixed_staggered_tight": _representative(MIXED_LENGTHS, MIXED_OUTPUTS,
                                             STAGGERED_ARRIVALS, 6),
}
REPRESENTATIVE_CASES = tuple(REPRESENTATIVE_INPUTS)
# Queue-growth workloads (provisional inputs): request k has prompt length
# QUEUE_PROMPT_LENGTHS[k % 6], output limit QUEUE_OUTPUT_LENGTHS[k % 3], and
# prompt IDs from 1000 + 256 * k. Block size 16, max_model_len 256, resident
# limit 8, budget/b_max/c_max 128, s_max 8, bound 8 nonempty decisions per
# request. Burst: every request at virtual time 0; staggered: request k at
# 2k. The watermark is the shared block manager's native int(0.01 * pool).
QUEUE_PROMPT_LENGTHS, QUEUE_OUTPUT_LENGTHS = [8, 16, 17, 32, 64, 128], [2, 4, 8]


def _queue_growth(count, staggered, pool):
    names = [f"R{k:02d}" for k in range(count)]
    return dict(
        prompts={n: list(range(1000 + 256 * k, 1000 + 256 * k
                               + QUEUE_PROMPT_LENGTHS[k % 6]))
                 for k, n in enumerate(names)},
        max_tokens={n: QUEUE_OUTPUT_LENGTHS[k % 3]
                    for k, n in enumerate(names)},
        arrivals={n: 2 * k if staggered else 0 for k, n in enumerate(names)},
        token_budget=128, num_gpu_blocks=pool, max_model_len=256,
        resident_limits=(8,), b_max=128, c_max=128, s_max=8,
        max_decisions=8 * count)


CONSTRAINED_QUEUE_CASE = "queue48_burst_pool12"
QUEUE_GROWTH_INPUTS = {
    "queue6_burst": _queue_growth(6, False, 128),
    "queue12_burst": _queue_growth(12, False, 128),
    "queue24_burst": _queue_growth(24, False, 128),
    "queue48_burst": _queue_growth(48, False, 128),
    "queue48_staggered": _queue_growth(48, True, 128),
    CONSTRAINED_QUEUE_CASE: _queue_growth(48, False, 12),
}
QUEUE_GROWTH_CASES = tuple(QUEUE_GROWTH_INPUTS)
CASES = (TWO_REQUEST, MEMORY_PRESSURE) + REPRESENTATIVE_CASES + QUEUE_GROWTH_CASES
PROCESS_TIMEOUT_SECONDS = 600
PROFILE_METHOD = preemption.PROFILE_METHOD
INIT_CACHE_METHOD = preemption.INIT_CACHE_METHOD

DRAINED, BOUNDED_UNFINISHED, FAILURE = "drained", "bounded_unfinished", "failure"
ADMISSION, PREFILL, DECODE = "admission", "resident_prefill", "decode"
LP_CALLS = ["map_scheduler_state", "solve_and_extract", "execute_plan"]
INTEGER_CALLS = ["map_scheduler_state", "solve_integer_plan", "execute_plan"]


def decision_calls(policy):
    return INTEGER_CALLS if policy in INTEGER_POLICIES else LP_CALLS


def case_inputs(case):
    """The workload, capacity, limit, and LP cap inputs of ``case``. The
    two-request values are read from the module constants at call time.
    ``max_tokens`` is one limit or a per-request mapping, and ``arrivals``
    maps each request to its virtual arrival time (0 for every request when
    a case gives none)."""
    check(case in CASES, f"unknown case {case}")
    if case != TWO_REQUEST:
        inputs = json.loads(json.dumps({
            MEMORY_PRESSURE: MEMORY_PRESSURE_INPUTS, **REPRESENTATIVE_INPUTS,
            **QUEUE_GROWTH_INPUTS}[case]))
        inputs["resident_limits"] = tuple(inputs["resident_limits"])
    else:
        inputs = dict(prompts=PROMPTS, max_tokens=MAX_TOKENS,
                      token_budget=TOKEN_BUDGET, num_gpu_blocks=NUM_GPU_BLOCKS,
                      max_model_len=MAX_MODEL_LEN,
                      resident_limits=RESIDENT_LIMITS, b_max=LP_B_MAX,
                      c_max=LP_C_MAX, s_max=LP_S_MAX,
                      max_decisions=MAX_DECISIONS)
    inputs["names"] = dict(enumerate(inputs["prompts"]))
    inputs.setdefault("arrivals", {n: 0 for n in inputs["prompts"]})
    return inputs


def max_tokens(c, name):
    """The output limit of request ``name`` in case inputs ``c``."""
    limit = c["max_tokens"]
    return limit[name] if isinstance(limit, dict) else limit


def run_parameters(policy, resident_limit, case=TWO_REQUEST):
    """The provisional inputs of one run, as recorded and checked."""
    c = case_inputs(case)
    check(policy in CPU_POLICIES and resident_limit in c["resident_limits"],
          f"unknown run {case} {policy} {resident_limit}")
    params = dict(
        case=case, policy=policy, resident_limit=resident_limit,
        block_size=BLOCK_SIZE, num_gpu_blocks=c["num_gpu_blocks"],
        max_model_len=c["max_model_len"],
        native_watermark_blocks=int(0.01 * c["num_gpu_blocks"]),
        prompts=c["prompts"], max_tokens=c["max_tokens"],
        virtual_arrivals=c["arrivals"],
        token_budget=c["token_budget"], max_decisions=c["max_decisions"])
    if policy == "adapted_vllm":
        params.update(scheduler_type="adapted_vllm",
                      max_num_batched_tokens=c["token_budget"],
                      chunk_cap=None, action_cap=None)
    else:
        params.update(scheduler_type="lp", b_max=c["b_max"], c_max=c["c_max"],
                      s_max=c["s_max"], numerical_policy=NUMERICAL_POLICY_FIELDS,
                      configured_triple=UNIFORM,
                      coefficient_policy=coefficient_policy_record(policy, c))
    if policy in INTEGER_POLICIES:
        import exact_integer_plan as eip
        params.update(decision_method="integer optimum (CPU comparison only)",
                      integer_solver=eip.SOLVER,
                      integer_solver_options=eip.SOLVER_OPTIONS,
                      integer_conversion_tol=eip.CONVERSION_TOL)
    return params


def coefficient_policy_record(policy, c):
    if policy == "lp_uniform":
        return "configured uniform triple"
    if policy == "lp_rank_based":
        return dict(rank_based_utility=RANK_SCALES,
                    note="configured triple unused")
    if policy in RECIPROCAL_SQUARE_POLICIES:
        return dict(reciprocal_square_rank_utility=dict(c_max=c["c_max"],
                                                        b_max=c["b_max"]),
                    formula="w = 1 / (1 + r) ** 2 (r: running then arrived "
                    "waiting position); q = min(remaining, c_max, b_max); "
                    "decode w; prefill token w / q; penalty w",
                    note="configured triple unused")
    return dict(coordinated_rank_utility=dict(COORDINATED, c_max=c["c_max"],
                                              b_max=c["b_max"]),
                formula="w = rank_base ** r (r: running then arrived waiting "
                "position); q = min(remaining, c_max, b_max); decode w; "
                "prefill token w / q; penalty w",
                note="configured triple unused")


def engine_args(policy, resident_limit, model_path, metrics_dir,
                case=TWO_REQUEST):
    """Native ``EngineArgs`` for one run."""
    from sarathi.engine.arg_utils import EngineArgs

    c = case_inputs(case)
    common = dict(
        model=str(model_path), tokenizer=str(model_path),
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=c["max_model_len"], max_num_seqs=resident_limit,
        # Enabled metrics with optional outputs off, as in the earlier
        # bounded checks; nothing is plotted.
        write_metrics=True, output_dir=str(metrics_dir),
        enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
        enable_chrome_trace=False, enable_request_outputs=False,
        keep_individual_batch_metrics=True,
        attention_backend=ATTENTION_BACKEND)
    if policy == "adapted_vllm":
        return EngineArgs(scheduler_type="adapted_vllm",
                          max_num_batched_tokens=c["token_budget"], **common)
    from lp_relaxation_scheduler import NumericalPolicy
    return EngineArgs(
        scheduler_type="lp", lp_b_max=c["b_max"], lp_c_max=c["c_max"],
        lp_s_max=c["s_max"], lp_decode_utility=UNIFORM["decode_utility"],
        lp_prefill_token_utility=UNIFORM["prefill_token_utility"],
        lp_preemption_penalty=UNIFORM["preemption_penalty"],
        lp_numerical_policy=NumericalPolicy(**NUMERICAL_POLICY_FIELDS),
        **common)


def select_coefficient_policy(scheduler, policy, case=TWO_REQUEST):
    """Rank-based and coordinated runs override the live hook on this
    instance only. The integer method additionally replaces this instance's
    decision with ``exact_integer_plan.integer_decision`` (CPU only)."""
    import lpserve_state_mapping as lsm
    if policy == "lp_rank_based":
        coefficients = functools.partial(lsm.rank_based_utility, **RANK_SCALES)
    elif policy in COORDINATED_POLICIES:
        c = case_inputs(case)
        coefficients = functools.partial(
            lsm.coordinated_rank_utility, c_max=c["c_max"], b_max=c["b_max"],
            **COORDINATED)
    elif policy in RECIPROCAL_SQUARE_POLICIES:
        coefficients = reciprocal_square_policy(case_inputs(case))
    else:
        return
    scheduler._coefficient_policy = lambda: coefficients
    if policy in INTEGER_POLICIES:
        import exact_integer_plan as eip
        scheduler._schedule = functools.partial(eip.integer_decision, scheduler)


def reciprocal_square_policy(c):
    """The reciprocal-square policy bound to case inputs ``c``'s LP caps."""
    import lpserve_state_mapping as lsm
    return functools.partial(lsm.reciprocal_square_rank_utility,
                             c_max=c["c_max"], b_max=c["b_max"])


def capture_reference_problems(runtime, case):
    """Wrap ``runtime.step`` so that, before each reference decision with
    unfinished work, the quiescent pre-decision state is mapped once,
    read-only, with the reciprocal-square coefficients and the case's LP
    caps. Returns the list that receives one JSON record (``LPProblem`` or
    ``MappingFailure``) per such step, in decision order. Diagnostic only:
    the reference does not read it, and ordinary idle steps map nothing."""
    import lp_relaxation_scheduler as lrs
    import lpserve_state_mapping as lsm
    c, records, native = case_inputs(case), [], runtime.step

    def step():
        scheduler = runtime.scheduler
        if scheduler.has_unfinished_seqs():
            snap = lsm.map_scheduler_state(
                scheduler, snapshot_time=time.monotonic(), b_max=c["b_max"],
                c_max=c["c_max"], s_max=c["s_max"],
                coefficient_policy=reciprocal_square_policy(c),
                rank_profile=lsm.vllm_oriented_ranks,
                numerical_policy=lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS))
            records.append(dict(
                result_type=type(snap).__name__,
                result=to_jsonable(getattr(snap, "lp_problem", snap))))
        return native()
    runtime.step = step
    return records


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
            solver = lrs
            if policy in INTEGER_POLICIES:
                import exact_integer_plan as solver
            for module, name in zip((lsm, solver, lpe), decision_calls(policy)):
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


def check_reference_decision(tag, before, record, resident_limit, c):
    check(len(record["proposals"]) == 1 and record["lp_events"] == [],
          f"{tag}: {len(record['proposals'])} reference proposals")
    p = record["proposals"][0]
    check(p["running_order"] == before["running"]
          and p["arrived_waiting_order"] == before["waiting"]
          and p["free_blocks"] == before["free_blocks"]
          and p["watermark_blocks"] == before["watermark_blocks"]
          and p["token_budget"] == c["token_budget"],
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
    check(sum(a["num_tokens"] for a in actions) <= c["token_budget"],
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


def expected_coefficients(policy, snap, request, c):
    """The coefficient triple the run's policy specifies for ``request``."""
    if policy == "lp_uniform":
        return UNIFORM
    order = list(snap["running_order"]) + list(snap["arrived_waiting_order"])
    r = order.index(request["raw_seq_id"])
    if policy in COORDINATED_POLICIES + RECIPROCAL_SQUARE_POLICIES:
        w = (COORDINATED["rank_base"] ** r if policy in COORDINATED_POLICIES
             else 1.0 / (1 + r) ** 2)
        q = min(request["prompt_tokens_remaining"], c["c_max"], c["b_max"])
        return dict(
            decode_utility=w if request["decode_eligible"] else 0.0,
            prefill_token_utility=w / q if request["prefill_eligible"] else 0.0,
            preemption_penalty=w if request["preemption_eligible"] else 0.0)
    penalty = 0.0
    if request["preemption_eligible"]:
        s = list(snap["running_order"]).index(request["raw_seq_id"])
        penalty = RANK_SCALES["penalty_scale"] / (1 + s)
    return dict(decode_utility=RANK_SCALES["decode_scale"] / (1 + r),
                prefill_token_utility=RANK_SCALES["prefill_scale"] / (1 + r),
                preemption_penalty=penalty)


def check_lp_decision(tag, before, record, policy, resident_limit, c):
    events = record["lp_events"]
    integer = policy in INTEGER_POLICIES
    check([e["call"] for e in events] == decision_calls(policy)
          and not record["proposals"],
          f"{tag}: LP calls {[e['call'] for e in events]}; expected one "
          "mapping, one solve, and one execution")
    mapped, solved, executed = events
    check(mapped["result_type"] == "StateSnapshot",
          f"{tag}: mapping returned {mapped['result']}")
    check(solved["result_type"] == ("IntegerOptimum" if integer
                                    else "SchedulingSuccess"),
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
          == (c["b_max"], c["c_max"], c["s_max"], resident_limit,
              len(before["running"]))
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
        want = expected_coefficients(policy, snap, fact, c)
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
    check(sorted(d["request_id"] for d in plan["decisions"]) == sorted(inputs),
          f"{tag}: plan decisions are not one per mapped request")
    prefill = decode = actions = admitted = preempted = 0
    memory = problem["m_free"]
    objective = 0.0
    for d in plan["decisions"]:
        r = inputs[d["request_id"]]
        x, y, z = d["prefill_tokens"], d["decode"], d["preempt"]
        check((x > 0) + y + z <= 1 and y in (0, 1) and z in (0, 1) and x >= 0,
              f"{tag}: conflicting actions {d}")
        check(x == 0 or (r["prefill_eligible"] and x <= r["prefill_upper_bound"]
                         <= min(c["c_max"], r["prompt_tokens_remaining"])),
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
    check(prefill + decode == plan["token_use"] <= c["b_max"]
          and 1 <= actions == plan["action_count"] <= c["s_max"],
          f"{tag}: token or action limits")
    check(problem["resident_count"] - preempted + admitted
          == plan["resident_count_after"] <= resident_limit,
          f"{tag}: resident count after the plan")
    demand = sum(inputs[d["request_id"]]["decode_demand"] * d["decode"]
                 for d in plan["decisions"])
    check(memory >= (problem["admission_watermark"] if admitted else 0)
          and memory == plan["free_blocks_before_decodes"]
          and demand == plan["decode_demand"] and memory - demand >= 0
          and (decode == 0 or memory >= 1),
          f"{tag}: plan memory H={memory} ({plan['free_blocks_before_decodes']}) "
          f"demand {demand} decodes {decode}: actual memory or decode "
          "prerequisite")
    tol = NUMERICAL_POLICY_FIELDS["feasibility_tol"]
    check(abs(objective - plan["objective"]) <= tol * max(1.0, abs(objective)),
          f"{tag}: plan objective {plan['objective']} vs {objective}")
    if integer:
        # A direct integer optimum: no relaxation was solved or extracted.
        source = plan["extraction"]
        check(source["raw_status"] == 0
              and source["max_integrality_deviation"] <= source["conversion_tol"]
              and abs(source["objective"] - objective)
              <= tol * max(1.0, abs(objective)),
              f"{tag}: integer optimum provenance {source}")
        record["lp_comparison"] = None
        record["integer_solution"] = dict(
            objective=objective, dual_bound=source["dual_bound"],
            reported_gap=source["reported_gap"],
            node_count=source["node_count"])
    else:
        _check_relaxed_bound(tag, record, result, plan, facts, inputs,
                             objective, tol)

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


def _check_relaxed_bound(tag, record, result, plan, facts, inputs, objective,
                         tol):
    # The integer plan is feasible for the relaxation, so the relaxed
    # objective (recomputed from the relaxed values) bounds it above.
    relaxed = {d["request_id"]: d for d in result["relaxed"]["decisions"]}
    check(set(relaxed) == set(inputs), f"{tag}: relaxed decisions")
    relaxed_objective = sum(
        relaxed[rid]["x"] * r["prefill_token_utility"]
        + relaxed[rid]["y"] * r["decode_utility"]
        - relaxed[rid]["z"] * r["preemption_penalty"]
        for rid, r in inputs.items())
    check(relaxed_objective >= objective - tol * max(1.0, abs(objective)),
          f"{tag}: relaxed objective {relaxed_objective} below the integer "
          f"plan's {objective}")
    values = {facts[rid]["raw_seq_id"]: [v["x"], v["y"], v["prefill_indicator"],
                                         v["z"]] for rid, v in relaxed.items()}
    chosen = {facts[d["request_id"]]["raw_seq_id"]:
              [d["prefill_tokens"], d["decode"], d["prefill_indicator"],
               d["preempt"]] for d in plan["decisions"]}
    eps = NUMERICAL_POLICY_FIELDS["integrality_tol"]
    record["lp_comparison"] = dict(
        relaxed_objective=relaxed_objective, integer_objective=objective,
        fractional_request_count=plan["extraction"]["fractional_request_count"],
        extraction_changed_relaxed=any(
            abs(a - b) > eps for k in values for a, b in zip(values[k], chosen[k])),
        relaxed=values, integer=chosen)


# ---------------------------------------------------------------------------
# Policy-independent execution and replay checks
# ---------------------------------------------------------------------------


def check_execution(tag, before, record, victims, layout, resident_limit, c):
    """Central calls, emitted output, worker replay, sampler association,
    queue order, progress, completion, and release for one completed step."""
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
    # Retained residents keep their order, admissions are appended in
    # emitted order, and victims are front-inserted in preemption order.
    admitted = [sid for sid, kind, _, _ in layout if kind == ADMISSION]
    running_order = [s for s in before["running"] if s not in victims] + admitted
    waiting_order = victims[::-1] + [s for s in before["waiting"]
                                     if s not in admitted]
    check(sched_state["free_blocks"]
          == before["free_blocks"] + recovered - sum(demand.values())
          and sched_state["seqs"] == before["seqs"]
          and sched_state["num_running_batches"] == 1
          and sched_state["running"] == running_order
          and sched_state["waiting"] == waiting_order
          and len(sched_state["running"]) <= resident_limit,
          f"{tag}: central state at the end of scheduling (running "
          f"{sched_state['running']} vs {running_order}, waiting "
          f"{sched_state['waiting']} vs {waiting_order})")

    after, worker_after = record["after"], record["worker_after"]
    check(after["iteration_id"] == before["iteration_id"] + 1,
          f"{tag}: iteration advanced {after['iteration_id'] - before['iteration_id']}")
    check_quiescent(f"{tag} (after)", dict(num_gpu_blocks=c["num_gpu_blocks"],
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
    check(after["running"] == [s for s in running_order if s not in finished]
          and after["waiting"] == waiting_order,
          f"{tag}: queues after completion {after['running']} "
          f"{after['waiting']}; expected {running_order} less {finished}, "
          f"{waiting_order}")
    sampled = dict(sampler)
    outputs = {o["seq_id"]: o for o in record["request_outputs"]}

    # Progress: prefill advances by its chunk, decode appends one token,
    # victims are reset for recomputation, everything else is unchanged.
    chunks = {sid: chunk for sid, _, chunk, _ in layout}
    for key, seq in before["seqs"].items():
        sid = int(key)
        if sid in finished:
            # Native completion: the counter (reset by recomputation)
            # reaches max_tokens with this step's actual sampled token.
            out = outputs[sid]
            check(chunks.get(sid) == 0 and out["finish_reason"] == "length"
                  and len(seq["output_token_ids"]) + 1
                  == max_tokens(c, c["names"][sid])
                  and out["token_ids"] == seq["output_token_ids"] + [sampled[sid]]
                  and out["prompt_token_ids"] == seq["prompt_token_ids"]
                  and key not in after["seqs"] and key not in worker_after["seqs"]
                  and key not in after["block_tables"]
                  and key not in worker_after["block_tables"],
                  f"{tag}: finished request {sid} {out}")
            continue
        now = after["seqs"][key]
        if sid in victims:
            # Recomputation: generated tokens move into the prompt and the
            # native output counter resets; the logical context is kept.
            ok = (now["status"] == "WAITING" and now["prompt_tokens_processed"] == 0
                  and not now["prompt_processing_finished"]
                  and now["output_token_ids"] == []
                  and now["prompt_token_ids"]
                  == seq["prompt_token_ids"] + seq["output_token_ids"]
                  and now["prompt_len"] == len(now["prompt_token_ids"])
                  and now["logical_blocks"] == seq["logical_blocks"])
        elif chunks.get(sid, 0) > 0:
            # A prompt-completing chunk's sample is discarded (inherited).
            done = seq["prompt_tokens_processed"] + chunks[sid]
            ok = (now["prompt_tokens_processed"] == done
                  and now["prompt_processing_finished"] == (done == seq["prompt_len"])
                  and now["output_token_ids"] == seq["output_token_ids"]
                  and outputs[sid]["token_ids"] == seq["output_token_ids"])
        elif sid in chunks:
            ok = (now["output_token_ids"]
                  == seq["output_token_ids"] + [sampled[sid]]
                  and outputs[sid]["token_ids"] == now["output_token_ids"]
                  and outputs[sid]["prompt_token_ids"] == seq["prompt_token_ids"]
                  and not outputs[sid]["finished"]
                  and now["prompt_tokens_processed"] == seq["prompt_tokens_processed"])
        else:
            ok = now == seq
        check(ok, f"{tag}: request {sid} progress {seq} -> {now}")
    return finished


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def _accounting(c):
    """Driver-side cumulative accounting (no production history ledger).

    ``generated_token_events`` counts emitted decodes cumulatively; the
    native output counter resets on recomputation, so ``overgeneration`` is
    events minus ``max_tokens`` at completion. ``discarded_prompt_tokens`` is
    prompt progress lost at preemption and ``generated_tokens_moved_to_context``
    the outputs a victim had. Executed prefill tokens are split by prompt
    position: ``first_time`` (original prompt, never prefilled before),
    ``replayed`` (prefilled in an earlier attempt), and ``expanded_context``
    (a generated token moved into the prompt, prefilled for the first time;
    its cache entry may already have been computed by an earlier decode)."""
    return {n: dict(original_prompt_len=len(c["prompts"][n]), first_prefill=None,
                    first_decode=None, completion=None,
                    generated_token_events=0, native_output_count=0,
                    overgeneration=None, preemptions=0,
                    preemptions_after_output=0, discarded_prompt_tokens=0,
                    generated_tokens_moved_to_context=0, prefill_actions=0,
                    prefill_tokens_executed=0, prefill_tokens_first_time=0,
                    prefill_tokens_replayed=0,
                    prefill_tokens_expanded_context=0,
                    prefilled_position_high_water=0, omitted_decisions=[])
            for n in c["names"].values()}


def _update_accounting(acc, c, index, before, after, victims, layout, finished):
    names = c["names"]
    served = {sid for sid, _, _, _ in layout}
    for sid in victims:
        seq, a = before["seqs"][str(sid)], acc[names[sid]]
        a["preemptions"] += 1
        a["preemptions_after_output"] += bool(seq["output_token_ids"])
        a["discarded_prompt_tokens"] += seq["prompt_tokens_processed"]
        a["generated_tokens_moved_to_context"] += len(seq["output_token_ids"])
    for sid, _, chunk, _ in layout:
        a = acc[names[sid]]
        if chunk > 0:
            start = before["seqs"][str(sid)]["prompt_tokens_processed"]
            end, high = start + chunk, a["prefilled_position_high_water"]
            fresh_start = max(start, high)
            a["prefill_actions"] += 1
            a["prefill_tokens_executed"] += chunk
            a["prefill_tokens_replayed"] += max(0, min(end, high) - start)
            original = a["original_prompt_len"]
            a["prefill_tokens_first_time"] += max(
                0, min(end, original) - fresh_start)
            a["prefill_tokens_expanded_context"] += max(
                0, end - max(fresh_start, original))
            a["prefilled_position_high_water"] = max(high, end)
            a["first_prefill"] = (index if a["first_prefill"] is None
                                  else a["first_prefill"])
        else:
            a["generated_token_events"] += 1
            a["first_decode"] = index if a["first_decode"] is None else a["first_decode"]
    for sid in before["waiting"] + before["running"]:
        if sid not in served and sid not in victims:
            acc[names[sid]]["omitted_decisions"].append(index)
        a = acc[names[sid]]
        if sid in finished:
            a["completion"] = index
            a["native_output_count"] = max_tokens(c, names[sid])
            a["overgeneration"] = (a["generated_token_events"]
                                   - max_tokens(c, names[sid]))
        else:
            a["native_output_count"] = len(after["seqs"][str(sid)]["output_token_ids"])


def victim_need(policy, before, victims, layout, resident_limit, record):
    """Why victims were or were not required, relative to this decision's
    selected batch and the policy's own contract (diagnostic, not a check).

    Without victims, the selected decodes need ``demands`` free blocks and
    the selected admissions ``charges`` more; the LP decode prerequisite
    additionally needs one free block before any decode (zero-demand decodes
    included). Reasons: ``decode_block_demand`` (the decodes alone exceed the
    free blocks), ``admission_charges`` (admitted waiting requests use the
    recovered blocks, i.e. replacement), ``lp_decode_prerequisite`` (blocks
    needed only for H >= 1), ``resident_capacity``, or
    ``not_required_by_selected_batch``. ``non_preemptive_actions`` lists the
    single actions the policy's contract allows with no victim; when it is
    empty, every useful batch under that contract preempts."""
    free, running = before["free_blocks"], before["running"]
    charges = sum(d for _, k, _, d in layout if k == ADMISSION)
    demands = sum(d for _, k, _, d in layout if k == DECODE)
    decodes = sum(k == DECODE for _, k, _, _ in layout)
    admissions = sum(k == ADMISSION for _, k, _, _ in layout)
    lp = policy != "adapted_vllm"
    for_decodes = max(0, demands - free)
    physical = max(0, charges + demands - free)
    prerequisite = (max(0, charges + max(demands, 1 if decodes else 0) - free)
                    if lp else None)
    residents = max(0, len(running) + admissions - resident_limit)
    reasons = []
    if for_decodes:
        reasons.append("decode_block_demand")
    if physical > for_decodes:
        reasons.append("admission_charges")
    if lp and prerequisite > physical:
        reasons.append("lp_decode_prerequisite")
    if residents:
        reasons.append("resident_capacity")
    if victims and not reasons:
        reasons.append("not_required_by_selected_batch")
    alone = []
    for key, seq in before["seqs"].items():
        sid = int(key)
        if sid in running and not seq["prompt_processing_finished"]:
            alone.append([sid, PREFILL])
        elif sid in running:
            if _gap(before, sid) <= free and (not lp or free >= 1):
                alone.append([sid, DECODE])
        elif (len(running) < resident_limit
              and seq["logical_blocks"] <= free - before["watermark_blocks"]):
            alone.append([sid, ADMISSION])
    return dict(
        free_blocks=free, victims=list(victims),
        recovered_blocks=sum(len(before["block_tables"][str(v)])
                             for v in victims),
        selected_admission_charges=charges, selected_decode_demands=demands,
        selected_decodes=decodes,
        decode_demands={str(s): _gap(before, s) for s in running
                        if before["seqs"][str(s)]["prompt_processing_finished"]},
        victim_blocks_needed_for_decodes=for_decodes,
        victim_blocks_needed_physical=physical,
        victim_blocks_needed_with_lp_prerequisite=prerequisite,
        victims_needed_for_resident_limit=residents,
        reasons=reasons if victims else [],
        non_preemptive_actions=alone,
        preemption_free_alternative=bool(alone),
        extraction_changed_relaxed=(
            record["lp_comparison"]["extraction_changed_relaxed"]
            if record.get("lp_comparison") else None))


def run_comparison(policy, resident_limit, runtime, trace, case=TWO_REQUEST):
    """Submit the workload, observe at most the case's bound of nonempty decisions,
    and check each against its policy's contract, filling ``trace`` as it
    goes so a failure keeps the partial record. Returns the workload outcome
    (``drained`` or ``bounded_unfinished``); raises on any failure."""
    from sarathi.core.datatypes.sampling_params import SamplingParams

    c = case_inputs(case)
    names, pool = c["names"], c["num_gpu_blocks"]
    quiescent = dict(num_gpu_blocks=pool, max_num_seqs=resident_limit)
    scheduler = runtime.scheduler
    config = scheduler.scheduler_config
    expected_type = ("AdaptedVLLMScheduler" if policy == "adapted_vllm"
                     else "LPScheduler")
    check(policy in CPU_POLICIES, f"unknown policy {policy}")
    check(type(scheduler).__name__ == expected_type,
          f"registered scheduler is {type(scheduler).__name__}")
    limits = (config.max_num_seqs, config.max_model_len, config.num_pipeline_stages,
              config.max_num_batched_tokens)
    check(limits == (resident_limit, c["max_model_len"], 1, c["token_budget"]),
          f"scheduler configuration {limits}")
    if policy != "adapted_vllm":
        check((config.b_max, config.c_max, config.s_max)
              == (c["b_max"], c["c_max"], c["s_max"])
              and to_jsonable(config.numerical_policy) == NUMERICAL_POLICY_FIELDS
              and (config.decode_utility, config.prefill_token_utility,
                   config.preemption_penalty) == tuple(UNIFORM.values()),
              "LP scheduler configuration")
    manager = scheduler.block_manager
    check(manager.num_total_gpu_blocks == pool
          and manager.get_num_free_gpu_blocks() == pool
          and manager.watermark_blocks == int(0.01 * pool)
          and not manager.block_tables and manager.block_size == BLOCK_SIZE,
          "central pool before requests")

    select_coefficient_policy(scheduler, policy, case)
    observer = DecisionObserver(scheduler, runtime.seq_map, policy)
    trace.update(case=case, policy=policy, resident_limit=resident_limit,
                 prompts={}, arrival_times={}, virtual_arrivals=c["arrivals"],
                 submitted_at_virtual_time={}, idle_intervals=[],
                 decisions=[], final=None, idle=None)
    acc = _accounting(c)
    pending = list(names.items())
    clock = 0  # virtual time: nonempty decisions executed (see case_inputs)

    def submit_arrived():
        if not (pending and c["arrivals"][pending[0][1]] <= clock):
            return
        while pending and c["arrivals"][pending[0][1]] <= clock:
            sid, name = pending.pop(0)
            runtime.submit(list(c["prompts"][name]), SamplingParams(
                temperature=TEMPERATURE, max_tokens=max_tokens(c, name),
                ignore_eos=IGNORE_EOS, stop=None))
            check(sid in runtime.seq_map and max(runtime.seq_map) == sid,
                  f"request {name} must be seq {sid}")
            trace["prompts"][name] = list(c["prompts"][name])
            trace["arrival_times"][name] = runtime.seq_map[sid].arrival_time
            trace["submitted_at_virtual_time"][name] = clock
        runtime.worker_state()  # drop registration-time worker operations

    try:
        submit_arrived()
        for index in range(c["max_decisions"]):
            if not scheduler.has_unfinished_seqs() and pending:
                # Idle until the next arrival: no scheduler call, no work.
                start, clock = clock, c["arrivals"][pending[0][1]]
                trace["idle_intervals"].append([start, clock])
                submit_arrived()
            if not scheduler.has_unfinished_seqs():
                break
            tag = f"{case} {policy} limit {resident_limit} decision {index}"
            before = central_state(scheduler, runtime.seq_map)
            before["arrival_times"] = {str(s): runtime.seq_map[s].arrival_time
                                       for s in runtime.seq_map}
            worker_before = runtime.worker_state()
            check_quiescent(f"{tag} (before)", quiescent, before, worker_before)
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
                    tag, before, record, resident_limit, c)
            else:
                victims, layout = check_lp_decision(
                    tag, before, record, policy, resident_limit, c)
            finished = check_execution(tag, before, record, victims, layout,
                                       resident_limit, c)
            record["victim_need"] = victim_need(policy, before, victims, layout,
                                                resident_limit, record)
            # Recorded, not required: physical IDs may differ between managers.
            record["physical_block_ids_equal"] = (
                record["after"]["block_tables"]
                == record["worker_after"]["block_tables"])
            _update_accounting(acc, c, index, before, record["after"], victims,
                               layout, finished)
            record["accounting"] = json.loads(json.dumps(acc))
            print(f"{tag}: preempted {victims} emitted "
                  f"{record['schedules'][0]['outputs']['scheduled']} free "
                  f"{record['after']['free_blocks']}/"
                  f"{record['worker_after']['free_blocks']}", flush=True)
            record["virtual_time"] = clock
            clock += 1
            submit_arrived()

        final = central_state(scheduler, runtime.seq_map)
        worker_final = runtime.worker_state()
        unfinished = sorted(names[int(k)] for k in final["seqs"])
        trace["final"] = dict(
            central=final, worker=worker_final, unfinished=unfinished,
            residual_allocations={names[int(k)]: len(v) for k, v in
                                  final["block_tables"].items()},
            accounting=acc)
        trace["final"]["unsubmitted"] = [name for _, name in pending]
        if scheduler.has_unfinished_seqs() or pending:
            # Bounded: retain the final native state; nothing is forced.
            check((unfinished or pending)
                  and len(trace["decisions"]) == c["max_decisions"],
                  "unfinished work without reaching the decision bound")
            check_quiescent("final", quiescent, final, worker_final)
            return BOUNDED_UNFINISHED

        # Drained: full release, then one ordinary idle call.
        check(final["waiting"] == [] and final["running"] == []
              and final["seqs"] == {} and final["block_tables"] == {}
              and final["free_blocks"] == pool
              and worker_final["seqs"] == {} and worker_final["block_tables"] == {}
              and worker_final["free_blocks"] == pool
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
    "scripts/exact_integer_plan.py", "scripts/compare_coefficient_policies_cpu.py",
    "scripts/check_lp_scheduler_gpu.py",
    "scripts/check_lp_scheduler_preemption_gpu.py",
    "tests/test_compare_scheduler_gpu.py", "tests/test_exact_integer_plan.py",
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
    parser.add_argument("--case", default=TWO_REQUEST, choices=GPU_CASES)
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--resident-limit", type=int, choices=(2, 1, 4, 8),
                        help="two_request: 2 or 1 (required); "
                        "memory_pressure: 4 (default); "
                        "queue48_burst_pool12: 8 (default)")
    parser.add_argument("--model-path", required=True,
                        help="existing local directory with the TinyLlama "
                        "config and tokenizer")
    parser.add_argument("--output-dir", required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    c = case_inputs(args.case)
    if args.resident_limit is None and len(c["resident_limits"]) == 1:
        args.resident_limit = c["resident_limits"][0]
    if args.resident_limit not in c["resident_limits"]:
        parser.error(f"--resident-limit for case {args.case} must be one of "
                     f"{list(c['resident_limits'])}")
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
                   run=run_parameters(args.policy, args.resident_limit,
                                      args.case),
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
                                  out_dir / "metrics_store_unused", args.case)
        configs = args_record.create_engine_configs()
        model_config, cache_config, parallel_config, scheduler_config, _ = configs
        check((parallel_config.tensor_parallel_size,
               parallel_config.pipeline_parallel_size)
              == (TENSOR_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE)
              and cache_config.block_size == BLOCK_SIZE
              and model_config.max_model_len == c["max_model_len"]
              and scheduler_config.max_num_seqs == args.resident_limit,
              "native configuration differs from the run")

        summary["environment"] = environment_record()
        print("environment:", json.dumps(summary["environment"], indent=1), flush=True)
        register_worker_helpers_by_value()
        selection = CacheCapacitySelection(chosen=c["num_gpu_blocks"])
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
        adapted.check_capacity(dict(num_gpu_blocks=c["num_gpu_blocks"]),
                               selection, cache_config,
                               engine.scheduler.block_manager, worker_init)
        for name, ids in c["prompts"].items():
            check(select_prompt_ids(engine.tokenizer, runtime.vocab_size, ids[0],
                                    len(ids))[0] == ids,
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

        if args.policy == "adapted_vllm" and args.case in QUEUE_GROWTH_CASES:
            # Read-only diagnostic problems for the offline CPU comparison.
            trace["reference_diagnostic_problems"] = (
                capture_reference_problems(runtime, args.case))
        outcome = run_comparison(args.policy, args.resident_limit, runtime,
                                 trace, args.case)
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
