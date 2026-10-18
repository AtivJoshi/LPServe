"""Bounded dummy-weight GPU validation of the adapted vLLM reference scheduler.

Runs one planned case through a real ``BaseLLMEngine`` built from
``EngineArgs(scheduler_type="adapted_vllm")`` with the registered
``AdaptedVLLMScheduler``, one GPU worker (tensor parallel 1, one pipeline
stage), dummy TinyLlama weights, greedy sampling with EOS ignored, and native
replay and completion. Every nonempty decision is checked against the exact
expected proposal, emitted output, native block operations, and state.

Cases (one per process):

- ``mixed``: A (5 prompt tokens) and B (8), two output tokens each, budget 8,
  resident limit 2, 8 blocks of 16, ``max_model_len`` 32. Selection A:5/B:3,
  then A's decode and B's remaining 5-token prefill emitted prompt-first.
- ``preemption``: four 16-token requests, three output tokens each, budget
  64, resident limit 4, 4 blocks of 16, ``max_model_len`` 64. The prefills
  fill the pool, the zero-demand decodes create the next block demand, the
  third decision evicts tail requests 3 then 2, and both are readmitted with
  their 17-token expanded contexts after 0 and 1 finish.

Cache sizing: worker profiling runs once, unchanged; its result is recorded
and must be at least the case pool, which is then supplied to native cache
initialization (worker cache and block manager; the central scheduler uses
the same cache configuration). A validation worker subclass records its own
block-manager allocate/free/append calls (zero-demand appends included) and
exposes primitive state; validation wrappers on the central scheduler and
block manager record the proposal and native calls. All of them delegate
unchanged.

Not established: physical-ID equality between managers, cache-content or
attention correctness, sampler generality, generation quality, arbitrary
workload completion, or performance. The decision bound
(``MAX_NONEMPTY_DECISIONS``) and the process timeout are validation guards,
not scheduler restart caps or completion guarantees. Restarted requests can
produce more generated-token events than ``max_tokens`` (inherited output
counter reset); the counter is not cumulative generation.

Run from the repository root with the root on PYTHONPATH, one case per
process, on an allocated GPU node:

    PYTHONPATH="$PWD" timeout 600s python -B scripts/check_adapted_vllm_gpu.py \\
        --case mixed --model-path <local TinyLlama snapshot directory> \\
        --output-dir validation_output/adapted_vllm_gpu/<UTC timestamp>/mixed

Exit status is zero only if every check passes.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Read-only helpers from the earlier bounded checks; their main() is unused.
import check_lp_scheduler_gpu as single
import check_lp_scheduler_preemption_gpu as preemption
from check_lp_scheduler_gpu import (CheckFailure, check, environment_record,
                                    outputs_record, request_output_record,
                                    to_jsonable)
from check_lp_scheduler_preemption_gpu import (CacheCapacitySelection,
                                               check_pool, git,
                                               select_prompt_ids, sha256)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Approved scoped test inputs (provisional, not project-wide policy).
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "dummy"
DTYPE = "float16"
ATTENTION_BACKEND = "flash_attention"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
BLOCK_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.5
SEED = 42
TRUST_REMOTE_CODE = True
TOKENIZER_MODE = "auto"
TEMPERATURE = 0.0
IGNORE_EOS = True
SCHEDULER_TYPE = "adapted_vllm"
MAX_NONEMPTY_DECISIONS = 12  # validation guard per case
PROCESS_TIMEOUT_SECONDS = 600  # documented `timeout` for one case process
PROMPT_ID_SEARCH_STARTS = (1000, 2000, 3000, 4000)
PROFILE_METHOD = preemption.PROFILE_METHOD
INIT_CACHE_METHOD = preemption.INIT_CACHE_METHOD
WORKER_STATE_METHOD = "validation_state"

ADMISSION, PREFILL, DECODE = "admission", "resident_prefill", "decode"


def _decision(selected, scheduled, central, worker, after, finished=(),
              victims=(), stops=("running_exhausted",
                                 "arrived_waiting_exhausted"),
              completion=()):
    """One expected nonempty decision.

    ``selected``: (seq_id, kind, tokens, block_demand) in selection order.
    ``scheduled``: emitted [seq_id, prompt_chunk_len]. Block operations are
    (op, seq_id, table length before, after): ``central`` during scheduling,
    ``completion`` central frees at completion, ``worker`` the whole step.
    ``after``: (waiting, running, free blocks, {seq_id: (prompt_len,
    processed, prompt_done, outputs, logical_blocks, physical_blocks)}) for
    unfinished requests at the completed step boundary."""
    waiting, running, free, seqs = after
    return dict(
        selected=[list(s) for s in selected], victims=list(victims),
        stops=list(stops), preempted=list(victims),
        scheduled=[list(s) for s in scheduled],
        central=[list(o) for o in central],
        completion=[list(o) for o in completion],
        worker=[list(o) for o in worker],
        after=dict(waiting=list(waiting), running=list(running), free=free,
                   seqs={str(k): list(v) for k, v in seqs.items()}),
        finished=list(finished),
    )


def _al(sid, after):
    return ("allocate", sid, 0, after)


def _ap(sid, before, after=None):
    return ("append", sid, before, before if after is None else after)


def _fr(sid, before):
    return ("free", sid, before, 0)


CASES = dict(
    mixed=dict(
        prompt_lens=[5, 8], max_tokens=2, max_num_batched_tokens=8,
        max_num_seqs=2, max_model_len=32, num_gpu_blocks=8,
        token_events={0: 2, 1: 2},
        decisions=[
            _decision([(0, ADMISSION, 5, 1), (1, ADMISSION, 3, 1)],
                      [(0, 5), (1, 3)], [_al(0, 1), _al(1, 1)],
                      [_al(0, 1), _al(1, 1)],
                      ([], [0, 1], 6, {0: (5, 5, True, 0, 1, 1),
                                       1: (8, 3, False, 0, 1, 1)})),
            # Selection A decode, B prefill; emitted prompt-first.
            _decision([(0, DECODE, 1, 0), (1, PREFILL, 5, 0)],
                      [(1, 5), (0, 0)], [_ap(0, 1)],
                      [_ap(1, 1), _ap(0, 1)],
                      ([], [0, 1], 6, {0: (5, 5, True, 1, 1, 1),
                                       1: (8, 8, True, 0, 1, 1)})),
            _decision([(0, DECODE, 1, 0), (1, DECODE, 1, 0)],
                      [(0, 0), (1, 0)], [_ap(0, 1), _ap(1, 1)],
                      [_ap(0, 1), _ap(1, 1), _fr(0, 1)],
                      ([], [1], 7, {1: (8, 8, True, 1, 1, 1)}),
                      finished=[0], completion=[_fr(0, 1)]),
            _decision([(1, DECODE, 1, 0)], [(1, 0)], [_ap(1, 1)],
                      [_ap(1, 1), _fr(1, 1)], ([], [], 8, {}),
                      finished=[1], completion=[_fr(1, 1)]),
        ],
    ),
    preemption=dict(
        prompt_lens=[16, 16, 16, 16], max_tokens=3, max_num_batched_tokens=64,
        max_num_seqs=4, max_model_len=64, num_gpu_blocks=4,
        # Restarted 2 and 3: one event before preemption plus three after,
        # against max_tokens 3 (inherited counter reset, not repaired).
        token_events={0: 3, 1: 3, 2: 4, 3: 4},
        decisions=[
            _decision([(i, ADMISSION, 16, 1) for i in range(4)],
                      [(i, 16) for i in range(4)],
                      [_al(i, 1) for i in range(4)],
                      [_al(i, 1) for i in range(4)],
                      ([], [0, 1, 2, 3], 0,
                       {i: (16, 16, True, 0, 1, 1) for i in range(4)})),
            # Zero-demand decodes with no free block; each gap becomes 1.
            _decision([(i, DECODE, 1, 0) for i in range(4)],
                      [(i, 0) for i in range(4)],
                      [_ap(i, 1) for i in range(4)],
                      [_ap(i, 1) for i in range(4)],
                      ([], [0, 1, 2, 3], 0,
                       {i: (16, 16, True, 1, 2, 1) for i in range(4)})),
            # 0 needs a block: evict tail 3; 1 needs one: evict tail 2.
            _decision([(0, DECODE, 1, 1), (1, DECODE, 1, 1)],
                      [(0, 0), (1, 0)],
                      [_fr(3, 1), _fr(2, 1), _ap(0, 1, 2), _ap(1, 1, 2)],
                      [_fr(3, 1), _fr(2, 1), _ap(0, 1, 2), _ap(1, 1, 2)],
                      ([2, 3], [0, 1], 0,
                       {0: (16, 16, True, 2, 2, 2), 1: (16, 16, True, 2, 2, 2),
                        2: (17, 0, False, 0, 2, 0), 3: (17, 0, False, 0, 2, 0)}),
                      victims=[3, 2],
                      stops=("running_exhausted", "skipped_after_preemption")),
            # The 2-block head does not fit 0 free blocks; 0 and 1 finish.
            _decision([(0, DECODE, 1, 0), (1, DECODE, 1, 0)],
                      [(0, 0), (1, 0)], [_ap(0, 2), _ap(1, 2)],
                      [_ap(0, 2), _ap(1, 2), _fr(0, 2), _fr(1, 2)],
                      ([2, 3], [], 4, {2: (17, 0, False, 0, 2, 0),
                                       3: (17, 0, False, 0, 2, 0)}),
                      finished=[0, 1], completion=[_fr(0, 2), _fr(1, 2)],
                      stops=("running_exhausted", "admission_memory")),
            # Readmission of the expanded contexts, recomputed from token 0.
            _decision([(2, ADMISSION, 17, 2), (3, ADMISSION, 17, 2)],
                      [(2, 17), (3, 17)], [_al(2, 2), _al(3, 2)],
                      [_al(2, 2), _al(3, 2)],
                      ([], [2, 3], 0, {2: (17, 17, True, 0, 2, 2),
                                       3: (17, 17, True, 0, 2, 2)})),
            _decision([(2, DECODE, 1, 0), (3, DECODE, 1, 0)],
                      [(2, 0), (3, 0)], [_ap(2, 2), _ap(3, 2)],
                      [_ap(2, 2), _ap(3, 2)],
                      ([], [2, 3], 0, {2: (17, 17, True, 1, 2, 2),
                                       3: (17, 17, True, 1, 2, 2)})),
            _decision([(2, DECODE, 1, 0), (3, DECODE, 1, 0)],
                      [(2, 0), (3, 0)], [_ap(2, 2), _ap(3, 2)],
                      [_ap(2, 2), _ap(3, 2)],
                      ([], [2, 3], 0, {2: (17, 17, True, 2, 2, 2),
                                       3: (17, 17, True, 2, 2, 2)})),
            _decision([(2, DECODE, 1, 0), (3, DECODE, 1, 0)],
                      [(2, 0), (3, 0)], [_ap(2, 2), _ap(3, 2)],
                      [_ap(2, 2), _ap(3, 2), _fr(2, 2), _fr(3, 2)],
                      ([], [], 4, {}), finished=[2, 3],
                      completion=[_fr(2, 2), _fr(3, 2)]),
        ],
    ),
)


def validate_case_constants(name):
    """Static consistency of a case with the engine and policy limits."""
    case = CASES[name]
    pool, max_len = case["num_gpu_blocks"], case["max_model_len"]
    check(math.ceil(max_len / BLOCK_SIZE) <= pool,
          f"{name}: pool {pool} fails the engine maximum-context check")
    check(int(0.01 * pool) == 0, f"{name}: expected a zero native watermark")
    check(case["max_num_batched_tokens"] >= case["max_num_seqs"],
          f"{name}: budget below resident limit")
    check(len(case["decisions"]) <= MAX_NONEMPTY_DECISIONS,
          f"{name}: plan exceeds the decision bound")
    for n, length in enumerate(case["prompt_lens"]):
        events = case["token_events"][n]
        check(0 < length and length + events <= max_len,
              f"{name}: request {n} context exceeds max_model_len")
    return case


# ---------------------------------------------------------------------------
# Observation. These functions use only builtins: the worker copies are
# serialized by value into the Ray worker.
# ---------------------------------------------------------------------------


def instrument_block_manager(manager, ops):
    """Record allocate/free/append_slot calls as [op, seq_id, table before,
    table after] (physical block IDs) and delegate once, unchanged."""
    def table(seq):
        return [b.block_number for b in manager.block_tables.get(seq.seq_id, [])]

    def wrap(name, label):
        native = getattr(manager, name)

        def wrapper(seq):
            before = table(seq)
            result = native(seq)
            ops.append([label, seq.seq_id, before, table(seq)])
            return result
        setattr(manager, name, wrapper)

    wrap("allocate", "allocate")
    wrap("free", "free")
    wrap("append_slot", "append")


def seq_state(seq):
    return dict(
        status=seq.get_status().name,
        prompt_len=seq.get_prompt_len(),
        prompt_token_ids=list(seq.prompt_token_ids),
        prompt_tokens_processed=seq.get_num_prompt_tokens_processed(),
        prompt_processing_finished=seq.prompt_processing_finished,
        output_token_ids=list(seq.output_token_ids),
        logical_blocks=len(seq.logical_token_blocks),
    )


def manager_state(seq_map, manager):
    return dict(
        total_blocks=manager.num_total_gpu_blocks,
        watermark_blocks=manager.watermark_blocks,
        free_blocks=manager.get_num_free_gpu_blocks(),
        free_block_ids=[b.block_number for b in manager.gpu_allocator.free_blocks],
        block_tables={str(k): [b.block_number for b in v]
                      for k, v in manager.block_tables.items()},
        seqs={str(k): seq_state(s) for k, s in seq_map.items()},
    )


def worker_validation_state(seq_manager, ops):
    """Worker block-manager state and the operations since the last call."""
    state = manager_state(seq_manager.seq_map, seq_manager.block_manager)
    state["block_ops"] = list(ops)
    del ops[:]
    return state


def central_state(scheduler, seq_map):
    state = manager_state(seq_map, scheduler.block_manager)
    state.update(iteration_id=scheduler._iteration_id,
                 num_running_batches=scheduler.num_running_batches,
                 waiting=[s.seq_id for s in scheduler.waiting],
                 running=[s.seq_id for s in scheduler.running])
    return state


class CentralObserver:
    """Validation-only wrappers on the central scheduler instance: the
    proposal, the native block operations (split at the end of scheduling),
    and the emitted output with the state at that point."""

    def __init__(self, scheduler, seq_map):
        self.ops = []
        self.proposals = []
        self.schedules = []
        instrument_block_manager(scheduler.block_manager, self.ops)
        native_propose, native_schedule = scheduler._propose, scheduler.schedule

        def propose(now):
            proposal = native_propose(now)
            self.proposals.append(to_jsonable(proposal))
            return proposal

        def schedule():
            outputs = native_schedule()
            self.schedules.append(dict(
                outputs=outputs_record(outputs), num_ops=len(self.ops),
                state=central_state(scheduler, seq_map)))
            return outputs

        scheduler._propose, scheduler.schedule = propose, schedule

    def clear(self):
        del self.ops[:], self.proposals[:], self.schedules[:]


def _lens(ops):
    return [[op, sid, len(before), len(after)] for op, sid, before, after in ops]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_quiescent(tag, case, central, worker):
    """Ownership, per-manager conservation, and central/worker agreement at
    a completed step boundary. Physical block IDs may differ."""
    pool = case["num_gpu_blocks"]
    waiting, running = central["waiting"], central["running"]
    check(len(set(waiting + running)) == len(waiting) + len(running),
          f"{tag}: ownership not unique/disjoint: {waiting} {running}")
    check(len(running) <= case["max_num_seqs"], f"{tag}: residents {running}")
    check(central["num_running_batches"] == 0, f"{tag}: batches in flight")
    check(sorted(waiting + running) == sorted(int(k) for k in central["seqs"]),
          f"{tag}: owned set differs from the central sequence map")
    check(set(central["seqs"]) == set(worker["seqs"]),
          f"{tag}: central/worker sequence maps differ")
    check(set(central["block_tables"]) == {str(k) for k in running},
          f"{tag}: central tables {central['block_tables']} vs running")
    check({k: len(v) for k, v in central["block_tables"].items()}
          == {k: len(v) for k, v in worker["block_tables"].items()},
          f"{tag}: central/worker allocation counts differ")
    check(central["free_blocks"] == worker["free_blocks"],
          f"{tag}: free counts {central['free_blocks']} {worker['free_blocks']}")
    check_pool(tag, "central", pool, central["block_tables"],
               central["free_block_ids"])
    check_pool(tag, "worker", pool, worker["block_tables"],
               worker["free_block_ids"])
    for key, seq in central["seqs"].items():
        check(seq == worker["seqs"][key],
              f"{tag}: seq {key} central {seq} worker {worker['seqs'][key]}")
        status = "WAITING" if int(key) in waiting else "PAUSED"
        check(seq["status"] == status, f"{tag}: seq {key} status {seq}")


def check_decision(tag, case, expected, before, worker_before, record):
    """Exact expected proposal, output, operations, and resulting state."""
    proposal_list, schedules = record["proposals"], record["schedules"]
    check(len(proposal_list) == 1 and len(schedules) == 1,
          f"{tag}: {len(proposal_list)} proposals, {len(schedules)} schedules")
    proposal, scheduled = proposal_list[0], schedules[0]
    emitted = scheduled["outputs"]

    # Proposal: observed inputs, selection order, victims, stop reasons.
    check(proposal["running_order"] == before["running"]
          and proposal["arrived_waiting_order"] == before["waiting"]
          and proposal["free_blocks"] == before["free_blocks"]
          and proposal["watermark_blocks"] == 0
          and proposal["token_budget"] == case["max_num_batched_tokens"],
          f"{tag}: proposal inputs {proposal}")
    selected = [[a["seq_id"], a["kind"], a["num_tokens"], a["block_demand"]]
                for a in proposal["actions"]]
    check(selected == expected["selected"],
          f"{tag}: selected {selected}; expected {expected['selected']}")
    check(list(proposal["victims"]) == expected["victims"],
          f"{tag}: victims {proposal['victims']}")
    stops = [proposal["running_stop_reason"], proposal["waiting_stop_reason"]]
    check(stops == expected["stops"], f"{tag}: stop reasons {stops}")
    check(sum(a[2] for a in selected) <= case["max_num_batched_tokens"],
          f"{tag}: selected tokens exceed the budget")

    # Emitted: victims first, then the stable prompt-first partition of the
    # selection order; and the exact expectation.
    layout = ([[a[0], a[2]] for a in selected if a[1] != DECODE]
              + [[a[0], 0] for a in selected if a[1] == DECODE])
    check(emitted["preempted_seq_ids"] == list(proposal["victims"])
          and emitted["scheduled"] == layout and not emitted["ignored_seq_ids"],
          f"{tag}: emitted {emitted} is not the layout of the proposal")
    check(emitted["preempted_seq_ids"] == expected["preempted"]
          and emitted["scheduled"] == expected["scheduled"],
          f"{tag}: emitted {emitted}; expected {expected['preempted']} "
          f"{expected['scheduled']}")
    check(emitted["id"] == before["iteration_id"] + 1, f"{tag}: output id")

    # Central native operations: scheduling, then completion frees.
    ops = record["central_ops"]
    sched_ops, completion_ops = ops[:scheduled["num_ops"]], ops[scheduled["num_ops"]:]
    check(_lens(sched_ops) == expected["central"],
          f"{tag}: central scheduling ops {_lens(sched_ops)}; expected "
          f"{expected['central']}")
    check(_lens(completion_ops) == expected["completion"],
          f"{tag}: central completion ops {_lens(completion_ops)}")
    for op, sid, table_before, _ in sched_ops:
        prior = before["block_tables"].get(str(sid), [])
        check(table_before == prior,
              f"{tag}: {op} {sid} saw table {table_before}, expected {prior}")
    worker_ops = record["worker_after"]["block_ops"]
    check(_lens(worker_ops) == expected["worker"],
          f"{tag}: worker ops {_lens(worker_ops)}; expected {expected['worker']}")

    # At the end of scheduling: progress unchanged until completion, the
    # free count equals the proposal arithmetic, a batch is in flight.
    sched_state = scheduled["state"]
    victim_blocks = sum(len(before["block_tables"][str(v)])
                        for v in proposal["victims"])
    check(sched_state["free_blocks"] == before["free_blocks"] + victim_blocks
          - sum(a[3] for a in selected),
          f"{tag}: free blocks after scheduling {sched_state['free_blocks']}")
    check(sched_state["seqs"] == before["seqs"],
          f"{tag}: sequence progress changed before completion")
    check(sched_state["num_running_batches"] == 1, f"{tag}: batch count")

    # Completed step boundary.
    after, worker_after = record["after"], record["worker_after"]
    check(after["iteration_id"] == before["iteration_id"] + 1,
          f"{tag}: iteration advanced "
          f"{after['iteration_id'] - before['iteration_id']}")
    check_quiescent(f"{tag} (after)", case, after, worker_after)
    want = expected["after"]
    check(after["waiting"] == want["waiting"] and after["running"] == want["running"]
          and after["free_blocks"] == want["free"],
          f"{tag}: queues/free after {after['waiting']} {after['running']} "
          f"{after['free_blocks']}; expected {want}")
    got = {k: [s["prompt_len"], s["prompt_tokens_processed"],
               s["prompt_processing_finished"], len(s["output_token_ids"]),
               s["logical_blocks"], len(after["block_tables"].get(k, []))]
           for k, s in after["seqs"].items()}
    check(got == want["seqs"], f"{tag}: request state {got}; expected "
          f"{want['seqs']}")

    # Samples are associated with the scheduled entries in order.
    sampler = record["sampler_outputs"]
    check(sampler is not None
          and [s for s, _ in sampler] == [s for s, _ in emitted["scheduled"]],
          f"{tag}: sampler outputs {sampler} vs scheduled {emitted['scheduled']}")
    finished = [o["seq_id"] for o in record["request_outputs"] if o["finished"]]
    check(finished == expected["finished"], f"{tag}: finished {finished}")
    check(sorted(o["seq_id"] for o in record["request_outputs"])
          == sorted(s for s, _ in emitted["scheduled"]),
          f"{tag}: request outputs {record['request_outputs']}")
    for o in record["request_outputs"]:
        if o["finished"]:
            check(o["finish_reason"] == "length"
                  and len(o["token_ids"]) == case["max_tokens"]
                  and str(o["seq_id"]) not in after["seqs"]
                  and str(o["seq_id"]) not in worker_after["seqs"]
                  and str(o["seq_id"]) not in after["block_tables"]
                  and str(o["seq_id"]) not in worker_after["block_tables"],
                  f"{tag}: finished output or release {o}")


def run_case(name, runtime, trace):
    """Submit the case's requests and check every decision and the final
    idle call, filling ``trace`` as it goes (so a failure keeps the partial
    record). ``runtime`` provides ``scheduler``, ``seq_map``,
    ``prompt_ids(lengths)``, ``submit(prompt_ids, sampling_params)``,
    ``step()`` returning (request outputs, sampler outputs as [seq_id, token]
    or None), and ``worker_state()``. Raises CheckFailure."""
    from sarathi.core.datatypes.sampling_params import SamplingParams
    from sarathi.core.scheduler.adapted_vllm_scheduler import (
        AdaptedVLLMScheduler)

    case = validate_case_constants(name)
    scheduler = runtime.scheduler
    config = scheduler.scheduler_config
    check(type(scheduler) is AdaptedVLLMScheduler,
          f"registered scheduler is {type(scheduler).__name__}")
    check((config.max_num_seqs, config.max_model_len,
           config.num_pipeline_stages, config.max_num_batched_tokens)
          == (case["max_num_seqs"], case["max_model_len"], 1,
              case["max_num_batched_tokens"]),
          f"scheduler configuration {vars(config)}")
    manager = scheduler.block_manager
    check(manager.num_total_gpu_blocks == case["num_gpu_blocks"]
          and manager.get_num_free_gpu_blocks() == case["num_gpu_blocks"]
          and manager.watermark_blocks == 0 and not manager.block_tables
          and manager.block_size == BLOCK_SIZE, "central pool before requests")

    observer = CentralObserver(scheduler, runtime.seq_map)
    trace.update(case=name, prompts=[], decisions=[], idle=None,
                 token_events=None)
    for n, prompt_ids in enumerate(runtime.prompt_ids(case["prompt_lens"])):
        check(len(prompt_ids) == case["prompt_lens"][n], f"prompt {n} length")
        runtime.submit(list(prompt_ids), SamplingParams(
            temperature=TEMPERATURE, max_tokens=case["max_tokens"],
            ignore_eos=IGNORE_EOS, stop=None))
        check(sorted(runtime.seq_map) == list(range(n + 1)),
              f"request {n} must be seq {n}")
        trace["prompts"].append(list(prompt_ids))
    arrivals = [s.arrival_time for s in scheduler.waiting]
    runtime.worker_state()  # drop registration-time worker operations

    events = {n: 0 for n in range(len(case["prompt_lens"]))}
    expected_steps = case["decisions"]
    for index in range(MAX_NONEMPTY_DECISIONS):
        if not scheduler.has_unfinished_seqs():
            break
        tag = f"{name} decision {index}"
        check(index < len(expected_steps),
              f"{tag}: more nonempty decisions than the {len(expected_steps)} "
              "planned")
        before = central_state(scheduler, runtime.seq_map)
        worker_before = runtime.worker_state()
        check_quiescent(f"{tag} (before)", case, before, worker_before)
        check(worker_before["block_ops"] == [],
              f"{tag}: worker operations outside a step")
        observer.clear()
        request_outputs, sampler_outputs = runtime.step()
        record = dict(
            index=index, before=before, worker_before=worker_before,
            proposals=list(observer.proposals),
            schedules=list(observer.schedules),
            central_ops=list(observer.ops),
            after=central_state(scheduler, runtime.seq_map),
            worker_after=runtime.worker_state(),
            sampler_outputs=sampler_outputs,
            request_outputs=[request_output_record(o) for o in request_outputs])
        trace["decisions"].append(record)
        if index == 0:
            check(all(a <= record["proposals"][0]["decision_time"]
                      for a in arrivals), f"{tag}: requests not yet arrived")
        check_decision(tag, case, expected_steps[index], before,
                       worker_before, record)
        for sid, chunk in record["schedules"][0]["outputs"]["scheduled"]:
            if chunk == 0:
                events[sid] += 1
        print(f"{tag}: preempted "
              f"{record['schedules'][0]['outputs']['preempted_seq_ids']} "
              f"scheduled {record['schedules'][0]['outputs']['scheduled']} "
              f"free {record['after']['free_blocks']}", flush=True)
    else:
        check(not scheduler.has_unfinished_seqs(),
              f"{name}: unfinished after {MAX_NONEMPTY_DECISIONS} decisions")
    check(len(trace["decisions"]) == len(expected_steps),
          f"{name}: {len(trace['decisions'])} decisions; planned "
          f"{len(expected_steps)}")
    trace["token_events"] = {str(k): v for k, v in events.items()}
    check(events == case["token_events"],
          f"{name}: generated-token events {events}; expected "
          f"{case['token_events']}")

    # Drained, then one ordinary idle call that changes nothing but the id.
    drained = central_state(scheduler, runtime.seq_map)
    worker_drained = runtime.worker_state()
    check(drained["waiting"] == [] and drained["running"] == []
          and drained["seqs"] == {} and drained["block_tables"] == {}
          and drained["free_blocks"] == case["num_gpu_blocks"]
          and worker_drained["seqs"] == {} and worker_drained["block_tables"] == {}
          and worker_drained["free_blocks"] == case["num_gpu_blocks"],
          f"{name}: not drained {drained} {worker_drained}")
    observer.clear()
    idle_outputs, idle_sampler = runtime.step()
    after = central_state(scheduler, runtime.seq_map)
    trace["idle"] = dict(before=drained, after=after,
                         proposals=list(observer.proposals),
                         schedules=list(observer.schedules),
                         central_ops=list(observer.ops),
                         worker_after=runtime.worker_state())
    check(list(idle_outputs) == [] and idle_sampler is None,
          f"{name}: idle step returned outputs")
    check(len(observer.schedules) == 1
          and observer.schedules[0]["outputs"]["scheduled"] == []
          and observer.schedules[0]["outputs"]["preempted_seq_ids"] == []
          and observer.ops == [] and trace["idle"]["worker_after"]["block_ops"] == []
          and len(observer.proposals) == 1
          and observer.proposals[0]["actions"] == []
          and observer.proposals[0]["victims"] == [],
          f"{name}: idle step was not an ordinary empty decision")
    check({**after, "iteration_id": None} == {**drained, "iteration_id": None}
          and after["iteration_id"] == drained["iteration_id"] + 1,
          f"{name}: idle step changed state")
    return trace


# ---------------------------------------------------------------------------
# GPU engine construction (lazy; nothing here runs at import or --help)
# ---------------------------------------------------------------------------


def engine_args(name, model_path, metrics_dir):
    """Native ``EngineArgs`` for the adapted reference and the case."""
    from sarathi.engine.arg_utils import EngineArgs

    case = CASES[name]
    return EngineArgs(
        model=str(model_path), tokenizer=str(model_path),
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        scheduler_type=SCHEDULER_TYPE, max_model_len=case["max_model_len"],
        max_num_seqs=case["max_num_seqs"],
        max_num_batched_tokens=case["max_num_batched_tokens"],
        # Enabled metrics with optional outputs off, as in the earlier
        # bounded checks (the disabled mode lacks attributes the engine
        # reads each step); nothing is plotted.
        write_metrics=True, output_dir=str(metrics_dir),
        enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
        enable_chrome_trace=False, enable_request_outputs=False,
        keep_individual_batch_metrics=True,
        attention_backend=ATTENTION_BACKEND,
    )


def build_worker_class():
    """``BaseWorker`` subclass that records its block-manager operations and
    exposes primitive state; native behavior is unchanged."""
    from sarathi.worker.base_worker import BaseWorker

    class ValidationWorker(BaseWorker):
        def init_cache_engine(self, cache_config):
            super().init_cache_engine(cache_config)
            self._validation_ops = []
            instrument_block_manager(self.seq_manager.block_manager,
                                     self._validation_ops)

        def validation_state(self):
            caches = [t for layer in self.gpu_cache for t in (
                layer if isinstance(layer, (tuple, list)) else (layer,))]
            state = worker_validation_state(self.seq_manager,
                                            self._validation_ops)
            state.update(
                cache_config_num_gpu_blocks=self.cache_config.num_gpu_blocks,
                cache_engine_num_gpu_blocks=self.cache_engine.num_gpu_blocks,
                cache_layers=len(self.gpu_cache),
                cache_tensor_block_dims=sorted({int(t.shape[0]) for t in caches}),
                cache_tensor_shapes=sorted({tuple(t.shape) for t in caches}),
                allocator_num_blocks=(
                    self.seq_manager.block_manager.gpu_allocator.num_blocks),
            )
            return state

    return ValidationWorker


def build_engine_class(selection):
    """``BaseLLMEngine`` subclass using the validation worker, passing the
    chosen capacity to native cache initialization, and recording sampler
    outputs. Every worker call is delegated unchanged."""
    from sarathi.engine.base_llm_engine import BaseLLMEngine

    class AdaptedReferenceValidationEngine(BaseLLMEngine):
        validation_sampler_outputs = None

        def _get_worker_impl(self):
            return build_worker_class()

        def _run_workers(self, method, *args, **kwargs):
            result = super()._run_workers(method, *args, **kwargs)
            if method == PROFILE_METHOD:
                return selection.select(result)
            if method == INIT_CACHE_METHOD:
                selection.record_initialization(
                    kwargs["cache_config"].num_gpu_blocks, result)
            if method == "execute_model":
                self.validation_sampler_outputs = (
                    None if result is None
                    else [[o.seq_id, o.output_token] for o in result])
            return result

    return AdaptedReferenceValidationEngine


class EngineRuntime:
    """``run_case`` runtime over the real engine and its single worker."""

    def __init__(self, engine, tokenizer, vocab_size):
        self.engine = engine
        self.scheduler = engine.scheduler
        self.seq_map = engine.seq_manager.seq_map
        self.tokenizer, self.vocab_size = tokenizer, vocab_size

    def prompt_ids(self, lengths):
        prompts = [select_prompt_ids(self.tokenizer, self.vocab_size, start, n)[0]
                   for start, n in zip(PROMPT_ID_SEARCH_STARTS, lengths)]
        flat = [t for p in prompts for t in p]
        check(len(flat) == len(set(flat)), "prompts must be disjoint")
        return prompts

    def submit(self, prompt_ids, sampling_params):
        self.engine.add_request(prompt=None, sampling_params=sampling_params,
                                prompt_token_ids=prompt_ids)

    def step(self):
        self.engine.validation_sampler_outputs = None
        outputs = self.engine.step()
        return outputs, self.engine.validation_sampler_outputs

    def worker_state(self):
        states = self.engine._run_workers(WORKER_STATE_METHOD,
                                          get_all_outputs=True)
        check(len(states) == 1, f"expected one worker, got {len(states)}")
        return states[0]


def check_capacity(case, selection, cache_config, manager, worker):
    n = case["num_gpu_blocks"]
    check(selection.profile_calls == 1 and len(selection.initialized) == 1,
          "profiling/initialization did not each run exactly once")
    check(cache_config.num_gpu_blocks == n
          and manager.num_total_gpu_blocks == n
          and manager.gpu_allocator.num_blocks == n
          and manager.get_num_free_gpu_blocks() == n
          and not manager.block_tables and manager.watermark_blocks == 0,
          "central capacity")
    check(worker["cache_config_num_gpu_blocks"] == n
          and worker["cache_engine_num_gpu_blocks"] == n
          and worker["cache_tensor_block_dims"] == [n]
          and worker["cache_layers"] > 0
          and worker["total_blocks"] == n and worker["allocator_num_blocks"] == n
          and worker["free_blocks"] == n and worker["block_tables"] == {}
          and worker["block_ops"] == [], f"worker capacity {worker}")


PROVENANCE_FILES = [
    "scripts/check_adapted_vllm_gpu.py", "scripts/check_lp_scheduler_gpu.py",
    "scripts/check_lp_scheduler_preemption_gpu.py",
    "tests/test_check_adapted_vllm_gpu.py",
    "tests/test_adapted_vllm_scheduler.py",
    "sarathi/core/scheduler/adapted_vllm_scheduler.py",
    "sarathi/core/scheduler/base_scheduler.py", "sarathi/config.py",
    "sarathi/engine/arg_utils.py", "sarathi/engine/base_llm_engine.py",
    "sarathi/core/block_space_manager/base_block_space_manager.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/core/sequence_manager/worker_sequence_manager.py",
    "sarathi/worker/base_worker.py", "sarathi/model_executor/model_runner.py",
]


def provenance_record():
    return dict(
        git_head=git("rev-parse", "HEAD").strip(),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        git_status_short=git("status", "--short", "--untracked-files=all"),
        git_diff_head_sha256=hashlib.sha256(
            git("diff", "HEAD").encode()).hexdigest(),
        file_sha256={f: (sha256(REPO_ROOT / f) if (REPO_ROOT / f).exists()
                         else None) for f in PROVENANCE_FILES},
        imported_helper_sha256={single.__name__: sha256(single.__file__),
                                preemption.__name__: sha256(preemption.__file__)},
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def write_artifacts(out_dir, summary, trace):
    (out_dir / "decision_trace.json").write_text(
        json.dumps(to_jsonable(trace), indent=1))
    (out_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=1))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--model-path", required=True,
                        help="existing local directory with the TinyLlama "
                        "config and tokenizer")
    parser.add_argument("--output-dir", required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / f).exists() for f in ("decision_trace.json",
                                            "summary.json")):
        parser.error(f"{out_dir} already holds run artifacts")
    model_path = Path(args.model_path).resolve()

    # Local assets only; never download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    case = CASES[args.case]
    trace = dict(case=args.case)
    summary = dict(passed=False, failure=None, case=args.case,
                   case_parameters={k: v for k, v in case.items()
                                    if k != "decisions"},
                   expected_decisions=case["decisions"],
                   max_nonempty_decisions=MAX_NONEMPTY_DECISIONS,
                   process_timeout_seconds=PROCESS_TIMEOUT_SECONDS,
                   provenance=provenance_record(), argv=sys.argv,
                   model_path=str(model_path))
    engine = None
    try:
        validate_case_constants(args.case)
        check((model_path / "config.json").is_file()
              and (model_path / "tokenizer.json").is_file(),
              f"model path {model_path} lacks config.json/tokenizer.json")
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import sarathi")

        args_record = engine_args(args.case, model_path,
                                  out_dir / "metrics_store_unused")
        configs = args_record.create_engine_configs()
        model_config, cache_config, parallel_config, scheduler_config, _ = configs
        check(type(scheduler_config).__name__ == "AdaptedVLLMSchedulerConfig"
              and (parallel_config.tensor_parallel_size,
                   parallel_config.pipeline_parallel_size)
              == (TENSOR_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE)
              and cache_config.block_size == BLOCK_SIZE
              and model_config.max_model_len == case["max_model_len"],
              "native configuration differs from the case")

        summary["environment"] = environment_record()
        print("environment:", json.dumps(summary["environment"], indent=1),
              flush=True)
        selection = CacheCapacitySelection(chosen=case["num_gpu_blocks"])
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
        print("cache capacity:", json.dumps(to_jsonable(
            summary["cache_capacity"])), flush=True)
        check_capacity(case, selection, cache_config,
                       engine.scheduler.block_manager, worker_init)
        summary["config"] = dict(
            model=MODEL_NAME, engine_args=to_jsonable(vars(args_record)),
            resolved_dtype=str(model_config.dtype),
            hf_config_commit_hash=getattr(model_config.hf_config,
                                          "_commit_hash", None),
            scheduler_class=type(engine.scheduler).__name__,
            scheduler_config=to_jsonable(vars(scheduler_config)),
            vocab_size=model_config.hf_config.vocab_size,
            eos_token_id=engine.tokenizer.eos_token_id,
            engine_init_seconds=round(init_seconds, 3))
        print("config:", json.dumps(summary["config"]), flush=True)

        run_case(args.case, runtime, trace)
        summary.update(
            passed=True,
            decisions=[dict(preempted=d["schedules"][0]["outputs"][
                "preempted_seq_ids"],
                scheduled=d["schedules"][0]["outputs"]["scheduled"],
                sampler_outputs=d["sampler_outputs"])
                for d in trace["decisions"]],
            token_events=trace["token_events"],
            note=("token_events counts emitted decode entries; restarted "
                  "requests exceed max_tokens because the inherited output "
                  "counter resets on recomputation"))
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        cleanup = dict(engine_constructed=engine is not None)
        try:
            write_artifacts(out_dir, summary, trace)
        finally:
            # Stop only the Ray runtime this process started.
            if "ray" in sys.modules:
                ray = sys.modules["ray"]
                cleanup["ray_initialized_before_shutdown"] = ray.is_initialized()
                ray.shutdown()
                cleanup["ray_initialized_after_shutdown"] = ray.is_initialized()
            print("cleanup:", json.dumps(cleanup), flush=True)
            (out_dir / "cleanup.json").write_text(json.dumps(cleanup, indent=1))
    print("RESULT:", "PASS" if summary["passed"] else "FAIL", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
