"""Bounded soft-SLO pilot: timed and untimed PH1 coefficients through the
live single-stage LP scheduler.

TTFT is the time from a request's arrival to its first new token visible at
the engine output; TBT is the time between successive new visible tokens at
that boundary. Both are soft targets, resolved per request from an explicit
override or its class default (``SLO_CLASSES``), with priority 1 for every
request. They are separate from SLAI's prefill-deadline and strict-TBT
sequence fields, which are never set.

Policies (``lpserve_state_mapping``), selected through the live scheduler's
``_coefficient_policy`` hook:

- ``timed``: ``soft_slo_timed_utility`` with the provisional forecast of
  ``BATCH_FORECAST_SECONDS`` per batch (decode ``boost``, prefill token
  ``boost / K``, penalty 1);
- ``untimed``: ``soft_slo_untimed_utility`` (decode 1, prefill token
  ``1 / K``, penalty 1).

Clock: arrivals are explicit driver ``time.monotonic()`` values; the LP
snapshot uses the live scheduler's ``time.monotonic()``; output events and
the cutoff use ``time.monotonic()`` read immediately when ``engine.step()``
returns, before any trace processing. ``OutputClock`` accepts an event only
when the central cumulative output count (``seq.state.num_output_tokens``)
rises by exactly one in a step that scheduled that request as a decode and
the new token ID agrees with the sampler output and the returned
``RequestOutput``. Only a verified event advances ``last_visible``; native
recomputation does not reset it. Native stopping and output representation
are unchanged.

A trial pre-registers its fixed cohort with future arrival times, then
steps the engine until it drains, the cutoff passes, or the decision cap is
reached. Per decision, the timing inputs are copied once in the hook and
coefficients are built by a pure function of them and the mapper's single
capture; mapper, PH1, and executor results are retained by reference and
serialized after the trial. Worker probes run only before and after
trials. ``--cpu`` runs the same driver with a CPU engine (native central
scheduler, engine sequence manager, and single-stage completion; worker
sequence manager replaying copies; synthetic samples). Settings are frozen
in this module before measured runs.

Not established: the best policy, general coefficient or preemption policy,
recomputation output semantics, recovery or overhead criteria, generation
quality, or performance beyond the recorded trials.

GPU run, from the repository root, on an allocated node:

    PYTHONPATH="$PWD" python -B scripts/soft_slo_pilot.py --case ample \\
        --model-path <local TinyLlama snapshot> --output-dir <dir>
"""

import argparse
import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import MappingProxyType
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]

import check_adapted_vllm_gpu as adapted  # noqa: E402
import check_lp_scheduler_gpu as single  # noqa: E402
import check_lp_scheduler_preemption_gpu as preemption  # noqa: E402
from check_adapted_vllm_gpu import (central_state,  # noqa: E402
                                    instrument_block_manager,
                                    worker_validation_state)
from check_lp_scheduler_gpu import (CheckFailure, check,  # noqa: E402
                                    environment_record, to_jsonable)
from check_lp_scheduler_preemption_gpu import (  # noqa: E402
    CacheCapacitySelection, check_declared_order, git, select_prompt_ids,
    sha256)

# ---------------------------------------------------------------------------
# Frozen settings (provisional, scoped to this pilot; D-01 to D-07 OPEN)
# ---------------------------------------------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "auto"
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

SLO_CLASSES = {
    "tight": dict(ttft_target=0.100, tbt_target=0.040),
    "relaxed": dict(ttft_target=0.300, tbt_target=0.120),
}
PRIORITY = 1
BATCH_FORECAST_SECONDS = 0.020
POLICIES = ("timed", "untimed")
# Registration finishes before the arrival epoch t0 = start + this lead.
REGISTRATION_LEAD_SECONDS = 0.5
NUMERICAL_POLICY_FIELDS = dict(
    policy_id="lp_relaxation_mvp_v1", feasibility_tol=1e-7,
    integrality_tol=1e-6, objective_abs_tol=1e-9, objective_rel_tol=1e-9)
# Unused configured triple (the hook supplies every coefficient).
CONFIGURED_TRIPLE = dict(decode_utility=1.0, prefill_token_utility=1.0,
                         preemption_penalty=1.0)


def _request(name, slo_class, prompt_len, arrival_offset, max_tokens,
             prompt_start, **overrides):
    return dict(name=name, slo_class=slo_class, prompt_len=prompt_len,
                arrival_offset=arrival_offset, max_tokens=max_tokens,
                prompt_start=prompt_start, **overrides)


# Within each arrival group the relaxed request registers first, so the
# untimed control's service-rank tie-break favors it (a visible choice).
CASES = {
    "ample": dict(
        requests=(
            _request("relaxed17", "relaxed", 17, 0.000, 6, 1000),
            _request("tight17", "tight", 17, 0.000, 6, 2000),
            _request("relaxed33", "relaxed", 33, 0.020, 6, 3000),
            _request("tight33", "tight", 33, 0.020, 6, 4000),
        ),
        num_gpu_blocks=32, max_model_len=64, b_max=17, c_max=16, s_max=2,
        resident_limit=4, cutoff_seconds=5.0, max_decisions=100),
    "pressure": dict(
        requests=(
            _request("relaxed31", "relaxed", 31, 0.000, 3, 1000),
            _request("tight32", "tight", 32, 0.000, 3, 2000),
        ),
        num_gpu_blocks=5, max_model_len=64, b_max=32, c_max=16, s_max=2,
        resident_limit=2, cutoff_seconds=5.0, max_decisions=100),
    # Continuity control for "pressure": same prompts, enough cache for
    # five outputs each without preemption (six blocks needed).
    "pressure_control": dict(
        requests=(
            _request("relaxed31", "relaxed", 31, 0.000, 5, 1000),
            _request("tight32", "tight", 32, 0.000, 5, 2000),
        ),
        num_gpu_blocks=8, max_model_len=64, b_max=32, c_max=16, s_max=2,
        resident_limit=2, cutoff_seconds=5.0, max_decisions=100),
}
# Measured trial order after the warm-ups: both adjacent orders occur.
WARMUP_ORDER = ("untimed", "timed")
MEASURED_ORDER = ("timed", "untimed", "untimed", "timed")


def settings_record():
    record = dict(
        model=MODEL_NAME, load_format=LOAD_FORMAT, dtype=DTYPE,
        attention_backend=ATTENTION_BACKEND,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=SEED,
        temperature=TEMPERATURE, ignore_eos=IGNORE_EOS, stop=None,
        slo_classes=SLO_CLASSES, priority=PRIORITY,
        batch_forecast_seconds=BATCH_FORECAST_SECONDS,
        registration_lead_seconds=REGISTRATION_LEAD_SECONDS,
        numerical_policy=NUMERICAL_POLICY_FIELDS, cases=CASES,
        warmup_order=WARMUP_ORDER, measured_order=MEASURED_ORDER,
        policies=dict(
            timed="decode boost, prefill token boost/K, penalty 1; "
                  "boost = 1 + clamp((T - slack)/T, 0, 2); slack = due - "
                  "snapshot_time - (1 + ceil(R/K)) * forecast; K = "
                  "min(b_max, c_max)",
            untimed="decode 1, prefill token 1/K, penalty 1"),
        miss_rule="latency strictly above target",
        p99_rule="nearest rank: sorted gaps[ceil(0.99 n) - 1]",
    )
    record["sha256"] = hashlib.sha256(
        json.dumps(to_jsonable(record), sort_keys=True).encode()).hexdigest()
    return record


def resolve_targets(request):
    """``(ttft_target, tbt_target)`` from explicit overrides or the class
    default. Unknown classes and non-positive targets are errors."""
    check(request["slo_class"] in SLO_CLASSES,
          f"{request['name']}: unknown SLO class {request['slo_class']!r}")
    default = SLO_CLASSES[request["slo_class"]]
    targets = tuple(request.get(k, default[k])
                    for k in ("ttft_target", "tbt_target"))
    check(all(isinstance(v, (int, float)) and not isinstance(v, bool)
              and math.isfinite(v) and v > 0 for v in targets),
          f"{request['name']}: targets {targets} must be finite and positive")
    return tuple(float(v) for v in targets)


def nominal_chunk(case):
    return min(case["b_max"], case["c_max"])


def coefficient_policy(policy, case, timing):
    """The coefficient policy of one decision, bound to its timing copy."""
    import lpserve_state_mapping as lsm
    if policy == "timed":
        return functools.partial(
            lsm.soft_slo_timed_utility, timing=timing,
            batch_forecast=BATCH_FORECAST_SECONDS, b_max=case["b_max"],
            c_max=case["c_max"])
    check(policy == "untimed", f"unknown policy {policy!r}")
    return functools.partial(lsm.soft_slo_untimed_utility,
                             b_max=case["b_max"], c_max=case["c_max"])


# ---------------------------------------------------------------------------
# Output clock
# ---------------------------------------------------------------------------


class OutputClock:
    """Verified visible-output events of one trial's registered cohort.

    Holds a reference to every registered ``Sequence`` through completion
    (finished requests leave the engine's sequence map), its timing record,
    and its last verified cumulative output count."""

    def __init__(self, cutoff=None):
        self.cutoff = cutoff
        self.requests = {}
        self.events = []

    def register(self, seq, name, slo_class, ttft_target, tbt_target,
                 priority):
        sid = seq.seq_id
        check(sid not in self.requests, f"seq {sid} registered twice")
        check(seq.state.num_output_tokens == 0 and not seq.output_token_ids,
              f"seq {sid} has output before registration")
        self.requests[sid] = dict(
            seq=seq, name=name, slo_class=slo_class, ttft_target=ttft_target,
            tbt_target=tbt_target, priority=priority,
            arrival_time=float(seq.arrival_time), count=0, last_visible=None,
            tokens=[], event_times=[], finish_time=None)

    def timing_inputs(self):
        """One immutable copy of every registered request's timing input."""
        import lpserve_state_mapping as lsm
        return MappingProxyType({
            sid: lsm.RequestTiming(
                raw_seq_id=sid, slo_class=r["slo_class"],
                ttft_target=r["ttft_target"], tbt_target=r["tbt_target"],
                priority=r["priority"], arrival_time=r["arrival_time"],
                last_visible=r["last_visible"])
            for sid, r in self.requests.items()})

    def observe(self, returned_at, decision, scheduled, preempted, sampler,
                request_outputs):
        """Verify one returned step and record its new visible tokens.

        ``scheduled`` is ``[(seq_id, prompt_chunk_len)]`` in emitted order,
        ``sampler`` the step's ``[[seq_id, token_id]]`` (``None`` when no
        model ran), and ``request_outputs`` ``[(seq_id, token_ids copy,
        finished)]`` as returned. Returns the new event records."""
        tag = f"decision {decision}"
        ids = [sid for sid, _ in scheduled]
        if not scheduled:
            check(sampler is None and not request_outputs,
                  f"{tag}: outputs without scheduled work")
            return []
        check(sampler is not None and len(sampler) == len(scheduled)
              and [s[0] for s in sampler] == ids,
              f"{tag}: sampler outputs {sampler} do not match scheduled "
              f"{scheduled} in count and order")
        check([o[0] for o in request_outputs] == ids,
              f"{tag}: returned outputs {[o[0] for o in request_outputs]} "
              f"do not match scheduled {ids}")
        check(len(set(ids)) == len(ids), f"{tag}: repeated scheduled ID")
        kind = {sid: ("prefill" if chunk > 0 else "decode")
                for sid, chunk in scheduled}
        sampled = {sid: token for sid, token in sampler}
        returned = {o[0]: o for o in request_outputs}
        check(set(ids) <= set(self.requests),
              f"{tag}: scheduled an unregistered request")
        new = []
        for sid, r in self.requests.items():
            seq = r["seq"]
            count = seq.state.num_output_tokens
            delta = count - r["count"]
            check(delta in (0, 1),
                  f"{tag}: seq {sid} cumulative output count moved by "
                  f"{delta}")
            if delta == 0:
                check(kind.get(sid) != "decode",
                      f"{tag}: decode of seq {sid} produced no visible token")
                check(not (seq.is_finished() and r["finish_time"] is None),
                      f"{tag}: seq {sid} finished without a new token")
                continue
            check(kind.get(sid) == "decode" and sid not in preempted,
                  f"{tag}: seq {sid} gained a token without a decode")
            native = list(seq.output_token_ids)
            check(native, f"{tag}: seq {sid} count rose with no output IDs")
            token = native[-1]
            out_ids = returned[sid][1]
            check(token == sampled[sid],
                  f"{tag}: seq {sid} token {token} differs from sampler "
                  f"{sampled[sid]}")
            check(out_ids == native,
                  f"{tag}: seq {sid} returned IDs {out_ids} differ from "
                  f"native {native}")
            event = dict(seq_id=sid, name=r["name"], index=count,
                         token_id=token, time=returned_at, decision=decision,
                         after_cutoff=(self.cutoff is not None
                                       and returned_at > self.cutoff),
                         finished=seq.is_finished())
            r["count"] = count
            r["last_visible"] = returned_at
            r["tokens"].append(token)
            r["event_times"].append(returned_at)
            if seq.is_finished():
                check(returned[sid][2], f"{tag}: seq {sid} finish not returned")
                r["finish_time"] = returned_at
            self.events.append(event)
            new.append(event)
        return new


# ---------------------------------------------------------------------------
# Decision recording (references only inside a step)
# ---------------------------------------------------------------------------


class DecisionRecorder:
    """Installs the coefficient hook and reference-only wrappers for one
    trial; ``close`` restores everything it replaced."""

    def __init__(self, scheduler, clock, policy, case):
        import lp_relaxation_scheduler as lrs
        import lpserve_plan_execution as lpe
        import lpserve_state_mapping as lsm
        self.current = None
        self._restore = []
        self._policy, self._case, self._clock = policy, case, clock

        def hook():
            timing = clock.timing_inputs()
            self._add("hooks", timing)
            return coefficient_policy(policy, case, timing)

        self._set(scheduler, "_coefficient_policy", hook)
        for owner, name, key in ((lsm, "map_scheduler_state", "snapshots"),
                                 (lrs, "solve_and_extract", "results"),
                                 (lpe, "execute_plan", "executions"),
                                 (scheduler, "schedule", "outputs")):
            self._wrap(owner, name, key)

    def _add(self, key, value):
        if self.current is not None:
            self.current[key].append(value)

    def _set(self, owner, name, value):
        had = name in vars(owner)
        self._restore.append((owner, name, had, vars(owner).get(name)))
        setattr(owner, name, value)

    def _wrap(self, owner, name, key):
        native = getattr(owner, name)

        def wrapper(*args, **kwargs):
            result = native(*args, **kwargs)
            self._add(key, result)
            return result
        self._set(owner, name, wrapper)

    def begin(self):
        self.current = dict(hooks=[], snapshots=[], results=[],
                            executions=[], outputs=[])

    def end(self):
        record, self.current = self.current, None
        return record

    def close(self):
        for owner, name, had, value in reversed(self._restore):
            if had:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        del self._restore[:]


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


def prompt_ids(case, tokenizer=None, vocab_size=None):
    """Fixed prompt token IDs: ordinary vocabulary tokens on GPU (special
    and unknown tokens skipped), consecutive integers on CPU."""
    prompts = {}
    for r in case["requests"]:
        if tokenizer is None:
            prompts[r["name"]] = list(range(r["prompt_start"],
                                            r["prompt_start"] + r["prompt_len"]))
        else:
            prompts[r["name"]] = select_prompt_ids(
                tokenizer, vocab_size, r["prompt_start"], r["prompt_len"])[0]
        check(len(prompts[r["name"]]) == r["prompt_len"], "prompt length")
    flat = [t for p in prompts.values() for t in p]
    check(len(flat) == len(set(flat)), "prompts must be disjoint")
    return prompts


def check_quiescent_engine(engine, case, tag):
    scheduler = engine.scheduler
    manager = scheduler.block_manager
    check(not scheduler.waiting and not scheduler.running
          and scheduler.num_running_batches == 0
          and not manager.block_tables and not engine.seq_manager.seq_map
          and manager.get_num_free_gpu_blocks() == case["num_gpu_blocks"]
          and manager.num_total_gpu_blocks == case["num_gpu_blocks"],
          f"{tag}: engine is not quiescent and empty")


def run_trial(engine, case_name, policy, label, prompts, worker_state):
    """One trial; returns its record (references, serialized later). A
    failure is recorded in the returned record and re-raised as
    ``TrialFailure``."""
    from sarathi.core.datatypes.sampling_params import SamplingParams
    case = CASES[case_name]
    scheduler = engine.scheduler
    seq_map = engine.seq_manager.seq_map
    check_quiescent_engine(engine, case, f"{label} start")
    worker_start = worker_state()
    check(worker_start["seqs"] == {} and worker_start["block_tables"] == {}
          and worker_start["free_blocks"] == case["num_gpu_blocks"],
          f"{label}: worker not empty at start")
    clock = OutputClock()
    recorder = DecisionRecorder(scheduler, clock, policy, case)
    trial = dict(label=label, case=case_name, policy=policy, cohort=[],
                 steps=[], stop_reason=None, failure=None)
    try:
        start = time.monotonic()
        t0 = start + REGISTRATION_LEAD_SECONDS
        for r in case["requests"]:
            ttft, tbt = resolve_targets(r)
            before = set(seq_map)
            engine.add_request(
                prompt=None, prompt_token_ids=list(prompts[r["name"]]),
                sampling_params=SamplingParams(
                    temperature=TEMPERATURE, max_tokens=r["max_tokens"],
                    ignore_eos=IGNORE_EOS, stop=None),
                arrival_time=t0 + r["arrival_offset"])
            added = set(seq_map) - before
            check(len(added) == 1, f"{label}: registration of {r['name']}")
            sid = added.pop()
            clock.register(seq_map[sid], r["name"], r["slo_class"], ttft, tbt,
                           PRIORITY)
            trial["cohort"].append(dict(
                seq_id=sid, name=r["name"], slo_class=r["slo_class"],
                ttft_target=ttft, tbt_target=tbt, priority=PRIORITY,
                prompt_len=r["prompt_len"], max_tokens=r["max_tokens"],
                arrival_time=t0 + r["arrival_offset"]))
        registered = time.monotonic()
        cutoff = t0 + case["cutoff_seconds"]
        clock.cutoff = cutoff
        trial.update(registration_start=start, registration_end=registered,
                     t0=t0, cutoff=cutoff)
        check(registered < t0, f"{label}: cohort registration ended at "
              f"{registered} after the arrival epoch {t0}")
        worker_state()  # drop registration-time worker operations
        index = 0
        while True:
            if not scheduler.has_unfinished_seqs():
                trial["stop_reason"] = "drained"
                break
            now = time.monotonic()
            if now >= cutoff:
                trial["stop_reason"] = "cutoff"
                break
            pending = [r["arrival_time"] for r in clock.requests.values()
                       if r["finish_time"] is None]
            if min(pending) > now:  # nothing arrived: idle without a call
                time.sleep(min(min(pending), cutoff) - now)
                continue
            if index >= case["max_decisions"]:
                trial["stop_reason"] = "decision_cap"
                break
            recorder.begin()
            engine.validation_sampler_outputs = None
            started = time.monotonic()
            try:
                step_outputs = engine.step()
                returned = time.monotonic()
            except BaseException:
                trial["failed_step"] = dict(index=index, started=started,
                                            record=recorder.end())
                raise
            record = recorder.end()
            outputs = [(o.seq_id, list(o.token_ids), o.finished)
                       for o in step_outputs]
            sampler = engine.validation_sampler_outputs
            step = dict(index=index, started=started, returned=returned,
                        record=record, request_outputs=outputs,
                        sampler=sampler)
            trial["steps"].append(step)
            check(len(record["outputs"]) == 1,
                  f"{label} decision {index}: {len(record['outputs'])} "
                  "schedule calls")
            emitted = record["outputs"][0]
            step["scheduled"] = [(m.seq_id, m.prompt_chunk_len)
                                 for m in emitted.scheduled_seq_metadata_list]
            step["preempted"] = list(emitted.preempted_seq_ids)
            check(step["scheduled"], f"{label} decision {index}: empty "
                  "decision with arrived unfinished work")
            step["events"] = clock.observe(
                returned, index, step["scheduled"], step["preempted"],
                sampler, outputs)
            for sid, r in clock.requests.items():
                if r["finish_time"] == returned:
                    check(sid not in seq_map
                          and sid not in scheduler.block_manager.block_tables
                          and all(s.seq_id != sid for s in scheduler.running),
                          f"{label} decision {index}: finished seq {sid} "
                          "not released")
            index += 1
    except BaseException as err:
        trial["failure"] = dict(type=type(err).__name__, message=str(err),
                                traceback=traceback.format_exc())
        for name in ("stage", "category", "reason"):
            if hasattr(err, name):
                trial["failure"][name] = getattr(err, name)
    finally:
        recorder.close()
        trial["clock"] = clock
    try:
        trial["final_central"] = central_state(scheduler, seq_map)
        trial["final_worker"] = worker_state()
    except BaseException as err:  # keep the original failure visible
        trial.setdefault("probe_failure", repr(err))
    if trial["failure"] is None and trial["stop_reason"] == "drained":
        fc, fw = trial["final_central"], trial["final_worker"]
        try:
            check(fc["waiting"] == [] and fc["running"] == []
                  and fc["seqs"] == {} and fc["block_tables"] == {}
                  and fc["free_blocks"] == case["num_gpu_blocks"]
                  and fw["seqs"] == {} and fw["block_tables"] == {}
                  and fw["free_blocks"] == case["num_gpu_blocks"]
                  and all(r["finish_time"] is not None
                          for r in clock.requests.values()),
                  f"{label}: drained trial did not release everything")
        except CheckFailure as err:
            trial["failure"] = dict(type=type(err).__name__, message=str(err),
                                    traceback=traceback.format_exc())
    return trial


# ---------------------------------------------------------------------------
# Per-trial analysis (after the trial; nothing here runs inside a step)
# ---------------------------------------------------------------------------


def _problem_actions(problem, plan):
    raw = {r.request_id: int(r.request_id) for r in problem.requests}
    out = {}
    for d in plan.decisions:
        if d.preempt:
            out[raw[d.request_id]] = "preempt"
        elif d.prefill_tokens:
            out[raw[d.request_id]] = f"prefill:{d.prefill_tokens}"
        elif d.decode:
            out[raw[d.request_id]] = "decode"
    return out


def problem_under(snapshot, policy, case, timing):
    """The ``LPProblem`` of the same capture and timing under ``policy``."""
    import lpserve_state_mapping as lsm
    coefficients = lsm.construct_coefficients(
        snapshot, coefficient_policy(policy, case, timing))
    check(not isinstance(coefficients, lsm.MappingFailure),
          f"coefficients under {policy}: {coefficients}")
    ranks = tuple((r.service_rank, r.victim_rank) for r in snapshot.requests)
    rebuilt = lsm.build_state_snapshot(
        snapshot, ranks=ranks, coefficients=coefficients,
        b_max=snapshot.lp_problem.b_max, c_max=snapshot.lp_problem.c_max,
        s_max=snapshot.lp_problem.s_max,
        numerical_policy=snapshot.numerical_policy)
    check(not isinstance(rebuilt, lsm.MappingFailure),
          f"problem under {policy}: {rebuilt}")
    return rebuilt.lp_problem


def _contention(problem):
    eligible = [r for r in problem.requests
                if r.prefill_eligible or r.decode_eligible]
    tokens = sum(r.prefill_upper_bound if r.prefill_eligible else 1
                 for r in eligible)
    return dict(eligible_actions=len(eligible), eligible_tokens=tokens,
                action_contention=len(eligible) > problem.s_max,
                token_contention=tokens > problem.b_max,
                contended=len(eligible) > problem.s_max
                or tokens > problem.b_max)


def analyze_trial(trial):
    """Check every decision of a trial and derive its diagnostics."""
    import lp_relaxation_scheduler as lrs
    import lpserve_state_mapping as lsm
    case = CASES[trial["case"]]
    policy, clock = trial["policy"], trial["clock"]
    other = "untimed" if policy == "timed" else "timed"
    decisions, finished_before = [], set()
    by_request = {sid: dict(prefill_tokens=0, preemptions=0)
                  for sid in clock.requests}
    for step in trial["steps"]:
        tag = f"{trial['label']} decision {step['index']}"
        rec = step["record"]
        counts = {k: len(rec[k]) for k in ("hooks", "snapshots", "results",
                                           "executions", "outputs")}
        check(set(counts.values()) == {1}, f"{tag}: calls {counts}")
        snap, result = rec["snapshots"][0], rec["results"][0]
        timing = rec["hooks"][0]
        check(isinstance(snap, lsm.StateSnapshot),
              f"{tag}: mapping returned {type(snap).__name__}")
        check(isinstance(result, lrs.SchedulingSuccess),
              f"{tag}: planning returned {type(result).__name__}")
        problem, plan = snap.lp_problem, result.plan
        # Arrivals: exactly the arrived, unfinished cohort is captured.
        included = {r.raw_seq_id for r in snap.requests}
        expected = {sid for sid, r in clock.requests.items()
                    if r["arrival_time"] <= snap.snapshot_time
                    and sid not in finished_before}
        check(included == expected,
              f"{tag}: captured {sorted(included)}, expected arrived "
              f"unfinished {sorted(expected)}")
        check(step["started"] <= snap.snapshot_time <= step["returned"],
              f"{tag}: snapshot time outside the step")
        # Coefficients: the pure policy of the held inputs, unchanged.
        held = problem_under(snap, policy, case, timing)
        check(held == problem, f"{tag}: coefficients differ from a pure "
              "recomputation of the captured inputs")
        # Legality and declared/emitted replay order.
        check(not isinstance(lrs.validate_integer_plan(problem, plan),
                             lrs.Failure), f"{tag}: plan fails revalidation")
        emitted = dict(preempted_seq_ids=step["preempted"],
                       scheduled=[list(e) for e in step["scheduled"]])
        check_declared_order(tag, to_jsonable(snap), to_jsonable(plan),
                             emitted)
        forecasts = {
            r.raw_seq_id: lsm.soft_slo_forecast(
                snap, r, timing=timing, batch_forecast=BATCH_FORECAST_SECONDS,
                b_max=problem.b_max, c_max=problem.c_max)
            for r in snap.requests}
        alternative = lrs.solve_and_extract(
            problem_under(snap, other, case, timing))
        check(isinstance(alternative, lrs.SchedulingSuccess),
              f"{tag}: {other} counterfactual failed")
        actions = _problem_actions(problem, plan)
        other_actions = _problem_actions(problem, alternative.plan)
        for sid, chunk in step["scheduled"]:
            by_request[sid]["prefill_tokens"] += chunk
        for sid in step["preempted"]:
            by_request[sid]["preemptions"] += 1
        decisions.append(dict(
            index=step["index"], snapshot_time=snap.snapshot_time,
            batch_seconds=step["returned"] - step["started"],
            free_blocks=snap.free_physical_blocks,
            resident=list(snap.running_order),
            waiting=list(snap.arrived_waiting_order),
            actions={str(k): v for k, v in sorted(actions.items())},
            counterfactual_policy=other,
            counterfactual_actions={str(k): v for k, v in
                                    sorted(other_actions.items())},
            action_difference=actions != other_actions,
            contention=_contention(problem),
            forecasts={str(k): v for k, v in forecasts.items()},
            events=[(e["seq_id"], e["index"], e["token_id"])
                    for e in step["events"]]))
        for e in step["events"]:
            if e["finished"]:
                finished_before.add(e["seq_id"])
    return dict(decisions=decisions,
                forecast_errors=forecast_errors(trial, decisions),
                boost_regions=_boost_regions(decisions),
                work=_work(trial, by_request))


def _boost_regions(decisions):
    out = {}
    for d in decisions:
        for f in d["forecasts"].values():
            key = f"{f.target_kind}:{f.boost_region}"
            out[key] = out.get(key, 0) + 1
    return out


def forecast_errors(trial, decisions):
    """Per decision and request: observed time to the next verified token
    minus the forecast ``estimated_batches * batch_forecast``; plus each
    step's duration minus the per-batch forecast. Diagnostics only."""
    clock = trial["clock"]
    rows = []
    for d in decisions:
        for sid, f in d["forecasts"].items():
            times = [t for t in clock.requests[int(sid)]["event_times"]
                     if t > d["snapshot_time"]]
            if not times:
                continue
            nxt = times[0]
            batches = sum(1 for e in decisions
                          if d["snapshot_time"] <= e["snapshot_time"] < nxt)
            rows.append(dict(
                decision=d["index"], seq_id=int(sid),
                forecast_seconds=f.estimated_batches * f.batch_forecast,
                observed_seconds=nxt - d["snapshot_time"],
                error_seconds=(nxt - d["snapshot_time"]
                               - f.estimated_batches * f.batch_forecast),
                forecast_batches=f.estimated_batches,
                observed_batches=batches))
    batch_errors = [d["batch_seconds"] - BATCH_FORECAST_SECONDS
                    for d in decisions]
    return dict(next_token=rows, batch_error_seconds=batch_errors,
                mean_batch_seconds=(statistics.fmean(
                    d["batch_seconds"] for d in decisions)
                    if decisions else None))


def _work(trial, by_request):
    clock = trial["clock"]
    out = {}
    for c in trial["cohort"]:
        sid = c["seq_id"]
        r = clock.requests[sid]
        out[c["name"]] = dict(
            prefill_tokens=by_request[sid]["prefill_tokens"],
            repeated_prefill_tokens=(by_request[sid]["prefill_tokens"]
                                     - c["prompt_len"]),
            preemptions=by_request[sid]["preemptions"],
            visible_tokens=len(r["tokens"]), requested=c["max_tokens"],
            extra_outputs=max(0, len(r["tokens"]) - c["max_tokens"]),
            finished=r["finish_time"] is not None)
    return out


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------


def nearest_rank(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def latency_metrics(requests, cutoff):
    """Per-request and per-class TTFT/TBT results of a fixed cohort.

    ``requests`` holds dicts with ``name``, ``slo_class``, ``arrival_time``,
    ``ttft_target``, ``tbt_target``, ``event_times`` (all verified events,
    ascending), and ``finish_time`` (``None`` if unfinished). Events after
    ``cutoff`` are excluded from observations and listed separately. A miss
    is latency strictly above target."""
    per_request, after_cutoff = [], []
    for r in requests:
        within = [t for t in r["event_times"] if t <= cutoff]
        after_cutoff += [dict(name=r["name"], time=t) for t in
                         r["event_times"] if t > cutoff]
        finished = r["finish_time"] is not None and r["finish_time"] <= cutoff
        row = dict(name=r["name"], slo_class=r["slo_class"],
                   ttft_target=r["ttft_target"], tbt_target=r["tbt_target"],
                   events_within_cutoff=len(within),
                   finished_by_cutoff=finished)
        if within:
            ttft = within[0] - r["arrival_time"]
            row.update(ttft_status="observed", ttft=ttft,
                       ttft_miss=ttft > r["ttft_target"],
                       ttft_excess=max(0.0, ttft - r["ttft_target"]))
        else:
            due = r["arrival_time"] + r["ttft_target"]
            if due <= cutoff:
                row.update(ttft_status="known_miss_missing",
                           ttft_miss=True,
                           ttft_excess_lower_bound=cutoff - due)
            else:
                row.update(ttft_status="unresolved", ttft_miss=None)
        gaps = [b - a for a, b in zip(within, within[1:])]
        row.update(gaps=gaps, gap_misses=[g > r["tbt_target"] for g in gaps],
                   gap_excess=[max(0.0, g - r["tbt_target"]) for g in gaps])
        if within and not finished:
            open_gap = cutoff - within[-1]
            row["open_gap"] = dict(
                elapsed=open_gap, overdue=open_gap > r["tbt_target"],
                excess_lower_bound=max(0.0, open_gap - r["tbt_target"]))
        per_request.append(row)
    by_class = {}
    for cls in sorted({r["slo_class"] for r in per_request}):
        rows = [r for r in per_request if r["slo_class"] == cls]
        observed = [r for r in rows if r["ttft_status"] == "observed"]
        known = [r for r in rows if r["ttft_status"] == "known_miss_missing"]
        unresolved = [r for r in rows if r["ttft_status"] == "unresolved"]
        misses = sum(r["ttft_miss"] for r in observed) + len(known)
        n = len(rows)
        ttft = dict(
            cohort=n, observed=len(observed),
            median=(statistics.median(r["ttft"] for r in observed)
                    if observed else None),
            miss_rate=(misses / n) if not unresolved else None,
            miss_rate_bounds=([misses / n, (misses + len(unresolved)) / n]
                              if unresolved else None),
            mean_excess_observed=(statistics.fmean(
                r["ttft_excess"] for r in observed) if observed else None),
            known_missing_misses=len(known), unresolved=len(unresolved),
            censored_excess_lower_bounds=[
                dict(name=r["name"], lower_bound=r["ttft_excess_lower_bound"])
                for r in known])
        gaps = [g for r in rows for g in r["gaps"]]
        gap_miss = [m for r in rows for m in r["gap_misses"]]
        excess = [e for r in rows for e in r["gap_excess"]]
        with_gap = [r for r in rows if r["gaps"]]
        tbt = dict(
            observed_gaps=len(gaps),
            miss_rate=(sum(gap_miss) / len(gaps)) if gaps else None,
            mean_excess=statistics.fmean(excess) if excess else None,
            p99=nearest_rank(gaps, 0.99),
            streams_with_observed_gap=len(with_gap),
            streams_without_observed_gap=len(rows) - len(with_gap),
            share_streams_with_late_gap=(
                sum(any(r["gap_misses"]) for r in with_gap) / len(with_gap)
                if with_gap else None),
            overdue_open_gaps=[dict(name=r["name"], **r["open_gap"])
                               for r in rows if r.get("open_gap")
                               and r["open_gap"]["overdue"]])
        by_class[cls] = dict(ttft=ttft, tbt=tbt)
    return dict(cutoff=cutoff, per_request=per_request, by_class=by_class,
                events_after_cutoff=after_cutoff)


def trial_metrics(trial):
    clock = trial["clock"]
    rows = [dict(name=r["name"], slo_class=r["slo_class"],
                 arrival_time=r["arrival_time"], ttft_target=r["ttft_target"],
                 tbt_target=r["tbt_target"], event_times=list(r["event_times"]),
                 finish_time=r["finish_time"])
            for r in clock.requests.values()]
    return latency_metrics(rows, trial["cutoff"])


# ---------------------------------------------------------------------------
# Offline same-state comparisons
# ---------------------------------------------------------------------------


def _captured_request(raw_id, prompt_len, arrival, block_size):
    import lpserve_state_mapping as lsm
    blocks = -(-prompt_len // block_size)
    return lsm.CapturedRequest(
        raw_seq_id=raw_id, request_id=str(raw_id), order_key=(raw_id,),
        ownership=lsm.OWNERSHIP_WAITING, queue_position=raw_id,
        status="WAITING", arrival_time=arrival, prompt_len=prompt_len,
        prompt_tokens_processed=0, prompt_tokens_remaining=prompt_len,
        prompt_processing_finished=False, output_len=0,
        context_len=prompt_len, logical_block_count=blocks,
        physical_block_count=0, prefill_eligible=True, decode_eligible=False,
        preemption_eligible=False, prefill_fixed_charge=blocks,
        decode_demand=0, preemption_recovery=0)


def flip_example():
    """Hand-checkable flip: B=C=16, S=1, ample cache, both arrivals at 0,
    now = 60 ms; relaxed 16-token prefill versus tight 12-token prefill.
    Untimed rewards 1 and 0.75; timed about 1.33 and 1.50. Both PH1 and the
    exact integer optimum under each policy."""
    import exact_integer_plan as eip
    import lp_relaxation_scheduler as lrs
    import lpserve_state_mapping as lsm
    case = dict(b_max=16, c_max=16, s_max=1)
    requests = (_captured_request(0, 16, 0.0, BLOCK_SIZE),
                _captured_request(1, 12, 0.0, BLOCK_SIZE))
    captured = lsm.CapturedState(
        snapshot_id="flip", snapshot_time=0.060, scheduler_iteration_id=0,
        num_pipeline_stages=1, num_running_batches=0, resident_count=0,
        resident_limit=4, max_model_len=64, block_size=BLOCK_SIZE,
        total_blocks=32, free_physical_blocks=32, watermark_blocks=0,
        requests=requests, running_order=(), arrived_waiting_order=(0, 1))
    timing = MappingProxyType({
        0: lsm.RequestTiming(0, "relaxed", **SLO_CLASSES["relaxed"],
                             priority=1, arrival_time=0.0, last_visible=None),
        1: lsm.RequestTiming(1, "tight", **SLO_CLASSES["tight"], priority=1,
                             arrival_time=0.0, last_visible=None)})
    out = {}
    for policy in POLICIES:
        coefficients = lsm.construct_coefficients(
            captured, coefficient_policy(policy, case, timing))
        snap = lsm.build_state_snapshot(
            captured, ranks=lsm.vllm_oriented_ranks(captured),
            coefficients=coefficients, numerical_policy=lrs.NumericalPolicy(
                **NUMERICAL_POLICY_FIELDS), **case)
        check(isinstance(snap, lsm.StateSnapshot), f"flip snapshot {snap}")
        ph1 = lrs.solve_and_extract(snap.lp_problem)
        exact = eip.solve_integer_plan(snap.lp_problem)
        check(isinstance(ph1, lrs.SchedulingSuccess)
              and isinstance(exact, eip.IntegerOptimum),
              f"flip {policy}: {ph1} {exact}")
        rewards = {r.request_id: r.prefill_token_utility * r.prompt_tokens_remaining
                   for r in snap.lp_problem.requests}
        out[policy] = dict(
            prefill_token_utility={r.request_id: r.prefill_token_utility
                                   for r in snap.lp_problem.requests},
            full_prefill_reward=rewards,
            ph1_actions=_problem_actions(snap.lp_problem, ph1.plan),
            exact_actions=_problem_actions(snap.lp_problem, exact.plan),
            ph1_objective=ph1.plan.objective,
            exact_objective=exact.plan.objective,
            forecasts=({r.raw_seq_id: lsm.soft_slo_forecast(
                captured, r, timing=timing,
                batch_forecast=BATCH_FORECAST_SECONDS, **{
                    k: case[k] for k in ("b_max", "c_max")})
                for r in requests} if policy == "timed" else None))
    return out


def same_state_exact(trial, limit=None):
    """PH1 and the exact integer optimum under both policies on the
    retained snapshots of ``trial`` (identical capture and timing)."""
    import exact_integer_plan as eip
    import lp_relaxation_scheduler as lrs
    case = CASES[trial["case"]]
    rows = []
    for step in trial["steps"][:limit]:
        snap, timing = step["record"]["snapshots"][0], step["record"]["hooks"][0]
        row = dict(decision=step["index"])
        for policy in POLICIES:
            problem = problem_under(snap, policy, case, timing)
            ph1 = lrs.solve_and_extract(problem)
            exact = eip.solve_integer_plan(problem)
            row[policy] = dict(
                ph1_status=type(ph1).__name__,
                exact_status=type(exact).__name__,
                ph1_actions=({str(k): v for k, v in _problem_actions(
                    problem, ph1.plan).items()}
                    if isinstance(ph1, lrs.SchedulingSuccess) else None),
                exact_actions=({str(k): v for k, v in _problem_actions(
                    problem, exact.plan).items()}
                    if isinstance(exact, eip.IntegerOptimum) else None),
                comparison=eip.compare_same_state(problem, ph1, exact))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def engine_args(case_name, model_path, metrics_dir):
    from lp_relaxation_scheduler import NumericalPolicy
    from sarathi.engine.arg_utils import EngineArgs
    c = CASES[case_name]
    return EngineArgs(
        model=str(model_path), tokenizer=str(model_path),
        tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
        load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE, block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=c["max_model_len"], max_num_seqs=c["resident_limit"],
        write_metrics=True, output_dir=str(metrics_dir),
        enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
        enable_chrome_trace=False, enable_request_outputs=False,
        keep_individual_batch_metrics=True,
        attention_backend=ATTENTION_BACKEND, scheduler_type="lp",
        lp_b_max=c["b_max"], lp_c_max=c["c_max"], lp_s_max=c["s_max"],
        lp_decode_utility=CONFIGURED_TRIPLE["decode_utility"],
        lp_prefill_token_utility=CONFIGURED_TRIPLE["prefill_token_utility"],
        lp_preemption_penalty=CONFIGURED_TRIPLE["preemption_penalty"],
        lp_numerical_policy=NumericalPolicy(**NUMERICAL_POLICY_FIELDS))


def cpu_configs(case_name, tmp):
    """Native engine configurations with model metadata mocked (CPU)."""
    from sarathi.engine import arg_utils
    model_config = mock.Mock()
    model_config.get_max_model_len.return_value = CASES[case_name][
        "max_model_len"]
    model_config.get_total_num_layers.return_value = 2
    with mock.patch.object(arg_utils, "ModelConfig",
                           return_value=model_config):
        return engine_args(case_name, Path(tmp) / "model",
                           Path(tmp) / "metrics").create_engine_configs()


class _NullMetrics:
    """No-op metrics store for the CPU engine."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def distinct_sampler(seq_id, step):
    """Synthetic token distinct per request and step."""
    return 100 * (step + 1) + seq_id


def build_cpu_engine(case_name, sampler=distinct_sampler):
    """A native ``BaseLLMEngine`` whose single worker is replaced by a CPU
    worker: native central scheduler, engine sequence manager, and
    single-stage step/completion path; a worker sequence manager replays
    copies; samples come from ``sampler(seq_id, step)``. No text."""
    import tempfile

    from sarathi.core.datatypes.sequence import SamplerOutput
    from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry
    from sarathi.core.sequence_manager.engine_sequence_manager import (
        EngineSequenceManager)
    from sarathi.core.sequence_manager.worker_sequence_manager import (
        WorkerSequenceManager)
    from sarathi.engine.base_llm_engine import BaseLLMEngine
    from sarathi.metrics.metrics_store import MetricsStore
    from sarathi.utils import Counter
    from sarathi.utils.singleton import Singleton

    # BaseScheduler reads the MetricsStore singleton; use its disabled mode.
    if MetricsStore not in Singleton._instances:
        MetricsStore(None)

    class _NoTextSequenceManager(EngineSequenceManager):
        def __init__(self):
            super().__init__(tokenizer=None)

        def _decode_seq(self, seq):
            pass

    class CpuEngine(BaseLLMEngine):
        validation_sampler_outputs = None

        def __init__(self):  # no model, Ray worker, or GPU
            with tempfile.TemporaryDirectory() as tmp:
                configs = cpu_configs(case_name, tmp)
            (self.model_config, self.cache_config, self.parallel_config,
             self.scheduler_config, self.metrics_config) = configs
            self.cache_config.num_gpu_blocks = CASES[case_name][
                "num_gpu_blocks"]
            self.tokenizer = mock.Mock(eos_token_id=2)
            self.seq_manager = _NoTextSequenceManager()
            self.seq_counter = Counter()
            self.metrics_store = _NullMetrics()
            self.scheduler = SchedulerRegistry.get(
                self.scheduler_config.type, self.scheduler_config,
                self.cache_config)
            self._scheduler_timer = contextlib.nullcontext()
            self._process_model_outputs_timer = contextlib.nullcontext()
            self.worker_seqs = WorkerSequenceManager(self.cache_config,
                                                     self.scheduler_config)
            self.worker_ops = []
            instrument_block_manager(self.worker_seqs.block_manager,
                                     self.worker_ops)
            self.worker_steps = 0

        def _run_workers(self, method, *args, get_all_outputs=False,
                         ignore_output=False, **kwargs):
            if method == "add_seq":
                self.worker_seqs.add_seq(copy.deepcopy(kwargs["seq"]))
                return None
            if method == "execute_model":
                outputs = kwargs["scheduler_outputs"]
                self.worker_seqs.on_schedule(outputs)
                samples = [SamplerOutput(m.seq_id,
                                         sampler(m.seq_id, self.worker_steps))
                           for m in outputs.scheduled_seq_metadata_list]
                self.worker_steps += 1
                self.worker_seqs.on_step_completed(outputs, samples)
                self.validation_sampler_outputs = [
                    [s.seq_id, s.output_token] for s in samples]
                return samples
            if method == adapted.WORKER_STATE_METHOD:
                state = worker_validation_state(self.worker_seqs,
                                                self.worker_ops)
                return [state] if get_all_outputs else state
            raise CheckFailure(f"CPU worker has no method {method!r}")

    return CpuEngine()


IDENTITY_METHOD = "pilot_identity"


def build_gpu_engine_class(selection):
    """The validation engine and worker of the earlier checks plus a
    read-only worker identity probe (device and weight samples)."""
    from sarathi.engine.base_llm_engine import BaseLLMEngine

    base_worker = adapted.build_worker_class()

    class PilotWorker(base_worker):
        def pilot_identity(self, names):
            import torch
            params = dict(self.model_runner.model.named_parameters())
            probes = {}
            for name in names:
                p = params.get(name)
                probes[name] = None if p is None else dict(
                    shape=list(p.shape), dtype=str(p.dtype),
                    device=str(p.device),
                    head=p.detach().flatten()[:16].float().cpu().tolist(),
                    sum=float(p.detach().float().sum().item()))
            props = torch.cuda.get_device_properties(self.device)
            return dict(
                device=str(self.device),
                device_name=torch.cuda.get_device_name(self.device),
                device_total_memory=props.total_memory,
                device_uuid=str(getattr(props, "uuid", None)),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                load_format=self.model_config.load_format,
                dtype=str(self.model_config.dtype),
                num_parameters=sum(p.numel() for p in params.values()),
                parameter_names_sample=sorted(params)[:8], probes=probes)

    class PilotEngine(BaseLLMEngine):
        validation_sampler_outputs = None

        def _get_worker_impl(self):
            return PilotWorker

        def _run_workers(self, method, *args, **kwargs):
            result = super()._run_workers(method, *args, **kwargs)
            if method == adapted.PROFILE_METHOD:
                return selection.select(result)
            if method == adapted.INIT_CACHE_METHOD:
                selection.record_initialization(
                    kwargs["cache_config"].num_gpu_blocks, result)
            if method == "execute_model":
                self.validation_sampler_outputs = (
                    None if result is None
                    else [[o.seq_id, o.output_token] for o in result])
            return result

    return PilotEngine


WEIGHT_PROBES = ("model.layers.0.input_layernorm.weight", "model.norm.weight")


def check_weights(identity, model_path):
    """Compare worker parameter samples with the safetensors file."""
    from safetensors import safe_open
    record = {}
    with safe_open(str(Path(model_path) / "model.safetensors"), "pt") as f:
        for name in WEIGHT_PROBES:
            ref = f.get_tensor(name).float()
            got = identity["probes"].get(name)
            check(got is not None, f"worker has no parameter {name}")
            head = ref.flatten()[:16].tolist()
            record[name] = dict(file_head=head, worker_head=got["head"],
                                file_sum=float(ref.sum()),
                                worker_sum=got["sum"])
            check(got["head"] == head and list(ref.shape) == got["shape"]
                  and abs(float(ref.sum()) - got["sum"]) <= 1e-3 * max(
                      1.0, abs(float(ref.sum()))),
                  f"worker {name} differs from the safetensors file")
    return record


def _worker_state(engine):
    states = engine._run_workers(adapted.WORKER_STATE_METHOD,
                                 get_all_outputs=True)
    check(len(states) == 1, f"expected one worker, got {len(states)}")
    return states[0]


# ---------------------------------------------------------------------------
# Serialization and process
# ---------------------------------------------------------------------------


def trial_record(trial):
    """JSON record of a trial: raw decisions, outputs, events, cutoff."""
    clock = trial["clock"]
    steps = []
    for s in trial["steps"]:
        rec = s["record"]
        steps.append(dict(
            index=s["index"], started=s["started"], returned=s["returned"],
            timing_inputs=[to_jsonable(t) for t in
                           (dict(h) for h in rec["hooks"])],
            snapshots=[to_jsonable(x) for x in rec["snapshots"]],
            results=[to_jsonable(x) for x in rec["results"]],
            executions=[single.outputs_record(x)
                        if type(x).__name__ == "SchedulerOutputs"
                        else to_jsonable(x) for x in rec["executions"]],
            scheduled=s.get("scheduled"), preempted=s.get("preempted"),
            sampler=s["sampler"], request_outputs=s["request_outputs"],
            events=s.get("events")))
    return to_jsonable(dict(
        label=trial["label"], case=trial["case"], policy=trial["policy"],
        cohort=trial["cohort"], t0=trial.get("t0"),
        cutoff=trial.get("cutoff"),
        registration_start=trial.get("registration_start"),
        registration_end=trial.get("registration_end"),
        stop_reason=trial["stop_reason"], failure=trial["failure"],
        probe_failure=trial.get("probe_failure"),
        failed_step=(None if "failed_step" not in trial else dict(
            index=trial["failed_step"]["index"],
            started=trial["failed_step"]["started"],
            **{k: [to_jsonable(dict(x) if isinstance(x, MappingProxyType)
                               else x) for x in v]
               for k, v in trial["failed_step"]["record"].items()})),
        steps=steps,
        events=clock.events,
        final_central=trial.get("final_central"),
        final_worker=trial.get("final_worker")))


PROVENANCE_FILES = [
    "scripts/soft_slo_pilot.py", "tests/test_soft_slo_pilot.py",
    "lpserve_state_mapping.py", "lp_relaxation_scheduler.py",
    "lpserve_plan_execution.py", "sarathi/core/scheduler/lp_scheduler.py",
    "scripts/exact_integer_plan.py", "scripts/check_adapted_vllm_gpu.py",
    "scripts/check_lp_scheduler_gpu.py",
    "scripts/check_lp_scheduler_preemption_gpu.py",
    "sarathi/engine/base_llm_engine.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/core/datatypes/sequence.py",
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
        loaded_modules=os.environ.get("LOADEDMODULES"))


def summarize_pair(analyses):
    """Interpretation gates of a measured set: contention, preemptions,
    and a coefficient-driven action difference on a captured state."""
    decisions = [d for a in analyses.values() for d in a["decisions"]]
    return dict(
        contended_decisions=sum(d["contention"]["contended"]
                                for d in decisions),
        action_differences=sum(d["action_difference"] for d in decisions),
        preemptions=sum(w["preemptions"] for a in analyses.values()
                        for w in a["work"].values()),
        decisions=len(decisions))


def run_session(engine, case_name, prompts, labels, worker_state):
    """Run the trials in ``labels`` ([(label, policy, measured)]) on one
    engine; stop at the first trial that fails or does not drain."""
    trials, analyses, metrics = [], {}, {}
    for label, policy, measured in labels:
        trial = run_trial(engine, case_name, policy, label, prompts,
                          worker_state)
        trials.append(trial)
        print(f"{label}: {trial['stop_reason']} decisions "
              f"{len(trial['steps'])} failure "
              f"{trial['failure'] and trial['failure']['message']}",
              flush=True)
        if trial["failure"] is not None:
            break
        analyses[label] = analyze_trial(trial)
        metrics[label] = trial_metrics(trial)
        if trial["stop_reason"] != "drained":
            break
    return trials, analyses, metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--cpu", action="store_true",
                        help="CPU engine with synthetic samples")
    parser.add_argument("--model-path", help="local TinyLlama snapshot (GPU)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="one untimed trial with max_tokens 1 per request")
    parser.add_argument("--single", choices=POLICIES,
                        help="one trial of this policy, no warm-up")
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    if not args.cpu and not args.model_path:
        parser.error("--model-path is required without --cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists():
        parser.error(f"{out_dir} already holds run artifacts")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    case = CASES[args.case]
    if args.smoke:
        CASES[args.case] = case = dict(case, requests=tuple(
            dict(r, max_tokens=1) for r in case["requests"]))
    summary = dict(validation_passed=False, argv=sys.argv, case=args.case,
                   cpu=args.cpu, smoke=args.smoke, settings=settings_record(),
                   provenance=provenance_record(), failure=None)
    engine = None
    trials = []
    try:
        summary["flip_example"] = to_jsonable(flip_example())
        if args.cpu:
            engine = build_cpu_engine(args.case)
            prompts = prompt_ids(case)
            summary["environment"] = dict(
                hostname=os.uname().nodename, python=sys.executable,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        else:
            from ray import cloudpickle
            model_path = Path(args.model_path).resolve()
            summary["model_path"] = str(model_path)
            check((model_path / "config.json").is_file()
                  and (model_path / "model.safetensors").is_file(),
                  f"{model_path} lacks config.json/model.safetensors")
            summary["environment"] = environment_record()
            cloudpickle.register_pickle_by_value(adapted)
            if __name__ != "__main__":
                cloudpickle.register_pickle_by_value(sys.modules[__name__])
            args_record = engine_args(args.case, model_path,
                                      out_dir / "metrics_store")
            configs = args_record.create_engine_configs()
            selection = CacheCapacitySelection(chosen=case["num_gpu_blocks"])
            t_init = time.monotonic()
            engine = build_gpu_engine_class(selection)(*configs)
            summary["engine_init_seconds"] = time.monotonic() - t_init
            identity = engine._run_workers(IDENTITY_METHOD, get_all_outputs=True,
                                           names=list(WEIGHT_PROBES))
            check(len(identity) == 1, "expected one worker identity")
            summary["worker_identity"] = identity[0]
            summary["weight_check"] = check_weights(identity[0], model_path)
            check(identity[0]["load_format"] == LOAD_FORMAT,
                  f"worker load format {identity[0]['load_format']}")
            summary["cache_capacity"] = to_jsonable(selection.record())
            summary["config"] = dict(
                engine_args=to_jsonable(vars(args_record)),
                scheduler_class=type(engine.scheduler).__name__,
                scheduler_config=to_jsonable(vars(configs[3])),
                resolved_dtype=str(configs[0].dtype),
                hf_config_commit_hash=getattr(configs[0].hf_config,
                                              "_commit_hash", None),
                eos_token_id=engine.tokenizer.eos_token_id)
            prompts = prompt_ids(case, engine.tokenizer,
                                 configs[0].hf_config.vocab_size)
        summary["prompts"] = prompts
        fit = sum(-(-(r["prompt_len"] + r["max_tokens"]) // BLOCK_SIZE)
                  for r in case["requests"])
        summary["cohort_fit"] = dict(
            max_blocks_without_preemption=fit,
            pool=case["num_gpu_blocks"],
            max_context=max(r["prompt_len"] + r["max_tokens"]
                            for r in case["requests"]),
            max_model_len=case["max_model_len"])
        check(summary["cohort_fit"]["max_context"] <= case["max_model_len"],
              "cohort context exceeds max_model_len")
        if args.smoke:
            labels = [("smoke_untimed", "untimed", False)]
        elif args.single:
            labels = [(f"single_{args.single}", args.single, False)]
        else:
            labels = ([(f"warmup{i}_{p}", p, False)
                       for i, p in enumerate(WARMUP_ORDER)]
                      + [(f"trial{i}_{p}", p, True)
                         for i, p in enumerate(MEASURED_ORDER)])
        trials, analyses, metrics = run_session(
            engine, args.case, prompts, labels,
            functools.partial(_worker_state, engine))
        measured = {k: v for k, v in analyses.items()
                    if not k.startswith(("warmup", "smoke", "single"))}
        summary.update(
            trials={t["label"]: dict(stop_reason=t["stop_reason"],
                                     decisions=len(t["steps"]),
                                     failure=t["failure"])
                    for t in trials},
            analyses=to_jsonable(analyses), metrics=to_jsonable(metrics),
            gates=summarize_pair(measured))
        first = next((t for t in trials if t["label"] in measured), None)
        if first is not None:
            summary["same_state_exact"] = to_jsonable(same_state_exact(first))
        summary["validation_passed"] = (
            len(trials) == len(labels)
            and all(t["failure"] is None for t in trials))
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        cleanup = dict(engine_constructed=engine is not None)
        try:
            trial_dir = out_dir / "trials"
            trial_dir.mkdir(exist_ok=True)
            for t in trials:
                (trial_dir / f"{t['label']}.json").write_text(
                    json.dumps(trial_record(t), indent=1))
            (out_dir / "summary.json").write_text(
                json.dumps(to_jsonable(summary), indent=1))
        finally:
            if "ray" in sys.modules:
                ray = sys.modules["ray"]
                cleanup["ray_initialized_before_shutdown"] = ray.is_initialized()
                ray.shutdown()
            (out_dir / "cleanup.json").write_text(json.dumps(cleanup))
    print("RESULT:", "PASS" if summary["validation_passed"] else "FAIL",
          flush=True)
    return 0 if summary["validation_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
