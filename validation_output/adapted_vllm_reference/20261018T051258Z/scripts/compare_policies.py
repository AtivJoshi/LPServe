"""Evidence extraction only (not a repository test): bounded CPU comparison of
the adapted vLLM reference, the uniform LP (1/1/1), and the rank-based LP
(D = P = G = 1) on the existing A=5/B=8 workload, from fresh identical initial
states, with resident limits 2 and 1. At most DECISION_BOUND scheduling
decisions per run: a test-harness observation bound, not a scheduler restart
cap. Reuses the existing LP replay harness and _Spies wrappers and the
reference test harness; synthetic sampler outputs (no model, no GPU).

Run from the repository root: python -B <this script> <output dir>
"""
import dataclasses
import functools
import json
import sys

sys.path.insert(0, "tests")
import lpserve_state_mapping as lsm  # noqa: E402
import test_adapted_vllm_scheduler as avt  # noqa: E402
import test_lp_scheduler as lpt  # noqa: E402

from sarathi.core.scheduler import adapted_vllm_scheduler as avs  # noqa: E402
from sarathi.core.scheduler.lp_scheduler import LPSchedulingError  # noqa: E402

OUT = sys.argv[1]
POOL = 8
BLOCK_SIZE = lpt.BLOCK_SIZE
MAX_MODEL_LEN = 16
TOKEN_BUDGET = 8
LP_C_MAX = 8
LP_S_MAX = 1
MAX_TOKENS = 2
WORKLOAD = (("A", 0, 5, 1.0), ("B", 1, 8, 2.0))  # name, seq_id, prompt, arrival
DECISION_BOUND = 8


def record_native_calls(harness):
    harness.calls = []
    sched = harness.scheduler
    for name in ("_preempt", "_allocate", "_append_slot"):
        real = getattr(sched, name)

        def wrapper(seq, _real=real, _name=name):
            harness.calls.append((_name, seq.seq_id))
            return _real(seq)
        setattr(sched, name, wrapper)


def make_run(policy, resident_limit):
    if policy == "adapted_vllm":
        h = avt._Harness(POOL, resident_limit, TOKEN_BUDGET, MAX_MODEL_LEN)
    else:
        h = lpt._Harness(num_gpu_blocks=POOL, max_num_seqs=resident_limit,
                         max_model_len=MAX_MODEL_LEN, b_max=TOKEN_BUDGET,
                         c_max=LP_C_MAX, s_max=LP_S_MAX)
        record_native_calls(h)
        if policy == "lp_rank_based":
            coeff = functools.partial(lsm.rank_based_utility, decode_scale=1.0,
                                      prefill_scale=1.0, penalty_scale=1.0)
            h.scheduler._coefficient_policy = lambda: coeff
    seqs = {name: lpt._Harness.add(h, sid, n, arr, MAX_TOKENS)
            for name, sid, n, arr in WORKLOAD}
    return h, seqs


def state(h, seqs):
    names = {s.seq_id: n for n, s in seqs.items()}
    cm, wm = h.scheduler.block_manager, h.worker_seqs.block_manager
    reqs = {}
    for name, seq in seqs.items():
        worker = h.worker_seqs.get_seq(seq.seq_id)
        reqs[name] = dict(
            status=seq.get_status().name,
            prompt_len=seq.get_prompt_len(),
            prompt_processed=seq.get_num_prompt_tokens_processed(),
            phase=("finished" if seq.is_finished() else
                   "decode" if seq.prompt_processing_finished else "prefill"),
            native_output_count=seq.get_output_len(),
            context_len=seq.get_len(),
            logical_blocks=len(seq.logical_token_blocks),
            central_block_ids=([b.block_number for b in cm.block_tables[seq.seq_id]]
                               if seq.seq_id in cm.block_tables else None),
            worker_block_ids=([b.block_number for b in wm.block_tables[seq.seq_id]]
                              if worker is not None and seq.seq_id in wm.block_tables
                              else None),
        )
    return dict(
        running=[names[s.seq_id] for s in h.scheduler.running],
        waiting=[names[s.seq_id] for s in h.scheduler.waiting],
        central_free=cm.get_num_free_gpu_blocks(),
        worker_free=wm.get_num_free_gpu_blocks(),
        watermark_blocks=cm.watermark_blocks,
        requests=reqs,
    )


def lp_record(spies, names):
    rec = {}
    for call in spies.calls:
        if call[0] == "map":
            snap = call[4]
            if isinstance(snap, lsm.MappingFailure):
                rec["mapping_failure"] = dict(category=snap.category,
                                              reason=snap.reason)
                continue
            rec["coefficients_and_ranks"] = {
                names[r.raw_seq_id]: dict(
                    service_rank=r.service_rank, victim_rank=r.victim_rank,
                    alpha=r.utility.decode_utility,
                    beta=r.utility.prefill_token_utility,
                    gamma=r.utility.preemption_penalty)
                for r in snap.requests}
            p = snap.lp_problem
            rec["problem"] = dict(
                b_max=p.b_max, c_max=p.c_max, s_max=p.s_max,
                resident_limit=p.resident_limit, resident_count=p.resident_count,
                m_free=p.m_free, admission_watermark=p.admission_watermark,
                requests={names[int(r.request_id)]: dict(
                    resident=r.resident,
                    prompt_tokens_remaining=r.prompt_tokens_remaining,
                    prefill_upper_bound=r.prefill_upper_bound,
                    prefill_eligible=r.prefill_eligible,
                    decode_eligible=r.decode_eligible,
                    preemption_eligible=r.preemption_eligible,
                    prefill_fixed_charge=r.prefill_fixed_charge,
                    decode_demand=r.decode_demand,
                    preemption_recovery=r.preemption_recovery)
                    for r in p.requests})
        elif call[0] == "solve":
            result = call[2]
            solver = result.solver
            if solver is not None:
                d = solver.diagnostics
                rec["solver"] = dict(category=solver.category,
                                     raw_status=d.raw_status, message=d.message,
                                     iterations=d.iterations, method=d.method)
            if hasattr(result, "plan"):
                rec["relaxed"] = dict(
                    normalized_objective=result.relaxed.normalized_objective,
                    decisions={names[int(x.request_id)]: dict(
                        x=x.x, y=x.y, prefill_indicator=x.prefill_indicator, z=x.z)
                        for x in result.relaxed.decisions})
                plan = result.plan
                rec["integer_plan"] = dict(
                    decisions={names[int(x.request_id)]: dict(
                        prefill_tokens=x.prefill_tokens, decode=x.decode,
                        preempt=x.preempt) for x in plan.decisions},
                    victim_order=[names[int(i)] for i in plan.victim_order],
                    execution_order=[names[int(i)] for i in plan.execution_order],
                    token_use=plan.token_use, action_count=plan.action_count,
                    objective=plan.objective,
                    extraction=dataclasses.asdict(plan.extraction))
            else:
                rec["solve_failure"] = dict(category=result.category,
                                            reason=result.reason)
    return rec


def reference_record(proposal, names):
    return dict(
        selection_order=[dict(request=names[a.seq_id], kind=a.kind,
                              tokens=a.num_tokens, block_demand=a.block_demand)
                         for a in proposal.actions],
        victims=[names[v] for v in proposal.victims],
        running_stop_reason=proposal.running_stop_reason,
        waiting_stop_reason=proposal.waiting_stop_reason,
        token_budget=proposal.token_budget,
        free_blocks=proposal.free_blocks,
        watermark_blocks=proposal.watermark_blocks,
    )


def run(policy, resident_limit):
    h, seqs = make_run(policy, resident_limit)
    names = {s.seq_id: n for n, s in seqs.items()}
    events = {n: 0 for n in seqs}      # generated-token events
    restarts = {n: 0 for n in seqs}
    completion = {n: None for n in seqs}
    first_prefill = {n: None for n in seqs}
    first_decode = {n: None for n in seqs}
    trace = dict(policy=policy, resident_limit=resident_limit, decisions=[])
    stopped_by, failure = "decision_bound", None
    for step in range(DECISION_BOUND):
        if not h.scheduler.has_unfinished_seqs():
            stopped_by = "drained"
            break
        now = 10.0 + step
        entry = dict(decision=step, time=now, before=state(h, seqs))
        try:
            if policy == "adapted_vllm":
                outputs = h.schedule(now)
                entry["reference"] = reference_record(h.proposals[0], names)
            else:
                with lpt._Spies() as spies:
                    try:
                        outputs = h.schedule(now)
                    finally:
                        entry["lp"] = lp_record(spies, names)
        except (LPSchedulingError, avs.AdaptedVLLMSchedulingError) as error:
            failure = dict(decision=step, type=type(error).__name__,
                           category=error.category, reason=error.reason)
            if getattr(error, "proposal", None) is not None:
                entry["reference"] = reference_record(error.proposal, names)
            entry["failure"] = failure
            trace["decisions"].append(entry)
            stopped_by = "failure"
            break
        entry["native_calls"] = [(c, names[i]) for c, i in h.calls]
        entry["emitted"] = [(names[m.seq_id], m.prompt_chunk_len)
                            for m in outputs.scheduled_seq_metadata_list]
        entry["preempted"] = [names[i] for i in outputs.preempted_seq_ids]
        out_before = {n: s.get_output_len() for n, s in seqs.items()}
        h.replay(outputs)
        for n in entry["preempted"]:
            restarts[n] += 1
        for n, chunk in entry["emitted"]:
            if chunk > 0 and first_prefill[n] is None:
                first_prefill[n] = step
            if chunk == 0:
                events[n] += 1
                if first_decode[n] is None:
                    first_decode[n] = step
                assert seqs[n].get_output_len() == out_before[n] + 1
        for n, s in seqs.items():
            if s.is_finished() and completion[n] is None:
                completion[n] = step
        entry["after"] = state(h, seqs)
        entry["generated_token_events_cumulative"] = dict(events)
        trace["decisions"].append(entry)
    else:
        if not h.scheduler.has_unfinished_seqs():
            stopped_by = "drained"
    trace["summary"] = dict(
        decisions=len(trace["decisions"]), stopped_by=stopped_by,
        failure=failure, first_prefill=first_prefill, first_decode=first_decode,
        completion=completion, preemptions=sum(restarts.values()),
        restarts=restarts, generated_token_events=events,
        native_output_counts={n: s.get_output_len() for n, s in seqs.items()},
        finished=[n for n, s in seqs.items() if s.is_finished()],
        unfinished=[n for n, s in seqs.items() if not s.is_finished()],
    )
    return trace


lpt.setUpModule()
runs = {}
for limit in (2, 1):
    for policy in ("adapted_vllm", "lp_uniform", "lp_rank_based"):
        runs[f"{policy}_resident{limit}"] = run(policy, limit)

config = dict(
    workload=[dict(name=n, seq_id=i, prompt_tokens=p, arrival_time=a,
                   max_tokens=MAX_TOKENS) for n, i, p, a in WORKLOAD],
    pool_blocks=POOL, block_size=BLOCK_SIZE, max_model_len=MAX_MODEL_LEN,
    native_watermark_blocks=0, resident_limits=[2, 1],
    decision_bound=DECISION_BOUND, decision_times="10.0 + decision index",
    sampler="synthetic: every scheduled entry gets token "
            f"{lpt.SAMPLED_TOKEN}; final-prefill samples are discarded natively",
    adapted_vllm=dict(max_num_batched_tokens=TOKEN_BUDGET,
                      chunk_cap=None, action_cap=None),
    lp=dict(b_max=TOKEN_BUDGET, c_max=LP_C_MAX, s_max=LP_S_MAX,
            numerical_policy=dataclasses.asdict(lpt.NUMERICAL_POLICY),
            uniform=dict(decode_utility=1.0, prefill_token_utility=1.0,
                         preemption_penalty=1.0),
            rank_based=dict(decode_scale=1.0, prefill_scale=1.0,
                            penalty_scale=1.0)),
)
with open(f"{OUT}/trace.json", "w") as f:
    json.dump(dict(config=config, runs=runs), f, indent=1)

lines = ["Bounded CPU comparison (decision indices, not wall-clock latency)", ""]
fmt = "{:<28} {:>4} {:>13} {:>13} {:>13} {:>13} {:>4} {:>10} {:>9}"
lines.append(fmt.format("run", "dec", "first prefill", "first decode",
                        "completion", "token events", "pre", "unfinished",
                        "stopped"))
ab = lambda d: f"A{d['A']} B{d['B']}"  # noqa: E731
for key, t in runs.items():
    s = t["summary"]
    lines.append(fmt.format(key, s["decisions"], ab(s["first_prefill"]),
                            ab(s["first_decode"]), ab(s["completion"]),
                            ab(s["generated_token_events"]), s["preemptions"],
                            ",".join(s["unfinished"]) or "-", s["stopped_by"]))
lines += ["", "Per decision: running|waiting before -> preempted / emitted "
          "(free central/worker after)"]
for key, t in runs.items():
    lines.append(f"[{key}]")
    for d in t["decisions"]:
        b = d["before"]
        head = (f"  {d['decision']}: {','.join(b['running']) or '-'}|"
                f"{','.join(b['waiting']) or '-'} free={b['central_free']}")
        if "failure" in d:
            lines.append(head + f" -> FAILURE {d['failure']['category']}")
            continue
        extra = ""
        if "reference" in d:
            r = d["reference"]
            extra = (f" stop=({r['running_stop_reason']},"
                     f"{r['waiting_stop_reason']})")
        if "lp" in d and "relaxed" in d["lp"]:
            lp = d["lp"]
            coeffs = " ".join(
                f"{n}=({c['alpha']:.3g},{c['beta']:.3g},{c['gamma']:.3g})"
                for n, c in sorted(lp["coefficients_and_ranks"].items()))
            rel = " ".join(
                f"{n}:x={v['x']:.3g},y={v['y']:.3g},z={v['z']:.3g}"
                for n, v in sorted(lp["relaxed"]["decisions"].items()))
            extra = (f" coeffs {coeffs}; relaxed {rel} "
                     f"obj={lp['relaxed']['normalized_objective']:.4g}; "
                     f"frac={lp['integer_plan']['extraction']['fractional_request_count']}")
        a = d["after"]
        lines.append(head + f" -> pre={d['preempted']} emit={d['emitted']}"
                     f" (free {a['central_free']}/{a['worker_free']})" + extra)
with open(f"{OUT}/comparison.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
