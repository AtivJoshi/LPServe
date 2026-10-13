"""Read-only post-hoc analysis of the contention decision trace.

Loads only decision_trace.json and summary.json (no engine, no GPU) and
re-derives facts the driver does not assert directly: rank/queue agreement,
decode demand versus the observed block gap, decode-demand groups per batch,
running-order preservation with tail admission, and the memory, watermark,
token, action, and resident rows at each decision. Exit 0 only if every
check holds.
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
trace = json.loads((out / "decision_trace.json").read_text())
summary = json.loads((out / "summary.json").read_text())
cfg = summary["config"]
B_MAX, C_MAX, S_MAX, N_MAX = cfg["b_max"], cfg["c_max"], cfg["s_max"], cfg["max_num_seqs"]
failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


rows = []
nonempty = 0
for st in trace["steps"]:
    tag = f"step {st['step']}"
    ev = st["events"]
    calls = [e["call"] for e in ev]
    check(calls.count("map_scheduler_state") == 1 and calls.count("solve_and_extract") == 1
          and calls.count("execute_plan") == 1, f"{tag}: map/solve/execute counts {calls}")
    nonempty += 1
    before, after = st["before"], st["after"]
    snap = ev[0]["result"]
    plan = ev[1]["result"]["plan"]
    sched = ev[3]["state"]
    emitted = ev[3]["outputs"]
    facts = {r["request_id"]: r for r in snap["requests"]}
    names = {r["seq_id"]: n for n, r in before["requests"].items()}

    # Ranks equal observed queue positions (running, then arrived waiting).
    check(snap["running_order"] == before["running"]
          and snap["arrived_waiting_order"] == before["waiting"],
          f"{tag}: snapshot queues differ from observed queues")
    for r in snap["requests"]:
        sid = r["raw_seq_id"]
        if sid in before["running"]:
            pos = before["running"].index(sid)
            check(r["service_rank"] == pos and r["victim_rank"] == len(before["running"]) - 1 - pos,
                  f"{tag}: ranks of {sid}")
        else:
            check(r["service_rank"] == len(before["running"]) + before["waiting"].index(sid)
                  and r["victim_rank"] is None, f"{tag}: ranks of waiting {sid}")
        obs = before["requests"][names[sid]]
        # Decode demand equals observed logical-minus-physical gap for residents.
        if r["decode_eligible"]:
            check(r["decode_demand"] == obs["logical_blocks"] - obs["physical_blocks"],
                  f"{tag}: decode demand of {sid} vs observed gap")
        if r["ownership"] == "waiting":
            # Full-context admission charge: ceil(context / block size).
            check(r["prefill_fixed_charge"] == -(-r["context_len"] // snap["block_size"]),
                  f"{tag}: admission charge of {sid}")
        else:
            check(r["prefill_fixed_charge"] == 0, f"{tag}: resident charge of {sid}")

    dec = {d["request_id"]: d for d in plan["decisions"]}
    victims = [rid for rid, d in dec.items() if d["preempt"]]
    prefills = [rid for rid, d in dec.items() if d["prefill_tokens"] > 0]
    decodes = [rid for rid, d in dec.items() if d["decode"]]
    admissions = [rid for rid in prefills if facts[rid]["ownership"] == "waiting"]
    zero = [rid for rid in decodes if facts[rid]["decode_demand"] == 0]
    one = [rid for rid in decodes if facts[rid]["decode_demand"] == 1]
    check(set(zero) | set(one) == set(decodes), f"{tag}: decode demand outside {{0,1}}")

    # Memory, watermark, decode-start gate (integer rows at the executed plan).
    m, h = snap["lp_problem"]["m_free"], snap["lp_problem"]["admission_watermark"]
    H = (m - sum(facts[r]["prefill_fixed_charge"] for r in admissions)
         + sum(facts[r]["preemption_recovery"] for r in victims))
    demand = sum(facts[r]["decode_demand"] for r in decodes)
    check(H >= demand, f"{tag}: memory H={H} < demand {demand}")
    check(not admissions or H >= h, f"{tag}: watermark H={H} < h={h}")
    check(not decodes or H >= 1, f"{tag}: decode-start gate H={H}")
    check(plan["free_blocks_before_decodes"] == H and plan["decode_demand"] == demand,
          f"{tag}: plan resource totals")
    check(sched["free_blocks"] == H - demand, f"{tag}: observed free after scheduling {sched['free_blocks']} != H - demand {H - demand}")
    # Tokens, actions, residents, chunks.
    tokens = sum(dec[r]["prefill_tokens"] for r in prefills) + len(decodes)
    check(tokens <= B_MAX and len(prefills) + len(decodes) <= S_MAX, f"{tag}: token/action")
    check(all(dec[r]["prefill_tokens"] <= min(C_MAX, facts[r]["prompt_tokens_remaining"]) for r in prefills),
          f"{tag}: chunk")
    resident_after = len(before["running"]) - len(victims) + len(admissions)
    check(resident_after <= N_MAX and resident_after == len(sched["running"])
          and plan["resident_count_after"] == resident_after, f"{tag}: residents")
    # Running order: retained residents keep order; admissions join the tail
    # in declared execution order.
    raw = {rid: facts[rid]["raw_seq_id"] for rid in facts}
    retained = [s for s in before["running"] if str(s) not in victims]
    adm_in_order = [raw[rid] for rid in plan["execution_order"] if rid in admissions]
    check(sched["running"] == retained + adm_in_order,
          f"{tag}: running {sched['running']} != retained {retained} + admitted {adm_in_order}")
    fin = summary["decisions"][st["step"] - 1]["finished"]
    fin_ids = [before["requests"][n]["seq_id"] for n in fin]
    check(after["running"] == [s for s in sched["running"] if s not in fin_ids],
          f"{tag}: running after completion")
    # Batch state.
    check(before["num_running_batches"] == 0 and sched["num_running_batches"] == 1
          and after["num_running_batches"] == 0, f"{tag}: batch state")
    # Emitted equals declared order and groups are contiguous in the right order.
    groups = [0 if r in prefills else 1 + facts[r]["decode_demand"] for r in plan["execution_order"]]
    check(groups == sorted(groups), f"{tag}: group order {groups}")
    check([s for s, _ in emitted["scheduled"]] == [raw[r] for r in plan["execution_order"]],
          f"{tag}: emitted order")

    def lab(rids):
        return [names[raw[r]] for r in rids]

    rows.append(dict(
        step=st["step"],
        running_before=[names[s] for s in before["running"]],
        waiting_before=[names[s] for s in before["waiting"]],
        execution_order=lab(plan["execution_order"]),
        prefills=lab(prefills), admissions=lab(admissions),
        zero_demand_decodes=lab(zero), one_block_decodes=lab(one),
        m_free=m, h=h, H=H, demand=demand, free_after_sched=sched["free_blocks"],
        running_after_sched=[names[s] for s in sched["running"]],
        relaxed_objective=ev[1]["result"]["relaxed"]["normalized_objective"],
        integer_objective=plan["objective"],
        extraction=plan["extraction"],
    ))

idle = trace["idle"]
check([e["call"] for e in idle["events"]] == ["schedule"], "idle: calls")

coverage = dict(
    nonempty_decisions=nonempty,
    steps_with_admission=[r["step"] for r in rows if r["admissions"]],
    steps_with_resident_prefill=[r["step"] for r in rows if set(r["prefills"]) - set(r["admissions"])],
    steps_with_zero_demand_decode=[r["step"] for r in rows if r["zero_demand_decodes"]],
    steps_with_one_block_decode=[r["step"] for r in rows if r["one_block_decodes"]],
    steps_with_both_decode_groups_in_one_batch=[r["step"] for r in rows if r["zero_demand_decodes"] and r["one_block_decodes"]],
    steps_with_prefill_and_decode_in_one_batch=[r["step"] for r in rows if r["prefills"] and (r["zero_demand_decodes"] or r["one_block_decodes"])],
    steps_where_emitted_order_differs_from_running_order=[
        r["step"] for r in rows
        if [n for n in r["running_after_sched"] if n in r["execution_order"]] != r["execution_order"]],
    steps_with_preemption=[],
    watermark_h=rows[0]["h"],
    min_H_minus_h_at_admission=min((r["H"] - r["h"] for r in rows if r["admissions"]), default=None),
)
print(json.dumps(dict(rows=rows, coverage=coverage, failures=failures), indent=1))
print("ANALYSIS:", "PASS" if not failures else f"FAIL ({len(failures)})")
sys.exit(1 if failures else 0)
