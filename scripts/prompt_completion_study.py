"""Offline same-state study of finishing a nearly complete prompt.

Reads retained soft-SLO pilot trials (``scripts/soft_slo_pilot.py``) and
selects every saved timed-policy decision in which ``tight33`` waited with
one prompt token remaining and received no action. Each snapshot is rebuilt
from its JSON record through the live mapping path (captured state, saved
ranks, saved timing inputs, pilot coefficient policy). For each one:

1. the rebuilt timed problem must equal the recorded ``LPProblem`` and its
   PH1 plan must reproduce the recorded actions and objective (bonus off);
2. the current exact ILP is solved as written, then again with tight33
   required to finish its prompt (its prefill indicator fixed to 1 and its
   chunk to the remainder); the objective difference is the cost of
   finishing, and the displaced work is the set of baseline actions the
   forced optimum no longer takes;
3. the experimental bonus problem (``timed_bonus``, ``COMPLETION_BONUS``) is
   solved with the exact ILP and with PH1.

"Ready to decode" means the forced plan leaves no prompt remainder, the
request stays resident, and at least one free block remains for the decode
prerequisite at the next decision; it does not mean the next decision gives
it a decode slot, which only a live trace shows.

    PYTHONPATH="$PWD:$PWD/scripts" python -B scripts/prompt_completion_study.py \\
        --evidence <registration-order evidence dir> --output <json>
"""

import argparse
import json
import sys
from pathlib import Path
from types import MappingProxyType

import exact_integer_plan as eip
import soft_slo_pilot as pilot
from check_lp_scheduler_gpu import check, to_jsonable

import lp_relaxation_scheduler as lrs
import lpserve_state_mapping as lsm

TARGET = "tight33"
ORDERS = ("relaxed_first", "tight_first")


def captured_from_record(snap):
    """``CapturedState``, ranks, and numerical policy of a snapshot record."""
    req_fields = lsm.CapturedRequest.__dataclass_fields__
    requests = tuple(
        lsm.CapturedRequest(**{k: (tuple(v) if k == "order_key" else v)
                               for k, v in r.items() if k in req_fields})
        for r in snap["requests"])
    fields = {k: (tuple(v) if k in ("running_order", "arrived_waiting_order")
                  else v)
              for k, v in snap.items()
              if k in lsm.CapturedState.__dataclass_fields__
              and k != "requests"}
    ranks = tuple((r["service_rank"], r["victim_rank"])
                  for r in snap["requests"])
    return (lsm.CapturedState(requests=requests, **fields), ranks,
            lrs.NumericalPolicy(**snap["numerical_policy"]))


def timing_from_record(record):
    return MappingProxyType({int(k): lsm.RequestTiming(**v)
                             for k, v in record.items()})


def rebuild(captured, ranks, numerical, policy, case, timing):
    coefficients = lsm.construct_coefficients(
        captured, pilot.coefficient_policy(policy, case, timing))
    check(not isinstance(coefficients, lsm.MappingFailure),
          f"{policy} coefficients: {coefficients}")
    snap = lsm.build_state_snapshot(
        captured, ranks=ranks, coefficients=coefficients,
        b_max=case["b_max"], c_max=case["c_max"], s_max=case["s_max"],
        numerical_policy=numerical)
    check(isinstance(snap, lsm.StateSnapshot), f"{policy} problem: {snap}")
    return snap.lp_problem


def action_values(problem, plan):
    """``{raw_id: (action, objective contribution)}`` of a plan."""
    out = {}
    for r, d in zip(problem.requests, plan.decisions):
        rid = int(r.request_id)
        if d.prefill_tokens:
            value = r.prefill_token_utility * d.prefill_tokens
            if lrs._completion_bonus(r) > 0 and \
                    d.prefill_tokens == r.prompt_tokens_remaining:
                value += lrs._completion_bonus(r)
            out[rid] = (f"prefill:{d.prefill_tokens}", value)
        elif d.decode:
            out[rid] = ("decode", r.decode_utility)
        elif d.preempt:
            out[rid] = ("preempt", -r.preemption_penalty)
    return out


def exact(problem, **kwargs):
    result = eip.solve_integer_plan(problem, **kwargs)
    check(isinstance(result, eip.IntegerOptimum),
          f"exact solve failed: {to_jsonable(result)}")
    return result.plan


def study_snapshot(step, cohort, case):
    names = {c["seq_id"]: c["name"] for c in cohort}
    target = next(c["seq_id"] for c in cohort if c["name"] == TARGET)
    snap = step["snapshots"][0]
    captured, ranks, numerical = captured_from_record(snap)
    timing = timing_from_record(step["timing_inputs"][0])
    recorded = eip.problem_from_record(snap["lp_problem"])
    timed = rebuild(captured, ranks, numerical, "timed", case, timing)
    check(timed == recorded, "rebuilt timed problem differs from the record")
    ph1 = lrs.solve_and_extract(timed)
    check(isinstance(ph1, lrs.SchedulingSuccess), "PH1 on the rebuilt state")
    plan_record = step["results"][0]["plan"]
    check(eip.plan_actions(ph1.plan) == eip.plan_actions(plan_record)
          and ph1.plan.objective == plan_record["objective"],
          "PH1 does not reproduce the recorded plan")

    canonical = lrs.validate_problem(timed)
    n = len(canonical.requests)
    i = next(k for k, r in enumerate(canonical.requests)
             if int(r.request_id) == target)
    r = canonical.requests[i]
    check(r.prompt_tokens_remaining == 1 and r.prefill_eligible,
          "target is not waiting with one prompt token")
    base = exact(timed)
    forced = exact(timed, bound_overrides={
        i: (float(r.prompt_tokens_remaining), float(r.prefill_upper_bound)),
        2 * n + i: (1.0, 1.0)})
    base_actions = action_values(canonical, base)
    forced_actions = action_values(canonical, forced)
    displaced = {names[k]: dict(action=v[0], value=v[1])
                 for k, v in base_actions.items()
                 if forced_actions.get(k, (None,))[0] != v[0]}
    added = {names[k]: dict(action=v[0], value=v[1])
             for k, v in forced_actions.items()
             if base_actions.get(k, (None,))[0] != v[0]}
    forced_decision = forced.decisions[i]
    free_after = forced.free_blocks_before_decodes - forced.decode_demand
    ready = (r.prompt_tokens_remaining - forced_decision.prefill_tokens == 0
             and r.resident and free_after >= 1)

    bonus_problem = rebuild(captured, ranks, numerical, pilot.BONUS_POLICY,
                            case, timing)
    bonus_canonical = lrs.validate_problem(bonus_problem)
    bonus_exact = exact(bonus_problem)
    bonus_ph1 = lrs.solve_and_extract(bonus_problem)
    check(isinstance(bonus_ph1, lrs.SchedulingSuccess),
          f"PH1 with bonus failed: {to_jsonable(bonus_ph1)}")
    label = lambda acts: {names[k]: v[0] for k, v in sorted(acts.items())}  # noqa: E731
    bonus_exact_actions = action_values(bonus_canonical, bonus_exact)
    bonus_ph1_actions = action_values(bonus_canonical, bonus_ph1.plan)
    finishes = lambda acts: acts.get(target, (None,))[0] == "prefill:1"  # noqa: E731
    return dict(
        decision=step["index"], snapshot_time=snap["snapshot_time"],
        target_boost=lsm.soft_slo_forecast(
            captured, next(q for q in captured.requests
                           if q.raw_seq_id == target),
            timing=timing, batch_forecast=pilot.BATCH_FORECAST_SECONDS,
            b_max=case["b_max"], c_max=case["c_max"]).boost,
        reproduces_recorded=True,
        ph1_actions=label(action_values(canonical, ph1.plan)),
        exact_actions=label(base_actions), exact_objective=base.objective,
        exact_leaves_target_waiting=target not in base_actions,
        forced_actions=label(forced_actions),
        forced_objective=forced.objective,
        finishing_cost=base.objective - forced.objective,
        displaced_work=displaced, added_work=added,
        ready_to_decode_next=ready, free_blocks_after_forced=free_after,
        target_prefill_token_utility=r.prefill_token_utility,
        bonus_offered={names[int(q.request_id)]: lrs._completion_bonus(q)
                       for q in bonus_canonical.requests
                       if lrs._completion_bonus(q) > 0},
        bonus_exact_actions=label(bonus_exact_actions),
        bonus_exact_objective=bonus_exact.objective,
        bonus_exact_finishes_target=finishes(bonus_exact_actions),
        bonus_ph1_actions=label(bonus_ph1_actions),
        bonus_ph1_objective=bonus_ph1.plan.objective,
        bonus_ph1_finishes_target=finishes(bonus_ph1_actions),
        bonus_flips_exact=(finishes(bonus_exact_actions)
                           and target not in base_actions))


def replay_disabled(evidence):
    """Every recorded GPU decision (both orders, every trial and policy):
    the recorded problem (bonus field absent, so 0) must reproduce the
    recorded PH1 actions and objective exactly, and the recorded
    same-state exact objectives must be reproduced."""
    decisions = exact_rows = 0
    for order in ORDERS:
        run_dir = Path(evidence) / f"gpu_ample_{order}"
        for path in sorted((run_dir / "trials").glob("*.json")):
            trial = json.loads(path.read_text())
            for step in trial["steps"]:
                problem = eip.problem_from_record(
                    step["snapshots"][0]["lp_problem"])
                check(all(type(r) is lrs.RequestInput
                          for r in problem.requests),
                      "recorded problem carries a bonus")
                result = lrs.solve_and_extract(problem)
                plan = step["results"][0]["plan"]
                check(isinstance(result, lrs.SchedulingSuccess)
                      and eip.plan_actions(result.plan)
                      == eip.plan_actions(plan)
                      and result.plan.objective == plan["objective"],
                      f"{order} {trial['label']} decision {step['index']} "
                      "does not reproduce")
                decisions += 1
        summary = json.loads((run_dir / "summary.json").read_text())
        first = next(json.loads(p.read_text()) for p in sorted(
            (run_dir / "trials").glob("trial0_*.json")))
        for row in summary["same_state_exact"]:
            step = first["steps"][row["decision"]]
            snap = step["snapshots"][0]
            captured, ranks, numerical = captured_from_record(snap)
            timing = timing_from_record(step["timing_inputs"][0])
            for policy in ("timed", "untimed"):
                problem = rebuild(captured, ranks, numerical, policy,
                                  pilot.CASES["ample"], timing)
                recorded = row[policy]["comparison"]["integer_objective"]
                check(abs(exact(problem).objective - recorded) <= 1e-9,
                      f"{order} decision {row['decision']} {policy}: exact "
                      "objective differs from the record")
                exact_rows += 1
    return dict(ph1_decisions_reproduced=decisions,
                exact_objectives_reproduced=exact_rows)


def run(evidence):
    case = pilot.CASES["ample"]
    rows = []
    for order in ORDERS:
        run_dir = Path(evidence) / f"gpu_ample_{order}"
        for path in sorted((run_dir / "trials").glob("*_timed.json")):
            trial = json.loads(path.read_text())
            target = next(c["seq_id"] for c in trial["cohort"]
                          if c["name"] == TARGET)
            for step in trial["steps"]:
                snap = step["snapshots"][0]
                req = next((q for q in snap["requests"]
                            if q["raw_seq_id"] == target), None)
                scheduled = {sid for sid, _ in step["scheduled"]}
                if (req is None or req["prompt_tokens_remaining"] != 1
                        or target in scheduled):
                    continue
                row = study_snapshot(step, trial["cohort"], case)
                row.update(order=order, trial=trial["label"])
                rows.append(row)
    flips = sum(r["bonus_flips_exact"] for r in rows)
    costs = [r["finishing_cost"] for r in rows]
    return dict(
        evidence=str(evidence), target=TARGET,
        disabled_replay=replay_disabled(evidence),
        completion_bonus=pilot.COMPLETION_BONUS,
        snapshots=len(rows), exact_flips=flips,
        ph1_finishes_with_bonus=sum(r["bonus_ph1_finishes_target"]
                                    for r in rows),
        finishing_cost_min=min(costs) if costs else None,
        finishing_cost_max=max(costs) if costs else None,
        proceed_to_live=flips >= 2, rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output)
    check(not out.exists(), f"{out} exists")
    result = run(args.evidence)
    out.write_text(json.dumps(to_jsonable(result), indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
