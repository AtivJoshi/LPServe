"""Offline replay of the documented whole-action failure cases.

CPU only: no engine, model, scheduler, or GPU. The examples come from
retained comparison evidence under ``validation_output/scheduler_comparison``
(``EXAMPLES``). For each listed decision, a fixture is written holding the
starting state, the saved mapped problem (with its coefficients), the saved
LP result or integer result, the saved same-state integer comparison when
one exists, the executed outputs, the victim causes, and the resulting
state. Sources are recorded with their sha256.

Replay reads only the fixture files. Each saved problem is reloaded,
validated, and checked to serialize back to the same record. It then
receives one ``solve_and_extract`` (the current LP and whole-action
extraction) and one ``exact_integer_plan.solve_integer_plan``. These are
compared with the saved results: the LP plan and relaxed vector exactly,
integer actions exactly and objectives within
``exact_integer_plan.COMPARISON_TOL``. An integer result that was not
retained is labelled as a new solve of the original problem; an LP result
that was not retained (integer-driven rollouts) is labelled as offline only.
The executed outputs must equal the rollout's own plan. Any mismatch, failed
validation, or non-optimal solve makes the result FAIL. Nothing is rounded,
retried, or invented.

Usage::

    python scripts/replay_failure_cases.py --out <dir>
    python scripts/replay_failure_cases.py --out <dir> --fixtures <dir>

The first form writes ``<dir>/fixtures/`` from the retained evidence and
replays them; the second replays existing fixtures only. Both write
``replay.json`` and ``report.txt``.
"""

import argparse
import dataclasses
import enum
import gzip
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy  # noqa: E402
import scipy  # noqa: E402

import exact_integer_plan as eip  # noqa: E402
import lp_relaxation_scheduler as lrs  # noqa: E402

EVIDENCE = REPO_ROOT / "validation_output" / "scheduler_comparison"
CONTEXT192 = "20261024T213405Z_queue_growth_context192/cpu_comparison"
RUNNING_ORDER = "20261026T045354Z_running_order/cpu_comparison"
COORDINATED = "20261024T182347Z/cpu_comparison"

# (example, label, retained source, decisions). A rollout source is a run
# directory; a diagnostic source is "<diagnostics file>#<state name>".
EXAMPLES = (
    ("fractional_preemption", "reciprocal-square, queue48_burst_pool12",
     f"{CONTEXT192}/rollouts/queue48_burst_pool12/lp_reciprocal_square",
     (20, 34)),
    ("fractional_preemption", "running order, queue48_burst_pool12",
     f"{RUNNING_ORDER}/rollouts/queue48_burst_pool12_limit8/lp_running_order",
     (2,)),
    ("fractional_admission", "running order, fractional_admission state",
     f"{RUNNING_ORDER}/diagnostics.json#fractional_admission", (None,)),
    ("fractional_admission", "reciprocal-square, queue48_burst",
     f"{RUNNING_ORDER}/rollouts/queue48_burst_limit8/lp_reciprocal_square",
     (0,)),
    ("admit_then_evict", "running-order integer optimum, queue96_burst_pool12",
     f"{RUNNING_ORDER}/rollouts/queue96_burst_pool12_limit8/ilp_running_order",
     (16, 17, 55, 56)),
    ("repeated_prefill", "uniform, two_request, resident limit 1",
     f"{COORDINATED}/rollouts/two_request_limit1/lp_uniform", tuple(range(8))),
    ("repeated_prefill", "rank-based, two_request, resident limit 1",
     f"{COORDINATED}/rollouts/two_request_limit1/lp_rank_based",
     tuple(range(8))),
)


def jsonable(obj):
    """JSON form of a result, as the comparison drivers write it."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, (frozenset, set)):
        return sorted((jsonable(x) for x in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def first_existing(run_dir, names):
    for name in names:
        if (run_dir / name).exists():
            return run_dir / name
    return None


def state_summary(state):
    """Queues, free blocks, block tables, and per-request progress of a
    retained central state (token IDs omitted)."""
    return dict(
        total_blocks=state["total_blocks"], free_blocks=state["free_blocks"],
        watermark_blocks=state["watermark_blocks"],
        running=[str(s) for s in state["running"]],
        waiting=[str(s) for s in state["waiting"]],
        block_tables={str(k): v for k, v in state["block_tables"].items()},
        requests={str(k): dict(status=s["status"], prompt_len=s["prompt_len"],
                               prompt_processed=s["prompt_tokens_processed"],
                               outputs=len(s["output_token_ids"]))
                  for k, s in state["seqs"].items()})


def output_actions(outputs):
    """``{request_id: "prefill:<n>" | "decode" | "preempt"}`` of retained
    ``SchedulerOutputs`` (a scheduled chunk of 0 is a decode)."""
    actions = {str(s): "preempt" for s in outputs["preempted_seq_ids"]}
    for seq_id, chunk in outputs["scheduled"]:
        actions[str(seq_id)] = f"prefill:{chunk}" if chunk else "decode"
    return actions


def rollout_fixtures(example, label, source, decisions):
    run_dir = EVIDENCE / source
    trace_path = first_existing(run_dir, ("decision_trace.json.gz",
                                          "decision_trace.json"))
    comparisons_path = first_existing(run_dir, ("comparisons.json.gz",
                                                "comparisons.json"))
    trace = load_json(trace_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    rollout = dict(summary=str((run_dir / "summary.json").relative_to(
                       REPO_ROOT)),
                   summary_sha256=sha256(run_dir / "summary.json"),
                   **{k: summary[k] for k in (
                       "workload_outcome", "decisions", "finished",
                       "unfinished", "preemptions",
                       "prefill_tokens_replayed")})
    comparisons = ({c["index"]: c for c in load_json(comparisons_path)}
                   if comparisons_path else {})
    out = []
    for index in decisions:
        d = trace["decisions"][index]
        if d["index"] != index:
            raise ValueError(f"{trace_path}: decision {index} out of order")
        events = {e["call"]: e for e in d["lp_events"]}
        snapshot = events["map_scheduler_state"]
        if snapshot["result_type"] != "StateSnapshot":
            raise ValueError(f"{trace_path} decision {index}: no problem")
        lp = events.get("solve_and_extract")
        solved = events.get("solve_integer_plan")
        comparison = comparisons.get(index, {})
        if solved is not None:
            integer = eip.integer_summary(solved["result_type"],
                                          solved["result"])
            integer_source = "rollout's own integer solve"
        elif "integer" in comparison:
            integer = comparison["integer"]
            integer_source = "retained offline same-state comparison"
        else:
            integer, integer_source = None, None
        out.append(dict(
            example=example, label=label, kind="rollout",
            source=dict(trace=str(trace_path.relative_to(REPO_ROOT)),
                        trace_sha256=sha256(trace_path),
                        comparisons=(str(comparisons_path.relative_to(
                            REPO_ROOT)) if comparisons_path else None),
                        comparisons_sha256=(sha256(comparisons_path)
                                            if comparisons_path else None)),
            case=trace["case"], policy=trace["policy"],
            resident_limit=trace["resident_limit"], decision=index,
            rollout=rollout, before=state_summary(d["before"]),
            problem=snapshot["result"]["lp_problem"],
            executed_plan="integer" if solved is not None else "lp",
            retained_lp=(dict(result_type=lp["result_type"],
                              result=lp["result"]) if lp else None),
            retained_integer=integer, retained_integer_source=integer_source,
            victim_need=d.get("victim_need"),
            executed=events["execute_plan"]["result"],
            after=state_summary(d["after"])))
    return out


def diagnostic_fixture(example, label, source):
    path, name = source.split("#")
    path = EVIDENCE / path
    record = next(s for s in load_json(path) if s["name"] == name)
    counts = record["block_counts"]["lp"]
    return dict(
        example=example, label=label, kind="diagnostic",
        source=dict(trace=f"{path.relative_to(REPO_ROOT)}#{name}",
                    trace_sha256=sha256(path)),
        case=name, policy=None, resident_limit=record["spec"]["limit"],
        decision=None, rollout=None,
        before=dict(free_blocks=counts["before"]["free"],
                    spec=record["spec"], requests=record["mapped_requests"]),
        problem=record["problem"], executed_plan="lp",
        retained_lp=dict(result_type="SchedulingSuccess",
                         result=record["lp"]),
        retained_integer=eip.integer_summary("IntegerOptimum",
                                             record["integer"]),
        retained_integer_source="retained diagnostic integer solve",
        victim_need=None, executed=record["executed"]["lp"]["outputs"],
        after=dict(free_blocks=counts["after"]["free"],
                   physical_blocks=counts["after"]["physical"]))


def build_fixtures(fixture_dir):
    fixture_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, (example, label, source, decisions) in enumerate(EXAMPLES):
        fixtures = (rollout_fixtures(example, label, source, decisions)
                    if "#" not in source
                    else [diagnostic_fixture(example, label, source)])
        for f in fixtures:
            suffix = "" if f["decision"] is None else f"_d{f['decision']:03d}"
            path = fixture_dir / f"{n:02d}_{example}_{f['case']}{suffix}.json"
            path.write_text(json.dumps(f, indent=1))
            paths.append(path)
    return paths


def relaxed_values(lp_record):
    """Nonzero relaxed values ``{request_id: {x, y, I, z}}``."""
    out = {}
    for d in lp_record["relaxed"]["decisions"]:
        values = dict(x=d["x"], y=d["y"], I=d["prefill_indicator"], z=d["z"])
        values = {k: v for k, v in values.items() if abs(v) > 0}
        if values:
            out[d["request_id"]] = values
    return out


def replay(fixture):
    """Replay one fixture; every check and mismatch is recorded."""
    result = dict(example=fixture["example"], label=fixture["label"],
                  case=fixture["case"], decision=fixture["decision"],
                  mismatches=[])
    mismatches = result["mismatches"]
    problem = lrs.validate_problem(eip.problem_from_record(fixture["problem"]))
    if isinstance(problem, lrs.Failure):
        mismatches.append(f"saved problem failed validation: {problem.reason}")
        result["problem_status"] = jsonable(problem)
        return result
    result["round_trip_equal"] = jsonable(problem) == fixture["problem"]
    if not result["round_trip_equal"]:
        mismatches.append("saved problem does not serialize back unchanged")

    lp = lrs.solve_and_extract(problem)
    lp_type = type(lp).__name__
    lp_record = jsonable(lp)
    result["lp_status"] = lp_type
    if lp_type != "SchedulingSuccess":
        mismatches.append(f"LP solve did not succeed: {lp_record}")
        lp_summary = None
    else:
        lp_summary = eip.lp_summary(lp_type, lp)
        result.update(relaxed=relaxed_values(lp_record),
                      lp_solver=dict(
                          raw_status=lp.solver.diagnostics.raw_status,
                          message=lp.solver.diagnostics.message,
                          method=lp.solver.diagnostics.method,
                          options=jsonable(lp.solver.diagnostics.options)),
                      lp_summary=lp_summary)
    retained_lp = fixture["retained_lp"]
    if retained_lp is None:
        result["lp_source"] = "offline only; the rollout did not solve the LP"
    elif lp_summary is not None:
        saved = retained_lp["result"]
        result["lp_source"] = "retained and reproduced"
        result["lp_plan_equal"] = lp_record["plan"] == saved["plan"]
        result["relaxed_vector_equal"] = (lp_record["relaxed"]["raw_vector"]
                                          == saved["relaxed"]["raw_vector"])
        result["relaxed_vector_max_difference"] = max(
            abs(a - b) for a, b in zip(lp_record["relaxed"]["raw_vector"],
                                       saved["relaxed"]["raw_vector"]))
        for key in ("lp_plan_equal", "relaxed_vector_equal"):
            if not result[key]:
                mismatches.append(f"{key} is false")

    integer = eip.solve_integer_plan(problem)
    integer_summary = eip.integer_summary(type(integer).__name__, integer)
    result["integer"] = integer_summary
    if integer_summary["status"] != "optimal":
        mismatches.append(f"integer solve not optimal: {integer_summary}")
    retained_integer = fixture["retained_integer"]
    if retained_integer is None:
        result["integer_source"] = ("not retained; new solve of the saved "
                                    "problem (original state and "
                                    "coefficients)")
    else:
        result["integer_source"] = (f"{fixture['retained_integer_source']}; "
                                    "reproduced")
        same = (retained_integer["status"] == integer_summary["status"]
                and retained_integer.get("actions")
                == integer_summary.get("actions")
                and abs(retained_integer["objective"]
                        - integer_summary["objective"])
                <= eip.COMPARISON_TOL)
        result["integer_equal"] = same
        if not same:
            mismatches.append("integer result differs from the retained one")
    if lp_summary is not None and integer_summary["status"] == "optimal":
        result["comparison"] = eip.compare_summaries(
            problem.problem_id, lp_type, lp_summary, "optimal",
            integer_summary)

    executed = output_actions(fixture["executed"])
    result["executed_actions"] = executed
    planned = (integer_summary if fixture["executed_plan"] == "integer"
               else lp_summary) or {}
    result["executed_equals_plan"] = executed == planned.get("actions")
    if not result["executed_equals_plan"]:
        mismatches.append(f"executed actions differ from the "
                          f"{fixture['executed_plan']} plan")
    result["executed_evaluation"] = eip.evaluate_assignment(problem, executed)
    return result


def _num(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _actions(actions):
    if not actions:
        return "nothing"
    return ", ".join(f"R{int(r):02d} {a}" for r, a in sorted(
        actions.items(), key=lambda kv: int(kv[0])))


def _shown_requests(fixture, result):
    """Residents, every request with a relaxed or integer action, and the
    first three waiting requests, in problem order."""
    shown = {r["request_id"] for r in fixture["problem"]["requests"]
             if r["resident"]}
    shown |= set(result.get("relaxed", {}))
    for actions in ((result.get("lp_summary") or {}).get("actions"),
                    result["integer"].get("actions"),
                    result["executed_actions"]):
        shown |= set(actions or {})
    shown |= set(fixture["before"].get("waiting", [])[:3])
    return [r for r in fixture["problem"]["requests"]
            if r["request_id"] in shown]


def fixture_text(fixture, result):
    p, before, after = fixture["problem"], fixture["before"], fixture["after"]
    at = ("diagnostic state" if fixture["decision"] is None
          else f"decision {fixture['decision']}")
    lines = [f"--- {fixture['label']}: {at}",
             f"source: {fixture['source']['trace']}",
             f"start: free blocks {p['m_free']}, watermark "
             f"{p['admission_watermark']}, residents {p['resident_count']}/"
             f"{p['resident_limit']}, b_max {p['b_max']}, c_max "
             f"{p['c_max']}, s_max {p['s_max']}"]
    if "running" in before:
        lines.append(f"  running {before['running']}; waiting "
                     f"{len(before['waiting'])} (head {before['waiting'][:6]})")
        lines.append(f"  block tables {before['block_tables']}")
    lines.append("  request res left out blocks dem charge decode_u "
                 "prefill_u/token penalty")
    for r in _shown_requests(fixture, result):
        rid = r["request_id"]
        progress = before.get("requests", {}).get(rid, {})
        blocks = len(before.get("block_tables", {}).get(rid, [])) \
            if "block_tables" in before else progress.get("physical_blocks")
        lines.append(
            f"  R{int(rid):02d} {'yes' if r['resident'] else 'no ':>5} "
            f"{r['prompt_tokens_remaining']:>4} "
            f"{progress.get('outputs', progress.get('output_len', '-')):>3} "
            f"{_num(blocks):>6} {r['decode_demand']:>3} "
            f"{r['prefill_fixed_charge']:>6} {_num(r['decode_utility']):>8} "
            f"{_num(r['prefill_token_utility']):>15} "
            f"{_num(r['preemption_penalty'])}")
    lp = result.get("lp_summary") or {}
    if lp:
        relaxed = "; ".join(
            f"R{int(r):02d} " + " ".join(f"{k}={_num(v)}" for k, v in
                                         values.items())
            for r, values in sorted(result["relaxed"].items(),
                                    key=lambda kv: int(kv[0])))
        lines.append(f"LP proposal (objective {_num(lp['relaxed_objective'])}"
                     f"): {relaxed}")
        lines.append(f"whole-action plan (utility {_num(lp['plan_objective'])}"
                     f"; {result['lp_source']}): {_actions(lp['actions'])}")
    integer = result["integer"]
    lines.append(f"integer optimum (utility {_num(integer.get('objective'))}; "
                 f"{result['integer_source']}): "
                 f"{_actions(integer.get('actions'))}")
    comparison = result.get("comparison")
    if comparison:
        lines.append(f"relaxation gap {_num(comparison['relaxation_gap'])}, "
                     f"extraction loss {_num(comparison['extraction_loss'])}")
    lines.append(f"executed ({fixture['executed_plan']} plan, utility "
                 f"{_num(result['executed_evaluation']['objective'])}): "
                 f"{_actions(result['executed_actions'])}")
    if fixture.get("victim_need"):
        lines.append(f"victim causes: {fixture['victim_need']['reasons']}")
    if "running" in after:
        lines.append(f"after: free blocks {after['free_blocks']}; running "
                     f"{after['running']}; waiting {len(after['waiting'])} "
                     f"(head {after['waiting'][:6]})")
        lines.append(f"  block tables {after['block_tables']}")
        changed = {rid: s for rid, s in after["requests"].items()
                   if s != before["requests"].get(rid)}
        lines.append("  progress changed: " + "; ".join(
            f"R{int(r):02d} {s['status']} prompt {s['prompt_processed']}/"
            f"{s['prompt_len']} outputs {s['outputs']}"
            for r, s in sorted(changed.items(), key=lambda kv: int(kv[0]))))
    else:
        lines.append(f"after: free blocks {after['free_blocks']}; physical "
                     f"blocks {after['physical_blocks']}")
    rollout = fixture["rollout"]
    if rollout and fixture["decision"] == rollout["decisions"] - 1:
        lines.append(f"rollout end: {rollout['workload_outcome']} after "
                     f"{rollout['decisions']} decisions; finished "
                     f"{rollout['finished']}, unfinished "
                     f"{rollout['unfinished']}; preemptions "
                     f"{rollout['preemptions']}; replayed prefill tokens "
                     f"{rollout['prefill_tokens_replayed']}")
    lines.append("replay: " + ("all checks equal" if not result["mismatches"]
                               else "MISMATCH " + "; ".join(
                                   result["mismatches"])))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path,
                        help="replay these fixtures instead of building them")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fixture_dir = args.fixtures or args.out / "fixtures"
    if args.fixtures is None:
        build_fixtures(fixture_dir)
    paths = sorted(fixture_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no fixtures in {fixture_dir}")
    fixtures = [json.loads(p.read_text()) for p in paths]
    results = [replay(f) for f in fixtures]
    failed = [r for r in results if r["mismatches"]]
    status = "FAIL" if failed else "PASS"
    record = dict(
        status=status, fixtures=[str(p) for p in paths],
        fixture_sha256={p.name: sha256(p) for p in paths},
        environment=dict(python=sys.version, platform=platform.platform(),
                         numpy=numpy.__version__, scipy=scipy.__version__),
        integer_solver=dict(solver=eip.SOLVER,
                            options=jsonable(eip.SOLVER_OPTIONS),
                            comparison_tol=eip.COMPARISON_TOL),
        results=results)
    (args.out / "replay.json").write_text(json.dumps(record, indent=1))
    lines = [f"Failure-case replay: {len(results)} decisions, "
             f"{len(failed)} with mismatches",
             f"numpy {numpy.__version__}, scipy {scipy.__version__}", ""]
    example = None
    for fixture, result in zip(fixtures, results):
        if fixture["example"] != example:
            example = fixture["example"]
            lines += ["", f"===== {example}"]
        lines += fixture_text(fixture, result) + [""]
    lines.append(f"RESULT: {status}")
    (args.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
