"""CPU comparison batch: adapted vLLM reference, uniform / rank-based /
coordinated LP, and the coordinated integer optimum.

No engine, model, Ray worker, or GPU is used. Every closed-loop run goes
through ``compare_scheduler_gpu.run_comparison`` with the native CPU runtime
of ``tests/test_compare_scheduler_gpu.py`` (registered scheduler, central
block manager, engine and worker sequence managers, synthetic per-request
samples), so every decision passes the driver's policy-specific checks.

Sections written under ``--output-dir``:

- ``rollouts/<case>_limit<n>/<method>/``: ``summary.json`` and
  ``decision_trace.json`` for each workload and method. States diverge
  between methods after their first different decision.
- ``same_state/``: for each coordinated-LP decision, the integer optimum of
  the identical ``LPProblem`` and the LP's relaxed optimum and returned plan
  (``lp_coordinated_*.json``); for each reference decision, the coordinated
  problem of its before-state, its integer optimum, and the reference
  proposal's utility and LP-contract feasibility (``adapted_vllm_*.json``).
  Diagnostic solves run after the rollout, outside its decision path.
- ``diagnostic_states.json``: four single constructed states (native
  allocation and progress methods), each mapped once per method, with the LP
  plan and integer plan executed by the native executor on separate copies,
  the reference proposal, and bounded enumeration of the integer optimum.
- ``comparison.txt``: the summary tables; ``run_record.json``: provenance.

Exit status is zero only if every rollout and diagnostic passed its checks;
workload outcomes are reported separately.
"""

import argparse
import contextlib
import functools
import itertools
import json
import math
import sys
import time
import traceback
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_scheduler_gpu as driver  # noqa: E402
import exact_integer_plan as eip  # noqa: E402

METHODS = ("adapted_vllm", "lp_uniform", "lp_rank_based", "lp_coordinated",
           driver.INTEGER_POLICY)
WORKLOADS = ((driver.TWO_REQUEST, 1), (driver.TWO_REQUEST, 2),
             (driver.MEMORY_PRESSURE, 4)) + tuple(
                 (case, 4) for case in driver.REPRESENTATIVE_CASES)
ENUMERATION_LIMIT = 50000
# Documented SciPy/HiGHS default dual feasibility tolerance (not changed).
HIGHS_DEFAULT_DUAL_FEASIBILITY_TOL = 1e-7


def _numerical_policy():
    import lp_relaxation_scheduler as lrs
    return lrs.NumericalPolicy(**driver.NUMERICAL_POLICY_FIELDS)


def coordinated_policy(c_max, b_max):
    import lpserve_state_mapping as lsm
    return functools.partial(lsm.coordinated_rank_utility, c_max=c_max,
                             b_max=b_max, **driver.COORDINATED)


def coordinated_problem(scheduler, b_max, c_max, s_max):
    """Read-only coordinated mapping of the scheduler's current state."""
    import lpserve_state_mapping as lsm
    snap = lsm.map_scheduler_state(
        scheduler, snapshot_time=time.monotonic(), b_max=b_max, c_max=c_max,
        s_max=s_max, coefficient_policy=coordinated_policy(c_max, b_max),
        rank_profile=lsm.vllm_oriented_ranks,
        numerical_policy=_numerical_policy())
    return snap if isinstance(snap, lsm.MappingFailure) else snap.lp_problem


def reference_assignment(proposal):
    """A reference proposal record as ``{request_id: action}``."""
    out = {str(v): "preempt" for v in proposal["victims"]}
    for a in proposal["actions"]:
        out[str(a["seq_id"])] = ("decode" if a["kind"] == driver.DECODE
                                 else f"prefill:{a['num_tokens']}")
    return out


def enumeration_check(problem):
    """Bounded enumeration of the integer optimum (test oracle), or None
    when the plan space exceeds ``ENUMERATION_LIMIT``."""
    import test_lp_relaxation_scheduler as planner_tests
    size = math.prod(len(planner_tests.oracle_options(r))
                     for r in problem.requests)
    if size > ENUMERATION_LIMIT:
        return dict(skipped=f"plan space {size} exceeds {ENUMERATION_LIMIT}")
    exists, best = planner_tests.oracle_best(problem)
    return dict(plan_space=size, useful_plan_exists=exists, best=best)


def compare_reference(problem, proposal, integer_result):
    """Classify the reference proposal against the integer optimum of the
    coordinated problem of the same state."""
    assignment = reference_assignment(proposal)
    evaluation = eip.evaluate_assignment(problem, assignment)
    record = dict(problem_id=problem.problem_id, reference_actions=assignment,
                  reference_evaluation=evaluation,
                  integer_status=type(integer_result).__name__)
    if not isinstance(integer_result, eip.IntegerOptimum):
        record["classification"] = "integer_solve_failed"
        record["integer_failure"] = dict(stage=integer_result.stage,
                                         category=integer_result.category,
                                         reason=integer_result.reason)
        return record
    best = integer_result.plan.objective
    actions = eip.plan_actions(integer_result.plan)
    tol = eip.COMPARISON_TOL * max(1.0, abs(best), abs(evaluation["objective"]))
    record.update(integer_objective=best, integer_actions=actions)
    if not evaluation["feasible"]:
        record["classification"] = "contract_difference"
    elif actions == assignment:
        record["classification"] = "identical_actions"
    elif abs(evaluation["objective"] - best) <= tol:
        record["classification"] = "tied_optimum"
    elif evaluation["objective"] < best:
        record["classification"] = "strict_objective_preference"
    else:
        record["classification"] = "inconsistent_reference_above_optimum"
    return record


def consistent(record):
    """Same-state record agrees with the LP bound order and with bounded
    enumeration where it ran; the reference never beats the optimum."""
    best = (record.get("enumeration") or {}).get("best")
    integer = record.get("integer_objective")
    return (record.get("bounds_consistent", True)
            and record.get("classification")
            != "inconsistent_reference_above_optimum"
            and (best is None or integer is None
                 or abs(best - integer) <= eip.COMPARISON_TOL
                 * max(1.0, abs(best))))


def coefficient_magnitudes(problems):
    values = [v for p in problems for r in p.requests
              for v in (r.decode_utility, r.prefill_token_utility,
                        r.preemption_penalty) if v > 0]
    if not values:
        return None
    policy = _numerical_policy()
    return dict(min_positive=min(values), max=max(values),
                feasibility_tol=policy.feasibility_tol,
                objective_abs_tol=policy.objective_abs_tol,
                highs_default_dual_feasibility_tol=(
                    HIGHS_DEFAULT_DUAL_FEASIBILITY_TOL),
                min_over_dual_tol=min(values)
                / HIGHS_DEFAULT_DUAL_FEASIBILITY_TOL)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


def run_one(cpu, method, case, limit):
    import lp_relaxation_scheduler as lrs
    c = driver.case_inputs(case)
    runtime = cpu.CpuRuntime(method, limit, case)
    trace, captured, failure = {}, [], None
    with contextlib.ExitStack() as stack:
        if method == "lp_coordinated":
            native = lrs.solve_and_extract

            def recording(problem):
                result = native(problem)
                captured.append((problem, result))
                return result
            stack.enter_context(mock.patch.object(lrs, "solve_and_extract",
                                                  recording))
        elif method == "adapted_vllm":
            native_step = runtime.step

            def step():
                # Before the decision, at the quiescent boundary.
                captured.append(coordinated_problem(
                    runtime.scheduler, c["b_max"], c["c_max"], c["s_max"]))
                return native_step()
            runtime.step = step
        try:
            outcome = driver.run_comparison(method, limit, runtime, trace, case)
        except BaseException as err:
            outcome = driver.FAILURE
            failure = dict(type=type(err).__name__, message=str(err),
                           traceback=traceback.format_exc())
            for name in ("stage", "category", "reason"):
                if hasattr(err, name):
                    failure[name] = getattr(err, name)

    same_state = []
    if method == "lp_coordinated":
        for index, (problem, result) in enumerate(captured):
            integer = eip.solve_integer_plan(problem)
            record = eip.compare_same_state(problem, result, integer)
            record.update(index=index, enumeration=enumeration_check(problem))
            same_state.append(record)
        magnitudes = coefficient_magnitudes([p for p, _ in captured])
    elif method == "adapted_vllm":
        problems = []
        for index, decision in enumerate(trace.get("decisions", [])):
            problem = captured[index]
            if not hasattr(problem, "requests"):
                same_state.append(dict(index=index, mapping_failure=(
                    driver.to_jsonable(problem))))
                continue
            problems.append(problem)
            record = compare_reference(problem, decision["proposals"][0],
                                       eip.solve_integer_plan(problem))
            record.update(index=index, enumeration=enumeration_check(problem))
            same_state.append(record)
        magnitudes = coefficient_magnitudes(problems)
    else:
        magnitudes = None
    summary = rollout_summary(method, case, limit, c, trace, outcome, failure)
    summary["coefficient_magnitudes"] = magnitudes
    return summary, trace, same_state


def rollout_summary(method, case, limit, c, trace, outcome, failure):
    decisions = trace.get("decisions", [])
    final = trace.get("final") or {}
    acc = final.get("accounting") or {}
    reasons = {}
    replacements = 0
    for d in decisions:
        need = d.get("victim_need")
        if not need:
            continue
        for reason in need["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        if need["victims"] and need["selected_admission_charges"] > 0:
            replacements += 1

    def total(key):
        return sum(a[key] or 0 for a in acc.values())
    return dict(
        method=method, case=case, resident_limit=limit,
        validation_passed=failure is None, workload_outcome=outcome,
        failure=failure, decisions=len(decisions),
        idle_intervals=trace.get("idle_intervals"),
        finished=sorted(n for n, a in acc.items() if a["completion"] is not None),
        unfinished=final.get("unfinished"),
        unsubmitted=final.get("unsubmitted"),
        completion={n: a["completion"] for n, a in acc.items()},
        preemptions=total("preemptions"),
        preemptions_after_output=total("preemptions_after_output"),
        decisions_with_admission_and_victims=replacements,
        victim_reason_counts=reasons,
        discarded_prompt_tokens=total("discarded_prompt_tokens"),
        prefill_tokens_executed=total("prefill_tokens_executed"),
        prefill_tokens_replayed=total("prefill_tokens_replayed"),
        prefill_tokens_expanded_context=total("prefill_tokens_expanded_context"),
        generated_token_events=total("generated_token_events"),
        overgeneration=total("overgeneration"),
        omitted_decisions=sum(len(a["omitted_decisions"]) for a in acc.values()),
        emitted=[[d["schedules"][0]["outputs"]["preempted_seq_ids"],
                  d["schedules"][0]["outputs"]["scheduled"]]
                 for d in decisions if d.get("schedules")],
        parameters=driver.run_parameters(method, limit, case),
    )


def first_divergence(a, b, trace_a, trace_b):
    """First decision index whose emitted output differs, and whether the
    before-states there were identical (a same-state difference)."""
    for i, (x, y) in enumerate(zip(a["emitted"], b["emitted"])):
        if x != y:
            keys = ("running", "waiting", "free_blocks", "seqs", "block_tables")
            sa = trace_a["decisions"][i]["before"]
            sb = trace_b["decisions"][i]["before"]
            return dict(index=i, same_before_state=all(
                sa[k] == sb[k] for k in keys if k != "block_tables")
                and {k: len(v) for k, v in sa["block_tables"].items()}
                == {k: len(v) for k, v in sb["block_tables"].items()},
                a=x, b=y)
    if len(a["emitted"]) != len(b["emitted"]):
        return dict(index=min(len(a["emitted"]), len(b["emitted"])),
                    same_before_state=None, note="one run ended earlier")
    return None


# ---------------------------------------------------------------------------
# Diagnostic single states
# ---------------------------------------------------------------------------

# Residents: (prompt_len, prompt_tokens_processed, output_count) in running
# order; waiting: prompt lengths in waiting order. Raw IDs follow the list
# order (residents first). Every state uses block size 16 and watermark
# int(0.01 * pool).
DIAGNOSTIC_STATES = dict(
    mixed_ample=dict(
        note="resident A has 8 prefill tokens left; later resident B is "
        "decode-ready with zero additional demand; ample memory",
        residents=[(16, 8, 0), (8, 8, 0)], waiting=[], pool=8,
        max_model_len=32, limit=2, b_max=8, c_max=8, s_max=2, budget=8),
    mixed_pressure=dict(
        note="as mixed_ample with zero free blocks: B's decode needs one "
        "block; later decode-ready resident C (demand 0) is a legal victim",
        residents=[(16, 8, 0), (16, 16, 0), (8, 8, 0)], waiting=[], pool=3,
        max_model_len=32, limit=3, b_max=8, c_max=8, s_max=3, budget=8),
    zero_free_decodes=dict(
        note="four equal-memory decode-ready residents, zero free blocks, "
        "zero additional demand (memory_pressure decision 1 shape)",
        residents=[(16, 16, 0)] * 4, waiting=[], pool=4, max_model_len=64,
        limit=4, b_max=64, c_max=64, s_max=4, budget=64),
    fractional_admission=dict(
        note="waiting A then B; free pool 16, watermark 0; admission charges "
        "16 and 1; chunk cap 1; token, action, and resident limits 2. "
        "Single-state diagnostic, not a completion workload",
        residents=[], waiting=[256, 1], pool=16, max_model_len=256, limit=2,
        b_max=2, c_max=1, s_max=2, budget=2),
)


def build_state(spec, kind):
    """A registered scheduler holding the constructed state."""
    import lp_relaxation_scheduler as lrs
    from sarathi.config import (AdaptedVLLMSchedulerConfig, CacheConfig,
                                LPSchedulerConfig)
    from sarathi.core.datatypes.sampling_params import SamplingParams
    from sarathi.core.datatypes.sequence import Sequence
    from sarathi.core.datatypes.sequence_status import SequenceStatus
    from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry

    if kind == "adapted_vllm":
        config = AdaptedVLLMSchedulerConfig(spec["limit"], spec["max_model_len"],
                                            1, spec["budget"])
    else:
        config = LPSchedulerConfig(
            max_num_seqs=spec["limit"], max_model_len=spec["max_model_len"],
            num_pipeline_stages=1, b_max=spec["b_max"], c_max=spec["c_max"],
            s_max=spec["s_max"], decode_utility=1.0,
            prefill_token_utility=1.0, preemption_penalty=1.0,
            numerical_policy=lrs.NumericalPolicy(
                **driver.NUMERICAL_POLICY_FIELDS))
    cache = CacheConfig(block_size=driver.BLOCK_SIZE, gpu_memory_utilization=0.5)
    cache.num_gpu_blocks = spec["pool"]
    scheduler = SchedulerRegistry.get(config.type, config, cache)
    arrival = time.monotonic() - 1.0
    raw = itertools.count()
    for prompt, processed, outputs in spec["residents"]:
        sid = next(raw)
        seq = Sequence(sid, None, list(range(1000 * (sid + 1),
                                             1000 * (sid + 1) + prompt)),
                       driver.BLOCK_SIZE, 2, arrival, SamplingParams())
        seq.set_status(SequenceStatus.RUNNING)
        scheduler.block_manager.allocate(seq)
        seq.update_prompt_tokens_processed(processed)
        for k in range(outputs):
            seq.append_token_id(500 + k)
        seq.set_status(SequenceStatus.PAUSED)
        scheduler.running.append(seq)
    for prompt in spec["waiting"]:
        sid = next(raw)
        scheduler.add_seq(Sequence(
            sid, None, list(range(1000 * (sid + 1), 1000 * (sid + 1) + prompt)),
            driver.BLOCK_SIZE, 2, arrival, SamplingParams()))
    return scheduler


def run_diagnostic(name, spec):
    import lp_relaxation_scheduler as lrs
    import lpserve_plan_execution as lpe
    import lpserve_state_mapping as lsm

    def mapped(scheduler):
        snap = lsm.map_scheduler_state(
            scheduler, snapshot_time=time.monotonic(), b_max=spec["b_max"],
            c_max=spec["c_max"], s_max=spec["s_max"],
            coefficient_policy=coordinated_policy(spec["c_max"], spec["b_max"]),
            rank_profile=lsm.vllm_oriented_ranks,
            numerical_policy=_numerical_policy())
        driver.check(isinstance(snap, lsm.StateSnapshot),
                     f"{name}: mapping failed {snap}")
        return snap

    record = dict(name=name, spec=spec,
                  label="constructed with native allocation and progress "
                  "methods (shape reachable under the LP; see note)")
    lp_sched = build_state(spec, "lp")
    lp_snap = mapped(lp_sched)
    problem = lp_snap.lp_problem
    lp_result = lrs.solve_and_extract(problem)
    lp_outputs = (lpe.execute_plan(lp_sched, lp_snap, lp_result)
                  if isinstance(lp_result, lrs.SchedulingSuccess) else None)
    ilp_sched = build_state(spec, "lp")
    ilp_snap = mapped(ilp_sched)
    driver.check(ilp_snap.lp_problem == problem,
                 f"{name}: separate builds map to different problems")
    integer = eip.solve_integer_plan(ilp_snap.lp_problem)
    ilp_outputs = (lpe.execute_plan(ilp_sched, ilp_snap, lrs.SchedulingSuccess(
        integer.problem_id, integer.plan, None, None))
        if isinstance(integer, eip.IntegerOptimum) else None)
    for label, outputs in (("lp", lp_outputs), ("integer", ilp_outputs)):
        # Format the type only: the inherited SchedulerOutputs.__repr__
        # names a missing attribute.
        driver.check(outputs is not None and not isinstance(outputs, lrs.Failure),
                     f"{name}: native executor rejected the {label} plan: "
                     f"{type(outputs).__name__} "
                     f"{getattr(outputs, 'reason', '')}")
    ref = build_state(spec, "adapted_vllm")
    proposal = driver.to_jsonable(ref._propose(time.monotonic()))
    enumeration = enumeration_check(problem)
    driver.check("best" not in enumeration
                 or abs(enumeration["best"] - integer.plan.objective)
                 <= eip.COMPARISON_TOL,
                 f"{name}: enumeration {enumeration} disagrees with the "
                 f"integer optimum {integer.plan.objective}")
    record.update(
        coefficients={r.request_id: dict(
            decode=r.decode_utility, prefill_token=r.prefill_token_utility,
            penalty=r.preemption_penalty, service_rank=r.service_rank,
            victim_rank=r.victim_rank, upper_bound=r.prefill_upper_bound)
            for r in problem.requests},
        free_blocks=problem.m_free, watermark=problem.admission_watermark,
        lp=driver.to_jsonable(lp_result),
        lp_relaxed_values={d.request_id: [d.x, d.y, d.prefill_indicator, d.z]
                           for d in lp_result.relaxed.decisions},
        integer=driver.to_jsonable(integer),
        lp_vs_integer=eip.compare_same_state(problem, lp_result, integer),
        executed=dict(lp=driver.outputs_record(lp_outputs),
                      integer=driver.outputs_record(ilp_outputs)),
        reference_proposal=proposal,
        reference_vs_integer=compare_reference(problem, proposal, integer),
        enumeration=enumeration)
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def report(rollouts, same_state, divergences, diagnostics):
    lines = ["Rollouts (CPU, native central/worker replay)", ""]
    head = ("case", "limit", "method", "valid", "outcome", "dec", "done",
            "preempt", "after_out", "adm+victim", "discarded", "replayed",
            "expanded", "events", "overgen", "omitted")
    lines.append(" | ".join(head))
    for s in rollouts:
        lines.append(" | ".join(_fmt(v) for v in (
            s["case"], s["resident_limit"], s["method"], s["validation_passed"],
            s["workload_outcome"], s["decisions"], len(s["finished"]),
            s["preemptions"], s["preemptions_after_output"],
            s["decisions_with_admission_and_victims"],
            s["discarded_prompt_tokens"], s["prefill_tokens_replayed"],
            s["prefill_tokens_expanded_context"], s["generated_token_events"],
            s["overgeneration"], s["omitted_decisions"])))
    lines += ["", "Victim reasons per run (decisions)"]
    for s in rollouts:
        if s["victim_reason_counts"]:
            lines.append(f"  {s['case']} limit {s['resident_limit']} "
                         f"{s['method']}: {s['victim_reason_counts']}")
    lines += ["", "Same-state: coordinated LP vs integer optimum "
              "(identical problems)"]
    for key, records in same_state.items():
        if not key.endswith("lp_coordinated"):
            continue
        counts = {}
        for r in records:
            counts[r["attribution"]] = counts.get(r["attribution"], 0) + 1
        gaps = [(r["index"], round(r["relaxation_gap"], 9),
                 round(r["extraction_loss"], 9)) for r in records
                if r.get("relaxation_gap_present")
                or r.get("attribution") == "extraction_loss"]
        lines.append(f"  {key}: {len(records)} decisions {counts}; "
                     f"bounds consistent "
                     f"{all(r.get('bounds_consistent') for r in records)}; "
                     f"(index, relaxation gap, extraction loss) where nonzero: "
                     f"{gaps}")
    lines += ["", "Same-state: reference proposal vs coordinated integer "
              "optimum (reference states)"]
    for key, records in same_state.items():
        if not key.endswith("adapted_vllm"):
            continue
        counts, rows = {}, {}
        for r in records:
            cls = r.get("classification", "mapping_failure")
            counts[cls] = counts.get(cls, 0) + 1
            for row in (r.get("reference_evaluation") or {}).get(
                    "violated_rows", []):
                rows[row] = rows.get(row, 0) + 1
        lines.append(f"  {key}: {counts}; violated LP rows {rows}")
    lines += ["", "First emitted difference between rollouts"]
    for key, value in divergences.items():
        lines.append(f"  {key}: {value}")
    lines += ["", "Diagnostic single states"]
    for d in diagnostics:
        if "error" in d:
            lines.append(f"  {d['name']}: ERROR {d['error']['message']}")
            continue
        s = d["lp_vs_integer"]
        lines.append(
            f"  {d['name']}: LP relaxed {_fmt(s['lp_relaxed_objective'])}, "
            f"LP plan {s['lp_actions']} = {_fmt(s['lp_plan_objective'])}, "
            f"integer {s['integer_actions']} = {_fmt(s['integer_objective'])} "
            f"({s['attribution']}); reference "
            f"{d['reference_vs_integer']['reference_actions']} = "
            f"{_fmt(d['reference_vs_integer']['reference_evaluation']['objective'])}"
            f" ({d['reference_vs_integer']['classification']}, violated "
            f"{d['reference_vs_integer']['reference_evaluation']['violated_rows']}"
            f" {d['reference_vs_integer']['reference_evaluation']['domain_violations']}"
            f"); enumeration {d['enumeration']}")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    out = Path(args.output_dir)
    if (out / "run_record.json").exists():
        parser.error(f"{out} already holds run artifacts")
    out.mkdir(parents=True, exist_ok=True)

    import test_compare_scheduler_gpu as cpu
    cpu.setUpModule()
    record = dict(argv=sys.argv, methods=METHODS,
                  workloads=[list(w) for w in WORKLOADS],
                  coordinated=driver.COORDINATED,
                  integer_solver=dict(solver=eip.SOLVER,
                                      options=eip.SOLVER_OPTIONS,
                                      conversion_tol=eip.CONVERSION_TOL,
                                      comparison_tol=eip.COMPARISON_TOL),
                  provenance=driver.provenance_record(),
                  environment=driver.environment_record())
    rollouts, same_state, traces = [], {}, {}
    for case, limit in WORKLOADS:
        for method in METHODS:
            print(f"rollout {case} limit {limit} {method}", flush=True)
            summary, trace, records = run_one(cpu, method, case, limit)
            run_dir = out / "rollouts" / f"{case}_limit{limit}" / method
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps(driver.to_jsonable(summary), indent=1))
            (run_dir / "decision_trace.json").write_text(
                json.dumps(driver.to_jsonable(trace), indent=1))
            rollouts.append(summary)
            traces[case, limit, method] = trace
            if records:
                key = f"{case}_limit{limit}_{method}"
                same_state[key] = driver.to_jsonable(records)
                (out / "same_state").mkdir(exist_ok=True)
                (out / "same_state" / f"{key}.json").write_text(
                    json.dumps(same_state[key], indent=1))
            print(f"  -> valid {summary['validation_passed']} outcome "
                  f"{summary['workload_outcome']} decisions "
                  f"{summary['decisions']}", flush=True)
    by_key = {(s["case"], s["resident_limit"], s["method"]): s
              for s in rollouts}
    divergences = {}
    for case, limit in WORKLOADS:
        for a, b in (("lp_coordinated", driver.INTEGER_POLICY),
                     ("adapted_vllm", "lp_coordinated"),
                     ("adapted_vllm", driver.INTEGER_POLICY)):
            divergences[f"{case}_limit{limit} {a} vs {b}"] = first_divergence(
                by_key[case, limit, a], by_key[case, limit, b],
                traces[case, limit, a], traces[case, limit, b])
    diagnostics = []
    for name, spec in DIAGNOSTIC_STATES.items():
        print(f"diagnostic {name}", flush=True)
        try:
            diagnostics.append(driver.to_jsonable(run_diagnostic(name, spec)))
        except BaseException as err:
            diagnostics.append(dict(name=name, error=dict(
                type=type(err).__name__, message=str(err),
                traceback=traceback.format_exc())))
    (out / "diagnostic_states.json").write_text(json.dumps(diagnostics,
                                                           indent=1))
    (out / "divergences.json").write_text(json.dumps(
        driver.to_jsonable(divergences), indent=1))
    text = report(rollouts, same_state, divergences, diagnostics)
    (out / "comparison.txt").write_text(text)
    print(text, flush=True)
    passed = (all(s["validation_passed"] for s in rollouts)
              and not any("error" in d for d in diagnostics)
              and all(consistent(r) for rs in same_state.values()
                      for r in rs))
    record["all_checks_passed"] = passed
    (out / "run_record.json").write_text(json.dumps(
        driver.to_jsonable(record), indent=1))
    print("RESULT:", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
