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

``--batch queue_growth`` instead runs the reciprocal-square queue-growth
batch (``QUEUE_RUNS``: the six ``compare_scheduler_gpu.QUEUE_GROWTH_CASES``
at resident limit 8; six methods on the six-request anchor, and the
reference, ``lp_reciprocal_square``, and ``ilp_reciprocal_square`` on the
larger cases). Each run writes ``rollouts/<case>/<method>/`` with
``decision_trace.json.gz``, ``summary.json``, ``comparisons.json``, and
``console.log``. After each rollout, every reached decision receives a
same-state integer comparison from the retained JSON trace
(``decision_comparisons``): an offline solve of each LP decision's own
problem, the rollout's own integer result for the integer-driven method
(not solved again), and an offline solve of the reference's read-only
reciprocal-square diagnostic problem. Coverage counts are part of the pass
criterion; a missing record fails the batch.

``--batch retained_runs --runs <dir> ...`` performs the same offline
comparisons for retained runs (for example GPU runs), optionally checking
their emitted decisions against a queue-growth CPU batch (``--cpu-batch``).

``--batch running_order`` runs the running-order batch: the reference, the
unchanged reciprocal-square LP (control), ``lp_running_order``, and
``ilp_running_order`` on ``RUNNING_ORDER_WORKLOADS`` (15 workload/limit
pairs), each written to ``rollouts/<case>_limit<n>/<method>/`` with
``decision_trace.json.gz``, ``summary.json``, ``comparisons.json.gz``, and
``console.log``. Every reached decision receives the same-state integer
comparison of the queue-growth batch (the reference's read-only diagnostic
problem carries running-order coefficients). Decisions with running-order
coefficients also receive exact checks (``running_order_decision``):
weights recomputed from the problem's facts, dominance margins before and
after float conversion, coefficient scale, resident progress vectors, exact
utilities of the integer optimum, the LP plan, and the reference proposal,
and bounded enumeration of small problems. ``diagnostics.json`` holds the
single-state diagnostics (``RUNNING_ORDER_DIAGNOSTICS``). Experimental
outcomes (bounded noncompletion, contract differences, extraction losses)
are reported; solver failures, missing evidence, invalid plans, harness
errors, and numerical contradictions make the result FAIL.
"""

import argparse
import contextlib
import functools
import io
import itertools
import json
import math
from fractions import Fraction
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
    integer_type = type(integer_result).__name__
    return classify_reference(problem, proposal, integer_type,
                              eip.integer_summary(integer_type, integer_result))


def classify_reference(problem, proposal, integer_status, integer):
    """``compare_reference`` on an ``integer_summary`` of the integer
    result for the same problem."""
    assignment = reference_assignment(proposal)
    evaluation = eip.evaluate_assignment(problem, assignment)
    record = dict(problem_id=problem.problem_id, reference_actions=assignment,
                  reference_evaluation=evaluation,
                  integer_status=integer_status)
    if integer["status"] != "optimal":
        record["classification"] = "integer_solve_failed"
        record["integer_failure"] = dict(stage=integer["stage"],
                                         category=integer["category"],
                                         reason=integer["reason"])
        return record
    best = integer["objective"]
    actions = integer["actions"]
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
        note="as mixed_ample with zero free blocks: B (one generated token "
        "after its 16-token prompt) needs one new block to decode; later "
        "decode-ready resident C (demand 0) is a legal victim",
        residents=[(16, 8, 0), (16, 16, 1), (8, 8, 0)], waiting=[], pool=3,
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
    def block_counts(scheduler):
        manager = scheduler.block_manager
        return dict(free=manager.get_num_free_gpu_blocks(), physical={
            str(seq.seq_id): len(manager.get_block_table(seq))
            for seq in scheduler.running})

    lp_sched = build_state(spec, "lp")
    lp_blocks = dict(before=block_counts(lp_sched))
    lp_snap = mapped(lp_sched)
    problem = lp_snap.lp_problem
    lp_result = lrs.solve_and_extract(problem)
    lp_outputs = (lpe.execute_plan(lp_sched, lp_snap, lp_result)
                  if isinstance(lp_result, lrs.SchedulingSuccess) else None)
    ilp_sched = build_state(spec, "lp")
    ilp_blocks = dict(before=block_counts(ilp_sched))
    ilp_snap = mapped(ilp_sched)
    driver.check(ilp_snap.lp_problem == problem,
                 f"{name}: separate builds map to different problems")
    integer = eip.solve_integer_plan(ilp_snap.lp_problem)
    ilp_outputs = (lpe.execute_plan(ilp_sched, ilp_snap, lrs.SchedulingSuccess(
        integer.problem_id, integer.plan, None, None))
        if isinstance(integer, eip.IntegerOptimum) else None)
    lp_blocks["after"] = block_counts(lp_sched)
    ilp_blocks["after"] = block_counts(ilp_sched)
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
        mapped_requests={r.request_id: dict(
            prompt_tokens_remaining=r.prompt_tokens_remaining,
            output_len=r.output_len, logical_blocks=r.logical_block_count,
            physical_blocks=r.physical_block_count,
            decode_demand=r.decode_demand,
            prefill_fixed_charge=r.prefill_fixed_charge)
            for r in lp_snap.requests},
        block_counts=dict(lp=lp_blocks, integer=ilp_blocks),
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


# ---------------------------------------------------------------------------
# Queue-growth batch: reciprocal-square policy as the queue grows
# ---------------------------------------------------------------------------

QUEUE_LIMIT = 8
QUEUE_ANCHOR = "queue6_burst"
QUEUE_ANCHOR_METHODS = ("adapted_vllm", "lp_uniform", "lp_rank_based",
                        "lp_coordinated", driver.RECIPROCAL_SQUARE_POLICY,
                        driver.RECIPROCAL_SQUARE_INTEGER_POLICY)
QUEUE_METHODS = ("adapted_vllm", driver.RECIPROCAL_SQUARE_POLICY,
                 driver.RECIPROCAL_SQUARE_INTEGER_POLICY)
QUEUE_RUNS = tuple((case, method) for case in driver.QUEUE_GROWTH_CASES
                   for method in (QUEUE_ANCHOR_METHODS if case == QUEUE_ANCHOR
                                  else QUEUE_METHODS))
QUEUE_DIVERGENCE_PAIRS = (
    (driver.RECIPROCAL_SQUARE_POLICY, driver.RECIPROCAL_SQUARE_INTEGER_POLICY),
    ("adapted_vllm", driver.RECIPROCAL_SQUARE_POLICY),
    ("adapted_vllm", driver.RECIPROCAL_SQUARE_INTEGER_POLICY),
    ("lp_coordinated", driver.RECIPROCAL_SQUARE_POLICY))


def write_json_gz(path, value):
    """Deterministic gzip (no timestamp) of a JSON record."""
    import gzip
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb",
                                                mtime=0) as f:
        f.write(json.dumps(value, indent=1).encode())


def read_trace(run_dir):
    import gzip
    run_dir = Path(run_dir)
    if (run_dir / "decision_trace.json.gz").exists():
        with gzip.open(run_dir / "decision_trace.json.gz", "rt") as f:
            return json.load(f)
    return json.loads((run_dir / "decision_trace.json").read_text())


def scale_record(problem):
    """Rank range and coefficient magnitudes of one problem: per term kind,
    the eligible values' smallest and largest positive value, the smallest
    positive difference between distinct values, and the count of values
    equal to another request's (exact ties). Diagnostic only."""
    reqs = problem.requests
    kinds = dict(
        decode=[r.decode_utility for r in reqs if r.decode_eligible],
        prefill_token=[r.prefill_token_utility for r in reqs
                       if r.prefill_eligible],
        full_chunk=[r.prefill_token_utility * r.prefill_upper_bound
                    for r in reqs if r.prefill_eligible],
        penalty=[r.preemption_penalty for r in reqs if r.preemption_eligible])
    record = dict(requests=len(reqs), residents=problem.resident_count,
                  max_service_rank=max((r.service_rank for r in reqs
                                        if r.service_rank is not None),
                                       default=None))
    for kind, values in kinds.items():
        positive = sorted(v for v in values if v > 0)
        distinct = sorted(set(positive))
        gaps = [b - a for a, b in zip(distinct, distinct[1:])]
        record[kind] = dict(
            count=len(values), min_positive=positive[0] if positive else None,
            max=positive[-1] if positive else None,
            min_gap=min(gaps) if gaps else None,
            exact_ties=len(positive) - len(distinct))
    return record


def decision_comparisons(trace, method, supplement=None):
    """One same-state comparison record per reached decision of a retained
    (JSON) trace, plus coverage counts. ``supplement(problem, decision,
    record, integer, method)``, when given, adds ``record["supplement"]``
    for each decision with a valid problem; its errors are comparison
    errors.

    LP methods: the decision's own mapped problem, reloaded from the record,
    receives one offline integer solve, compared with the recorded LP result.
    Integer-driven methods: the rollout's own integer result is reused, not
    solved again. Reference: the read-only reciprocal-square diagnostic
    problem of the pre-decision state receives one offline integer solve,
    and the recorded proposal is classified against it. A decision without a
    valid problem gets an explicit record; none is invented."""
    records = []
    coverage = dict(reached_decisions=len(trace.get("decisions", [])),
                    valid_problems=0, mapping_failures=0,
                    missing_problems=0, round_trip_failures=0, attempted_solves=0,
                    reused_integer_results=0, optimal=0, time_limits=0,
                    other_failures=0, comparison_errors=0)
    reference = method == "adapted_vllm"
    integer_driven = method in driver.INTEGER_POLICIES
    diagnostic = trace.get("reference_diagnostic_problems") or []
    for d in trace.get("decisions", []):
        index = d["index"]
        record = dict(index=index)
        records.append(record)
        try:
            events = {e["call"]: e for e in d.get("lp_events", [])}
            if reference:
                entry = diagnostic[index] if index < len(diagnostic) else None
                source = "reference_diagnostic_mapping"
            else:
                entry = events.get("map_scheduler_state")
                source = "decision_mapping"
                if entry is not None:
                    entry = dict(result_type=entry["result_type"],
                                 result=entry["result"].get("lp_problem")
                                 if entry["result_type"] == "StateSnapshot"
                                 else entry["result"])
            if entry is None:
                record.update(problem_status="missing", problem_source=source)
                coverage["missing_problems"] += 1
                continue
            if entry["result_type"] not in ("LPProblem", "StateSnapshot"):
                record.update(problem_status="mapping_failure",
                              problem_source=source,
                              mapping_failure=entry["result"])
                coverage["mapping_failures"] += 1
                continue
            problem = lrs_validate(eip.problem_from_record(entry["result"]))
            round_trip = driver.to_jsonable(problem) == entry["result"]
            coverage["round_trip_failures"] += not round_trip
            coverage["valid_problems"] += 1
            record.update(problem_status="valid", problem_source=source,
                          problem_id=problem.problem_id,
                          round_trip_equal=round_trip,
                          scale=scale_record(problem))
            if integer_driven:
                solved = events.get("solve_integer_plan")
                if solved is None:
                    record["integer_source"] = "missing_from_rollout"
                    coverage["comparison_errors"] += 1
                    continue
                integer = eip.integer_summary(solved["result_type"],
                                              solved["result"])
                record["integer_source"] = "reused_from_rollout"
                coverage["reused_integer_results"] += 1
            else:
                start = time.perf_counter()
                result = eip.solve_integer_plan(problem)
                record["solve_seconds"] = time.perf_counter() - start
                integer = eip.integer_summary(type(result).__name__, result)
                record["integer_source"] = "offline_solve"
                coverage["attempted_solves"] += 1
            if integer["status"] == "optimal":
                coverage["optimal"] += 1
            elif integer["category"] == "solver_limit":
                coverage["time_limits"] += 1
            else:
                coverage["other_failures"] += 1
            record["integer"] = integer
            if reference:
                proposals = d.get("proposals") or []
                record["comparison"] = (
                    classify_reference(problem, proposals[0], integer["status"],
                                       integer)
                    if proposals else dict(classification="no_proposal"))
            elif not integer_driven:
                solved = events.get("solve_and_extract")
                lp_type = solved["result_type"] if solved else "missing"
                record["comparison"] = eip.compare_summaries(
                    problem.problem_id, lp_type,
                    eip.lp_summary(lp_type, solved["result"]) if solved
                    else None, integer["status"], integer)
            if supplement is not None:
                record["supplement"] = supplement(problem, d, record, integer,
                                                  method)
        except Exception as err:  # recorded and counted, never skipped
            record["comparison_error"] = dict(type=type(err).__name__,
                                              message=str(err))
            coverage["comparison_errors"] += 1
    coverage["records"] = len(records)
    coverage["missing_records"] = coverage["reached_decisions"] - len(records)
    coverage["no_valid_problem"] = (coverage["mapping_failures"]
                                    + coverage["missing_problems"])
    # A mapping failure is an explicit record; a missing problem is not.
    coverage["complete"] = (
        coverage["missing_records"] == 0 and coverage["comparison_errors"] == 0
        and coverage["missing_problems"] == 0
        and coverage["round_trip_failures"] == 0
        and coverage["valid_problems"] == coverage["attempted_solves"]
        + coverage["reused_integer_results"])
    return records, coverage


def lrs_validate(problem):
    import lp_relaxation_scheduler as lrs
    canonical = lrs.validate_problem(problem)
    driver.check(not isinstance(canonical, lrs.Failure),
                 f"retained problem failed validation: {canonical}")
    return canonical


def comparison_consistent(record):
    """An optimal same-state record agrees with the LP bound order; the
    reference never beats the integer optimum."""
    comparison = record.get("comparison") or {}
    return (comparison.get("bounds_consistent", True)
            and comparison.get("classification")
            != "inconsistent_reference_above_optimum")


def service_record(trace, case):
    """Service, omission, preemption, and order observations of one
    rollout (from the recorded decisions and driver accounting)."""
    c = driver.case_inputs(case)
    names = {int(k): v for k, v in c["names"].items()}
    order = list(c["prompts"])
    victims, resident_omissions, streak, longest = [], 0, {}, {}
    for d in trace.get("decisions", []):
        if not d.get("schedules"):
            continue
        emitted = d["schedules"][0]["outputs"]
        served = {sid for sid, _ in emitted["scheduled"]}
        preempted = emitted["preempted_seq_ids"]
        running = d["before"]["running"]
        admitted = [names[s] for s in served if s in d["before"]["waiting"]]
        for sid in preempted:
            seq = d["before"]["seqs"][str(sid)]
            victims.append(dict(
                index=d["index"], victim=names[sid],
                running_position=running.index(sid),
                running_count=len(running),
                had_output=bool(seq["output_token_ids"]),
                prompt_progress=seq["prompt_tokens_processed"],
                reasons=(d.get("victim_need") or {}).get("reasons"),
                admitted_same_decision=admitted))
        for sid in running:
            name = names[sid]
            if sid in served or sid in preempted:
                streak[name] = 0
                continue
            resident_omissions += 1
            streak[name] = streak.get(name, 0) + 1
            longest[name] = max(longest.get(name, 0), streak[name])
    acc = (trace.get("final") or {}).get("accounting") or (
        trace["decisions"][-1].get("accounting") if trace.get("decisions")
        else {}) or {}

    def inversions(key):
        seen = [(order.index(n), a[key]) for n, a in acc.items()
                if a[key] is not None]
        return sum(1 for i, a in seen for j, b in seen if i < j and b < a)

    by_length = {}
    for n, a in acc.items():
        group = by_length.setdefault(str(len(c["prompts"][n])), dict(
            requests=0, first_prefill=[], completion=[]))
        group["requests"] += 1
        for key in ("first_prefill", "completion"):
            if a[key] is not None:
                group[key].append(a[key])
    for group in by_length.values():
        for key in ("first_prefill", "completion"):
            values = group.pop(key)
            group[f"{key}_mean"] = (sum(values) / len(values) if values
                                    else None)
            group[f"{key}_max"] = max(values) if values else None
            group[f"{key}_count"] = len(values)
    return dict(
        victims=victims,
        victim_running_positions=[[v["running_position"], v["running_count"]]
                                  for v in victims],
        decisions_with_victims_and_admissions=len(
            {v["index"] for v in victims if v["admitted_same_decision"]}),
        resident_omissions=resident_omissions,
        longest_resident_omission_streak=max(longest.values(), default=0),
        admission_order_inversions=inversions("first_prefill"),
        completion_order_inversions=inversions("completion"),
        by_prompt_length=by_length)


def comparison_summary(records, method):
    """Counts and extremes of one run's same-state records."""
    out = dict(attributions={}, relaxation_gaps=[], extraction_losses=[],
               action_differences=[], reference_classes={},
               reference_violated_rows={}, near_ties=[],
               strict_preference_gap_min=None, fractional_lp_decisions=0,
               max_integrality_deviation=None, integer_raw_statuses={},
               max_service_rank=None, min_positive_coefficient=None,
               min_gaps={}, exact_ties={}, solve_seconds_max=None)
    for r in records:
        integer = r.get("integer") or {}
        if integer:
            key = str(integer["raw_status"])
            out["integer_raw_statuses"][key] = out["integer_raw_statuses"].get(
                key, 0) + 1
        if integer.get("max_integrality_deviation") is not None:
            out["max_integrality_deviation"] = max(
                out["max_integrality_deviation"] or 0.0,
                integer["max_integrality_deviation"])
        scale = r.get("scale")
        if scale:
            out["max_service_rank"] = max(out["max_service_rank"] or 0,
                                          scale["max_service_rank"] or 0)
            for kind in ("decode", "prefill_token", "full_chunk", "penalty"):
                k = scale[kind]
                for value, key, pick in ((k["min_positive"],
                                          "min_positive_coefficient", min),):
                    if value is not None and kind != "full_chunk":
                        out[key] = (value if out[key] is None
                                    else pick(out[key], value))
                if k["min_gap"] is not None:
                    out["min_gaps"][kind] = min(out["min_gaps"].get(
                        kind, k["min_gap"]), k["min_gap"])
                out["exact_ties"][kind] = (out["exact_ties"].get(kind, 0)
                                           + k["exact_ties"])
        if "solve_seconds" in r:
            out["solve_seconds_max"] = max(out["solve_seconds_max"] or 0.0,
                                           r["solve_seconds"])
        comparison = r.get("comparison") or {}
        if method == "adapted_vllm":
            cls = comparison.get("classification", r.get("problem_status"))
            out["reference_classes"][cls] = out["reference_classes"].get(
                cls, 0) + 1
            for row in (comparison.get("reference_evaluation") or {}).get(
                    "violated_rows", []):
                out["reference_violated_rows"][row] = out[
                    "reference_violated_rows"].get(row, 0) + 1
            if cls in ("tied_optimum", "strict_objective_preference"):
                gap = (comparison["integer_objective"]
                       - comparison["reference_evaluation"]["objective"])
                if cls == "tied_optimum":
                    out["near_ties"].append([r["index"], gap])
                else:
                    out["strict_preference_gap_min"] = min(
                        out["strict_preference_gap_min"] or gap, gap)
            continue
        if not comparison:
            continue
        a = comparison["attribution"]
        out["attributions"][a] = out["attributions"].get(a, 0) + 1
        out["fractional_lp_decisions"] += bool(
            comparison.get("fractional_request_count"))
        if comparison.get("relaxation_gap_present"):
            out["relaxation_gaps"].append([r["index"],
                                           comparison["relaxation_gap"]])
        if a == "extraction_loss":
            out["extraction_losses"].append([r["index"],
                                             comparison["extraction_loss"]])
        if a != "identical_actions" and a != "not_comparable":
            out["action_differences"].append([r["index"], a])
        if a == "tied_optimum_different_actions":
            out["near_ties"].append([r["index"],
                                     comparison["extraction_loss"]])
    return out


def queue_report(rows, divergences, title=None, label=None,
                 reference_policy="reciprocal-square"):
    """The queue-growth report; ``title``, ``label`` (run summary -> case
    column), and the name of the reference's diagnostic policy let another
    batch reuse it."""
    label = label or (lambda s: s["case"])
    lines = [title or "Queue-growth batch (CPU, native central/worker "
             "replay; resident limit 8)", "",
             "Rollouts"]
    head = ("case", "method", "valid", "outcome", "dec", "done", "preempt",
            "after_out", "adm+victim", "discarded", "replayed", "expanded",
            "events", "overgen", "res_omit", "max_streak", "adm_inv",
            "done_inv")
    lines.append(" | ".join(head))
    for r in rows:
        s, v = r["summary"], r["service"]
        lines.append(" | ".join(_fmt(x) for x in (
            label(s), s["method"], s["validation_passed"],
            s["workload_outcome"], s["decisions"], len(s["finished"]),
            s["preemptions"], s["preemptions_after_output"],
            v["decisions_with_victims_and_admissions"],
            s["discarded_prompt_tokens"], s["prefill_tokens_replayed"],
            s["prefill_tokens_expanded_context"], s["generated_token_events"],
            s["overgeneration"], v["resident_omissions"],
            v["longest_resident_omission_streak"],
            v["admission_order_inversions"],
            v["completion_order_inversions"])))
    lines += ["", "Comparison coverage (every reached decision)"]
    head = ("case", "method", "reached", "records", "valid", "no_problem",
            "solved", "reused", "optimal", "time_limit", "other_fail",
            "errors", "missing", "round_trip_fail", "complete")
    lines.append(" | ".join(head))
    for r in rows:
        k = r["coverage"]
        lines.append(" | ".join(_fmt(x) for x in (
            label(r["summary"]), r["summary"]["method"],
            k["reached_decisions"], k["records"], k["valid_problems"],
            k["no_valid_problem"], k["attempted_solves"],
            k["reused_integer_results"], k["optimal"], k["time_limits"],
            k["other_failures"], k["comparison_errors"], k["missing_records"],
            k["round_trip_failures"], k["complete"])))
    lines += ["", "Same-state results (identical problems only; objectives "
              "of different policies are not compared)"]
    for r in rows:
        m, s = r["comparisons"], r["summary"]
        tag = f"  {label(s)} {s['method']}:"
        solver = (f" integer raw statuses {m['integer_raw_statuses']}, max "
                  f"integrality deviation "
                  f"{_fmt(m['max_integrality_deviation'])}")
        if s["method"] == "adapted_vllm":
            lines.append(f"{tag} reference vs {reference_policy} integer "
                         f"optimum {m['reference_classes']}; violated LP rows "
                         f"{m['reference_violated_rows']}; smallest strict "
                         f"preference {_fmt(m['strict_preference_gap_min'])}; "
                         f"near ties {m['near_ties']};{solver}")
        elif s["method"] in driver.INTEGER_POLICIES:
            lines.append(f"{tag} integer results reused from the rollout;"
                         f"{solver}")
        else:
            gaps = m["relaxation_gaps"]
            lines.append(
                f"{tag} {m['attributions']}; fractional relaxations "
                f"{m['fractional_lp_decisions']}; relaxation gaps {len(gaps)} "
                f"(max {_fmt(max((g for _, g in gaps), default=0.0))}); "
                f"extraction losses {m['extraction_losses']}; action "
                f"differences {m['action_differences']}; near ties "
                f"{m['near_ties']};{solver}")
    lines += ["", "Numerical scale (eligible terms over all valid problems)"]
    for r in rows:
        m, s = r["comparisons"], r["summary"]
        lines.append(
            f"  {label(s)} {s['method']}: max rank {m['max_service_rank']}, "
            f"min positive coefficient {_fmt(m['min_positive_coefficient'])}, "
            f"min gaps { {k: float(f'{v:.3g}') for k, v in m['min_gaps'].items()} }, "
            f"exact ties {m['exact_ties']}, max offline solve "
            f"{_fmt(m['solve_seconds_max'])} s")
    lines += ["", "Victims: [decision, victim, running position / count, "
              "had output, reasons, admitted in the same decision]"]
    for r in rows:
        victims = r["service"]["victims"]
        if victims:
            lines.append(f"  {label(r['summary'])} {r['summary']['method']}:")
            for v in victims:
                lines.append(f"    {v['index']} {v['victim']} "
                             f"{v['running_position']}/{v['running_count']} "
                             f"{v['had_output']} {v['reasons']} "
                             f"{v['admitted_same_decision']}")
    lines += ["", "Mean first-prefill and completion decision index by "
              "prompt length (decision indices, not time)"]
    for r in rows:
        groups = r["service"]["by_prompt_length"]
        cells = ", ".join(
            f"{k}: {_fmt(g['first_prefill_mean'])}/{_fmt(g['completion_mean'])}"
            f" (max done {g['completion_max']})" for k, g in sorted(
                groups.items(), key=lambda kv: int(kv[0])))
        lines.append(f"  {label(r['summary'])} {r['summary']['method']}: "
                     f"{cells}")
    lines += ["", "First emitted difference between rollouts (closed loop): "
              "first index whose emitted output differs, and first index "
              "whose selected set (victims and served requests) differs"]
    for key, value in divergences.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def first_selection_divergence(a, b):
    """First decision index whose victims or served requests differ as sets
    (an emitted-order-only difference is not a selection difference)."""
    for i, (x, y) in enumerate(zip(a["emitted"], b["emitted"])):
        if (sorted(x[0]), sorted(map(tuple, x[1]))) != (
                sorted(y[0]), sorted(map(tuple, y[1]))):
            return i
    return None


def run_queue_rollout(cpu, method, case, run_dir, limit=QUEUE_LIMIT,
                      reference_policy=None):
    """One rollout in retained JSON form. ``reference_policy`` is the
    coefficient-policy factory of the reference's read-only diagnostic
    problems (default: reciprocal-square)."""
    runtime = cpu.CpuRuntime(method, limit, case)
    trace, failure = {}, None
    if method == "adapted_vllm":
        trace["reference_diagnostic_problems"] = (
            driver.capture_reference_problems(runtime, case, reference_policy))
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            outcome = driver.run_comparison(method, limit, runtime, trace,
                                            case)
    except BaseException as err:
        outcome = driver.FAILURE
        failure = dict(type=type(err).__name__, message=str(err),
                       traceback=traceback.format_exc())
        for name in ("stage", "category", "reason"):
            if hasattr(err, name):
                failure[name] = getattr(err, name)
    (run_dir / "console.log").write_text(log.getvalue())
    # The comparisons read the retained JSON form, as for a GPU run.
    trace = json.loads(json.dumps(driver.to_jsonable(trace)))
    write_json_gz(run_dir / "decision_trace.json.gz", trace)
    summary = rollout_summary(method, case, limit,
                              driver.case_inputs(case), trace, outcome, failure)
    return summary, trace


def analyse_run(trace, method, case, supplement=None):
    records, coverage = decision_comparisons(trace, method, supplement)
    return dict(records=records, coverage=coverage,
                comparisons=comparison_summary(records, method),
                service=service_record(trace, case),
                consistent=all(comparison_consistent(r) for r in records))


def queue_growth_main(out, cases=driver.QUEUE_GROWTH_CASES):
    import test_compare_scheduler_gpu as cpu
    cpu.setUpModule()
    runs = [r for r in QUEUE_RUNS if r[0] in cases]
    record = dict(argv=sys.argv, batch="queue_growth", selected_cases=list(cases),
                  runs=[list(r) for r in runs], resident_limit=QUEUE_LIMIT,
                  cases={case: driver.case_inputs(case) for case in cases},
                  reciprocal_square="w = 1 / (1 + r) ** 2; q = min(remaining, "
                  "c_max, b_max); decode w, prefill token w / q, penalty w",
                  coordinated=driver.COORDINATED,
                  integer_solver=dict(solver=eip.SOLVER,
                                      options=eip.SOLVER_OPTIONS,
                                      conversion_tol=eip.CONVERSION_TOL,
                                      comparison_tol=eip.COMPARISON_TOL),
                  provenance=driver.provenance_record(),
                  environment=driver.environment_record())
    rows, traces = [], {}
    for case, method in runs:
        print(f"rollout {case} {method}", flush=True)
        run_dir = out / "rollouts" / case / method
        run_dir.mkdir(parents=True)
        summary, trace = run_queue_rollout(cpu, method, case, run_dir)
        analysis = analyse_run(trace, method, case)
        (run_dir / "summary.json").write_text(json.dumps(
            driver.to_jsonable(dict(summary, coverage=analysis["coverage"],
                                    service=analysis["service"],
                                    comparisons=analysis["comparisons"])),
            indent=1))
        (run_dir / "comparisons.json").write_text(json.dumps(
            driver.to_jsonable(analysis["records"]), indent=1))
        rows.append(dict(summary=summary, **analysis))
        traces[case, method] = trace
        print(f"  -> valid {summary['validation_passed']} outcome "
              f"{summary['workload_outcome']} decisions {summary['decisions']} "
              f"coverage complete {analysis['coverage']['complete']}",
              flush=True)
    by_key = {(r["summary"]["case"], r["summary"]["method"]): r["summary"]
              for r in rows}
    divergences = {}
    for case in cases:
        for a, b in QUEUE_DIVERGENCE_PAIRS:
            if (case, a) in by_key and (case, b) in by_key:
                divergences[f"{case} {a} vs {b}"] = dict(
                    emitted=first_divergence(by_key[case, a], by_key[case, b],
                                             traces[case, a], traces[case, b]),
                    first_selection_difference=first_selection_divergence(
                        by_key[case, a], by_key[case, b]))
    (out / "divergences.json").write_text(json.dumps(
        driver.to_jsonable(divergences), indent=1))
    text = queue_report(rows, divergences)
    (out / "comparison.txt").write_text(text)
    print(text, flush=True)
    passed = all(r["summary"]["validation_passed"]
                 and r["coverage"]["complete"] and r["consistent"]
                 for r in rows)
    record["all_checks_passed"] = passed
    (out / "run_record.json").write_text(json.dumps(
        driver.to_jsonable(record), indent=1))
    print("RESULT:", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


def retained_runs_main(out, run_dirs, cpu_batch):
    """Offline CPU comparisons of retained (GPU) runs: every reached decision
    of each run, from its retained problems and results. With
    ``cpu_batch``, each run's emitted decisions are also compared with the
    CPU rollout of the same case and method."""
    record = dict(argv=sys.argv, batch="retained_runs",
                  runs=[str(d) for d in run_dirs], cpu_batch=cpu_batch,
                  integer_solver=dict(solver=eip.SOLVER,
                                      options=eip.SOLVER_OPTIONS,
                                      conversion_tol=eip.CONVERSION_TOL,
                                      comparison_tol=eip.COMPARISON_TOL),
                  provenance=driver.provenance_record(),
                  environment=driver.environment_record())
    rows, passed = [], True
    for run_dir in map(Path, run_dirs):
        summary = json.loads((run_dir / "summary.json").read_text())
        method, case = summary["run"]["policy"], summary["run"]["case"]
        trace = read_trace(run_dir)
        print(f"retained run {run_dir} ({case} {method})", flush=True)
        analysis = analyse_run(trace, method, case)
        rollout = rollout_summary(method, case, summary["run"]["resident_limit"],
                                  driver.case_inputs(case), trace,
                                  summary["workload_outcome"],
                                  summary["failure"])
        rollout["validation_passed"] = summary["validation_passed"]
        same_as_cpu = None
        if cpu_batch:
            cpu_summary = json.loads((Path(cpu_batch) / "rollouts" / case
                                      / method / "summary.json").read_text())
            same_as_cpu = cpu_summary["emitted"] == rollout["emitted"]
        target = out / run_dir.name
        target.mkdir(parents=True)
        (target / "comparisons.json").write_text(json.dumps(
            driver.to_jsonable(analysis["records"]), indent=1))
        (target / "summary.json").write_text(json.dumps(driver.to_jsonable(
            dict(rollout, coverage=analysis["coverage"],
                 service=analysis["service"],
                 comparisons=analysis["comparisons"],
                 emitted_equal_to_cpu_rollout=same_as_cpu)), indent=1))
        rows.append(dict(summary=rollout, **analysis))
        passed &= (analysis["coverage"]["complete"] and analysis["consistent"]
                   and same_as_cpu is not False)
        print(f"  -> coverage {analysis['coverage']}; emitted equal to CPU "
              f"rollout {same_as_cpu}", flush=True)
    text = queue_report(rows, {})
    (out / "comparison.txt").write_text(text)
    print(text, flush=True)
    record["all_checks_passed"] = passed
    (out / "run_record.json").write_text(json.dumps(
        driver.to_jsonable(record), indent=1))
    print("RESULT:", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Running-order batch
# ---------------------------------------------------------------------------

RUNNING_ORDER_METHODS = ("adapted_vllm", driver.RECIPROCAL_SQUARE_POLICY,
                         driver.RUNNING_ORDER_POLICY,
                         driver.RUNNING_ORDER_INTEGER_POLICY)
RUNNING_ORDER_WORKLOADS = (
    ((driver.TWO_REQUEST, 1), (driver.TWO_REQUEST, 2),
     (driver.MEMORY_PRESSURE, 4))
    + tuple((case, 4) for case in driver.REPRESENTATIVE_CASES)
    + tuple((case, QUEUE_LIMIT) for case in driver.QUEUE_GROWTH_CASES
            + driver.RUNNING_ORDER_QUEUE_CASES))
RUNNING_ORDER_PAIRS = (
    (driver.RUNNING_ORDER_POLICY, driver.RUNNING_ORDER_INTEGER_POLICY),
    ("adapted_vllm", driver.RUNNING_ORDER_INTEGER_POLICY),
    ("adapted_vllm", driver.RUNNING_ORDER_POLICY),
    (driver.RECIPROCAL_SQUARE_POLICY, driver.RUNNING_ORDER_POLICY))
# Methods whose retained problems carry running-order coefficients (the
# reference through its read-only diagnostic mapping).
RUNNING_ORDER_PROBLEM_METHODS = ("adapted_vllm",) + driver.RUNNING_ORDER_POLICIES
# Bounded enumeration in rollouts and diagnostics uses ENUMERATION_LIMIT.


def _frac(value):
    """A JSON record of an exact rational."""
    value = Fraction(value)
    return dict(exact=f"{value.numerator}/{value.denominator}",
                approx=float(value))


def intended_running_order(problem):
    """The running-order coefficients the policy intends for ``problem``,
    recomputed exactly from its facts (residency, eligibility, remaining
    prompt, caps, and service ranks, which the vLLM-oriented profile sets to
    the running position and then the arrived waiting position). The
    recorded float coefficients are not read."""
    reqs = problem.requests
    residents = sorted((r for r in reqs if r.resident),
                       key=lambda r: r.service_rank)
    waiting = sorted((r for r in reqs if not r.resident),
                     key=lambda r: r.service_rank)
    driver.check([r.service_rank for r in residents + waiting]
                 == list(range(len(reqs))),
                 f"{problem.problem_id}: service ranks are not running then "
                 "waiting positions")
    W = 1 if waiting else 0
    weights, span, penalty, later = {}, {}, {}, W
    for r in reversed(residents):
        a = 1 + later
        weights[r.request_id] = a
        span[r.request_id] = (
            min(r.prompt_tokens_remaining, problem.c_max, problem.b_max)
            if r.prefill_eligible else int(r.decode_eligible))
        penalty[r.request_id] = a if r.preemption_eligible else 0
        later += span[r.request_id] * a + penalty[r.request_id]
    return dict(
        order=[r.request_id for r in residents],
        waiting_order=[r.request_id for r in waiting], W=W, weights=weights,
        span=span, penalty=penalty, objective_range=later,
        waiting={r.request_id: Fraction(1, problem.b_max * (1 + j) ** 2)
                 if r.prefill_eligible else Fraction(0)
                 for j, r in enumerate(waiting)})


def progress_vector(intended, assignment):
    """Resident progress in running order: chunk length, 1 for a decode, 0
    idle, -1 preempted."""
    out = []
    for rid in intended["order"]:
        action = assignment.get(rid)
        out.append(0 if action is None else -1 if action == "preempt"
                   else 1 if action == "decode"
                   else int(action.split(":")[1]))
    return out


def exact_utility(intended, assignment):
    """Exact intended utility of an integer assignment."""
    total = Fraction(sum(
        (-intended["penalty"][rid] if p < 0 else intended["weights"][rid] * p)
        for rid, p in zip(intended["order"],
                          progress_vector(intended, assignment))))
    for rid, coef in intended["waiting"].items():
        action = assignment.get(rid)
        if action is not None and action.startswith("prefill:"):
            total += coef * int(action.split(":")[1])
    return total


def running_order_numerics(problem, intended):
    """Exact and float-converted dominance margins and coefficient scale of
    one running-order problem (diagnostic only)."""
    reqs = {r.request_id: r for r in problem.requests}
    order, weights = intended["order"], intended["weights"]
    mismatched, recorded = [], {}
    for rid in order:
        r = reqs[rid]
        values = [v for v, eligible in (
            (r.decode_utility, r.decode_eligible),
            (r.prefill_token_utility, r.prefill_eligible),
            (r.preemption_penalty, r.preemption_eligible)) if eligible]
        recorded[rid] = Fraction(values[0])
        if any(Fraction(v) != weights[rid] for v in values) or any(
                v != 0 for v, eligible in (
                    (r.decode_utility, r.decode_eligible),
                    (r.prefill_token_utility, r.prefill_eligible),
                    (r.preemption_penalty, r.preemption_eligible))
                if not eligible):
            mismatched.append(rid)
    for rid, coef in intended["waiting"].items():
        r = reqs[rid]
        if float(coef) != r.prefill_token_utility or (
                r.decode_utility or r.preemption_penalty):
            mismatched.append(rid)
    waiting_bound = (problem.b_max * max(
        (Fraction(reqs[rid].prefill_token_utility)
         for rid in intended["waiting"]), default=Fraction(0)))
    exact_margins, float_margins = [], []
    for k, rid in enumerate(order):
        later = order[k + 1:]
        exact_margins.append(weights[rid] - intended["W"] - sum(
            intended["span"][j] * weights[j] + intended["penalty"][j]
            for j in later))
        float_margins.append(recorded[rid] - waiting_bound - sum(
            intended["span"][j] * recorded[j]
            + (recorded[j] if reqs[j].preemption_eligible else 0)
            for j in later))
    positive = [v for r in problem.requests for v in (
        r.decode_utility, r.prefill_token_utility, r.preemption_penalty)
        if v > 0]
    scale = intended["objective_range"]
    ulp = math.ulp(float(scale)) if scale else 0.0
    waiting_coefs = [intended["waiting"][rid] for rid in intended["waiting_order"]]
    waiting_gaps = [a - b for a, b in zip(waiting_coefs, waiting_coefs[1:])]
    ratios = [Fraction(weights[a], weights[b]) for a, b in zip(order, order[1:])]
    return dict(
        weights={rid: weights[rid] for rid in order},
        max_weight=max(weights.values(), default=None),
        weights_exactly_representable=all(
            float(a) == a for a in weights.values()),
        recorded_mismatches=mismatched,
        exact_margins=exact_margins,
        float_margins=[_frac(m) for m in float_margins],
        min_float_margin=float(min(float_margins)) if float_margins else None,
        objective_range=scale,
        ulp_at_objective_range=ulp,
        unit_margin_relative=1 / scale if scale else None,
        comparison_tol_hides_unit_margin=(
            eip.COMPARISON_TOL * max(1.0, float(scale)) >= 1),
        ulp_exceeds_unit_margin=ulp >= 1,
        min_positive_coefficient=min(positive, default=None),
        max_coefficient=max(positive, default=None),
        coefficient_ratio=(max(positive) / min(positive) if positive
                           else None),
        min_positive_over_ulp=(min(positive) / ulp if positive and ulp
                               else None),
        min_adjacent_weight_ratio=float(min(ratios)) if ratios else None,
        min_adjacent_waiting_gap=(float(min(waiting_gaps)) if waiting_gaps
                                  else None),
        min_adjacent_waiting_gap_over_ulp=(
            float(min(waiting_gaps)) / ulp if waiting_gaps and ulp else None))


def lexicographic_enumeration(problem, intended):
    """Bounded enumeration of every legal useful plan (the planner test
    oracle): the lexicographically largest resident progress vector, the
    largest exact utility, and whether every exact-utility maximizer has
    that vector. ``None`` beyond ``ENUMERATION_LIMIT``."""
    import test_lp_relaxation_scheduler as planner_tests
    size = math.prod(len(planner_tests.oracle_options(r))
                     for r in problem.requests)
    if size > ENUMERATION_LIMIT:
        return dict(skipped=f"plan space {size} exceeds {ENUMERATION_LIMIT}")
    best_vector = best_exact = None
    maximizers = []
    legal_useful = 0
    for choice in planner_tests.oracle_enumerate(problem):
        legal, useful, _ = planner_tests.oracle_simulate(problem, choice)
        if not (legal and useful):
            continue
        legal_useful += 1
        assignment = {}
        for r, c in zip(problem.requests, choice):
            if c is not None:
                assignment[r.request_id] = (f"prefill:{c[1]}" if c[0] == "prefill"
                                            else c[0])
        vector = progress_vector(intended, assignment)
        exact = exact_utility(intended, assignment)
        best_vector = vector if best_vector is None else max(best_vector,
                                                             vector)
        if best_exact is None or exact > best_exact:
            best_exact, maximizers = exact, [vector]
        elif exact == best_exact:
            maximizers.append(vector)
    return dict(plan_space=size, legal_useful_plans=legal_useful,
                best_vector=best_vector,
                best_exact=None if best_exact is None else _frac(best_exact),
                exact_maximizers_have_best_vector=all(
                    v == best_vector for v in maximizers),
                _best_exact=best_exact)


def running_order_check(problem, candidates, enumerate_small=True):
    """Exact running-order checks of one problem. ``candidates`` maps a
    label to ``(assignment, recorded objective or None)``; the label
    ``integer`` is the claimed integer optimum. Labels (``findings``):
    ``counterexample`` (a feasible candidate with a better resident vector
    or exact utility than the claimed optimum), ``exact_tie``,
    ``tolerance_hides_difference`` (different exact utilities that the
    existing relative comparison tolerance calls equal),
    ``float_order_differs`` (float objectives order two candidates
    differently from their exact utilities), ``recorded_mismatch``,
    ``float_margin_nonpositive``, ``comparison_tol_hides_unit_margin``,
    ``ulp_exceeds_unit_margin``, ``enumeration_disagrees``, and
    ``dominance_violated`` (an exact-utility maximizer without the best
    vector)."""
    intended = intended_running_order(problem)
    numerics = running_order_numerics(problem, intended)
    out = dict(numerics=numerics, candidates={}, pairs=[], findings=[])
    for label, (assignment, recorded) in candidates.items():
        evaluation = eip.evaluate_assignment(problem, assignment)
        exact = exact_utility(intended, assignment)
        out["candidates"][label] = dict(
            actions=assignment, feasible=evaluation["feasible"],
            violated_rows=evaluation["violated_rows"],
            domain_violations=evaluation["domain_violations"],
            progress_vector=progress_vector(intended, assignment),
            exact_utility=_frac(exact), float_objective=evaluation["objective"],
            recorded_objective=recorded,
            float_minus_exact=float(Fraction(evaluation["objective"]) - exact),
            _exact=exact)
    findings = out["findings"]
    integer = out["candidates"].get("integer")
    if integer is not None:
        for label, other in out["candidates"].items():
            if label == "integer":
                continue
            diff = integer["_exact"] - other["_exact"]
            float_diff = integer["float_objective"] - other["float_objective"]
            vector = ("better" if other["progress_vector"]
                      > integer["progress_vector"] else "equal"
                      if other["progress_vector"] == integer["progress_vector"]
                      else "worse")
            same = other["actions"] == integer["actions"]
            tol = eip.COMPARISON_TOL * max(1.0, abs(integer["float_objective"]),
                                           abs(other["float_objective"]))
            pair = dict(other=label, same_actions=same,
                        other_feasible=other["feasible"],
                        other_vector_vs_integer=vector,
                        exact_integer_minus_other=_frac(diff),
                        float_integer_minus_other=float_diff,
                        existing_tolerance_calls_equal=abs(float_diff) <= tol)
            out["pairs"].append(pair)
            if not other["feasible"] or same:
                continue
            if vector == "better" or diff < 0:
                findings.append(f"counterexample:{label}")
            if diff == 0:
                findings.append(f"exact_tie:{label}")
            elif pair["existing_tolerance_calls_equal"]:
                findings.append(f"tolerance_hides_difference:{label}")
            if diff != 0 and (float_diff > 0) != (diff > 0):
                findings.append(f"float_order_differs:{label}")
    if numerics["recorded_mismatches"]:
        findings.append("recorded_mismatch")
    if numerics["min_float_margin"] is not None and numerics[
            "min_float_margin"] <= 0:
        findings.append("float_margin_nonpositive")
    for flag in ("comparison_tol_hides_unit_margin", "ulp_exceeds_unit_margin"):
        if numerics[flag]:
            findings.append(flag)
    if enumerate_small:
        enumeration = lexicographic_enumeration(problem, intended)
        best = enumeration.pop("_best_exact", None)
        out["enumeration"] = enumeration
        if "skipped" not in enumeration:
            if not enumeration["exact_maximizers_have_best_vector"]:
                findings.append("dominance_violated")
            if integer is not None and (
                    integer["progress_vector"] != enumeration["best_vector"]
                    or integer["_exact"] != best):
                findings.append("enumeration_disagrees")
    for record in out["candidates"].values():
        record.pop("_exact")
    return out


def running_order_decision(problem, decision, record, integer, method):
    """``decision_comparisons`` supplement for running-order problems: the
    integer optimum, the LP plan (LP methods), or the reference proposal."""
    candidates = {}
    if integer["status"] == "optimal":
        candidates["integer"] = (integer["actions"], integer["objective"])
    events = {e["call"]: e for e in decision.get("lp_events", [])}
    solved = events.get("solve_and_extract")
    if solved is not None and solved["result_type"] == "SchedulingSuccess":
        candidates["lp_plan"] = (eip.plan_actions(solved["result"]["plan"]),
                                 solved["result"]["plan"]["objective"])
    if method == "adapted_vllm" and decision.get("proposals"):
        candidates["reference"] = (
            reference_assignment(decision["proposals"][0]), None)
    out = running_order_check(problem, candidates)
    if integer["status"] != "optimal":
        out["findings"].append("integer_not_optimal")
    return out


def running_order_summary(records):
    """Counts and extremes of one run's running-order supplements."""
    out = dict(decisions=0, findings={}, finding_indices={}, max_weight=None,
               max_objective_range=None, min_positive_coefficient=None,
               max_coefficient_ratio=None, min_float_margin=None,
               enumerated=0, enumeration_skipped=0, integer_vs_other={})
    for r in records:
        sup = r.get("supplement")
        if not sup:
            continue
        out["decisions"] += 1
        n = sup["numerics"]
        for key, value, pick in (
                ("max_weight", n["max_weight"], max),
                ("max_objective_range", n["objective_range"], max),
                ("min_positive_coefficient", n["min_positive_coefficient"], min),
                ("max_coefficient_ratio", n["coefficient_ratio"], max),
                ("min_float_margin", n["min_float_margin"], min)):
            if value is not None:
                out[key] = value if out[key] is None else pick(out[key], value)
        if "enumeration" in sup:
            if "skipped" in sup["enumeration"]:
                out["enumeration_skipped"] += 1
            else:
                out["enumerated"] += 1
        for finding in sup["findings"]:
            out["findings"][finding] = out["findings"].get(finding, 0) + 1
            out["finding_indices"].setdefault(finding, []).append(r["index"])
        for pair in sup["pairs"]:
            key = (f"{pair['other']}: "
                   + ("same_actions" if pair["same_actions"] else
                      "infeasible" if not pair["other_feasible"] else
                      f"vector_{pair['other_vector_vs_integer']}"))
            out["integer_vs_other"][key] = out["integer_vs_other"].get(
                key, 0) + 1
    return out


def queue_input_checks(case):
    """The 96-request cases' generated inputs and native capacity rule."""
    c = driver.case_inputs(case)
    names = list(c["prompts"])
    ids = [t for n in names for t in c["prompts"][n]]
    expected = {n: list(range(1000 + 256 * k, 1000 + 256 * k
                              + [8, 16, 17, 32, 64, 128][k % 6]))
                for k, n in enumerate(names)}
    checks = dict(
        count=len(names) == 96,
        prompt_ids=c["prompts"] == expected,
        disjoint=len(ids) == len(set(ids)),
        outputs=[driver.max_tokens(c, n) for n in names]
        == [[2, 4, 8][k % 3] for k in range(96)],
        arrivals=set(c["arrivals"].values()) == {0},
        caps=(c["token_budget"], c["b_max"], c["c_max"], c["s_max"])
        == (128, 128, 128, 8),
        bound=c["max_decisions"] == 8 * 96,
        pool_and_context=(c["num_gpu_blocks"], c["max_model_len"])
        == {"queue96_burst": (128, 256),
            "queue96_burst_pool12": (12, 192)}[case],
        longest_request=max(len(p) + driver.max_tokens(c, n)
                            for n, p in c["prompts"].items()) <= c["max_model_len"])
    try:
        checks["native_capacity_blocks"] = driver.check_native_capacity(c)
    except driver.CheckFailure as err:
        checks["native_capacity_blocks"] = f"failed: {err}"
    checks["passed"] = all(v is True for k, v in checks.items()
                           if k != "native_capacity_blocks") and isinstance(
        checks["native_capacity_blocks"], int)
    return checks


# Single-state diagnostics: the existing four, five targeted states, and
# four numerical-stress states. ``expect`` holds the integer-policy and
# mapped-state expectations (an unmet one is a finding); ``observe`` holds
# predictions about the reference (reported, not required); ``alternatives``
# are named LP-contract plans evaluated alongside.
RUNNING_ORDER_DIAGNOSTICS = dict(
    {name: dict(spec) for name, spec in DIAGNOSTIC_STATES.items()},
    earlier_decode_vs_later_prefill=dict(
        note="A (16,16,1) holds one block and needs one more; B (16,8,0) "
        "holds one block with 8 prefill tokens left; no waiting requests",
        residents=[(16, 16, 1), (16, 8, 0)], waiting=[], pool=2,
        max_model_len=32, limit=2, b_max=8, c_max=8, s_max=2, budget=8,
        expect=dict(free_blocks=0, decode_demand={"0": 1},
                    physical={"0": 1, "1": 1}, weights={"0": 10, "1": 1},
                    integer_actions={"0": "decode", "1": "preempt"},
                    integer_utility=9),
        alternatives=dict(prefill_b={"1": "prefill:8"})),
    second_running_decode=dict(
        note="A and B (16,16,1) each need one more block; C (16,8,0); one "
        "free block",
        residents=[(16, 16, 1), (16, 16, 1), (16, 8, 0)], waiting=[], pool=4,
        max_model_len=64, limit=3, b_max=8, c_max=8, s_max=3, budget=8,
        expect=dict(free_blocks=1, decode_demand={"0": 1, "1": 1},
                    physical={"0": 1, "1": 1, "2": 1},
                    weights={"0": 30, "1": 10, "2": 1},
                    integer_actions={"0": "decode", "1": "decode",
                                     "2": "preempt"},
                    integer_utility=39),
        alternatives=dict(head_decode_and_c_prefill={"0": "decode",
                                                      "2": "prefill:7"})),
    large_earlier_victim=dict(
        note="A (32,32,1) holds two blocks; B and C (16,16,1) hold one each; "
        "all need one new block; zero free",
        residents=[(32, 32, 1), (16, 16, 1), (16, 16, 1)], waiting=[], pool=4,
        max_model_len=64, limit=3, b_max=3, c_max=3, s_max=3, budget=3,
        expect=dict(free_blocks=0, decode_demand={"0": 1, "1": 1, "2": 1},
                    physical={"0": 2, "1": 1, "2": 1},
                    weights={"0": 9, "1": 3, "2": 1},
                    integer_actions={"0": "decode", "2": "preempt"},
                    integer_utility=8),
        observe=dict(reference_actions={"0": "decode", "1": "preempt",
                                        "2": "preempt"}),
        alternatives=dict(evict_large_earlier={"0": "preempt", "1": "decode",
                                               "2": "decode"})),
    blocked_waiting_head=dict(
        note="one decode-ready resident (8,8,0); waiting prompts 48 then 8",
        residents=[(8, 8, 0)], waiting=[48, 8], pool=3, max_model_len=48,
        limit=3, b_max=8, c_max=8, s_max=3, budget=8,
        expect=dict(free_blocks=2, decode_demand={"0": 0}, weights={"0": 2},
                    integer_actions={"0": "decode", "2": "prefill:7"},
                    integer_utility=Fraction(71, 32)),
        observe=dict(reference_actions={"0": "decode"})),
    waiting_action_cap=dict(
        note="no residents; waiting prompts 1 then 8; one action",
        residents=[], waiting=[1, 8], pool=2, max_model_len=32, limit=2,
        b_max=8, c_max=8, s_max=1, budget=8,
        expect=dict(free_blocks=2, integer_actions={"1": "prefill:8"},
                    integer_utility=Fraction(1, 4)),
        alternatives=dict(head_only={"0": "prefill:1"})),
    **{f"numerical_stress_r{R}": dict(
        note=f"{R} residents (256,128,0), 128 prefill tokens left and 16 "
        "blocks each; one waiting 8-token prompt",
        residents=[(256, 128, 0)] * R, waiting=[8], pool=16 * R + 1,
        max_model_len=272, limit=R, b_max=128, c_max=128, s_max=8,
        budget=128,
        expect=dict(free_blocks=1,
                    physical={str(i): 16 for i in range(R)},
                    # Closed form of the recurrence here: every resident has
                    # m = 128 and gamma = a, so a_i = 2 * 130 ** (R - 1 - i).
                    weights={str(i): 2 * 130 ** (R - 1 - i) for i in range(R)},
                    integer_actions={"0": "prefill:128"},
                    integer_utility=128 * 2 * 130 ** (R - 1)))
       for R in (1, 2, 4, 8)},
)
# Independently stated head weight of the eight-resident stress state.
STRESS_R8_HEAD_WEIGHT = 1_254_970_340_000_000


def run_running_order_diagnostic(name, spec):
    """One single-state diagnostic under running-order coefficients. Solver,
    mapping, and executor outcomes are recorded, never raised; only accepted
    plans are executed, each on its own native build."""
    import lp_relaxation_scheduler as lrs
    import lpserve_plan_execution as lpe
    import lpserve_state_mapping as lsm
    policy = functools.partial(lsm.running_order_utility, c_max=spec["c_max"],
                               b_max=spec["b_max"])

    def mapped(scheduler):
        return lsm.map_scheduler_state(
            scheduler, snapshot_time=time.monotonic(), b_max=spec["b_max"],
            c_max=spec["c_max"], s_max=spec["s_max"],
            coefficient_policy=policy, rank_profile=lsm.vllm_oriented_ranks,
            numerical_policy=_numerical_policy())

    def block_counts(scheduler):
        manager = scheduler.block_manager
        return dict(free=manager.get_num_free_gpu_blocks(), physical={
            str(seq.seq_id): len(manager.get_block_table(seq))
            for seq in scheduler.running})

    def executed(outputs):
        if outputs is None:
            return None
        if isinstance(outputs, lrs.Failure):
            return dict(result_type="Failure", failure=driver.to_jsonable(outputs))
        return dict(result_type=type(outputs).__name__,
                    outputs=driver.outputs_record(outputs))

    record = dict(name=name, spec={k: v for k, v in spec.items()
                                   if k not in ("expect", "observe")},
                  findings=[])
    try:
        driver.check_native_capacity(dict(num_gpu_blocks=spec["pool"],
                                          max_model_len=spec["max_model_len"]))
        record["native_capacity"] = "satisfied"
    except driver.CheckFailure as err:
        record["native_capacity"] = f"violated: {err}"
        record["findings"].append("native_capacity_violated")
    lp_sched = build_state(spec, "lp")
    lp_blocks = dict(before=block_counts(lp_sched))
    lp_snap = mapped(lp_sched)
    if isinstance(lp_snap, lsm.MappingFailure):
        record.update(mapping_failure=driver.to_jsonable(lp_snap))
        record["findings"].append("mapping_failure")
        return record
    problem = lp_snap.lp_problem
    lp_result = lrs.solve_and_extract(problem)
    lp_outputs = (lpe.execute_plan(lp_sched, lp_snap, lp_result)
                  if isinstance(lp_result, lrs.SchedulingSuccess) else None)
    lp_blocks["after"] = block_counts(lp_sched)
    ilp_sched = build_state(spec, "lp")
    ilp_blocks = dict(before=block_counts(ilp_sched))
    ilp_snap = mapped(ilp_sched)
    same_problem = (not isinstance(ilp_snap, lsm.MappingFailure)
                    and ilp_snap.lp_problem == problem)
    if not same_problem:
        record["findings"].append("separate_builds_differ")
    integer = eip.solve_integer_plan(problem)
    ilp_outputs = (lpe.execute_plan(ilp_sched, ilp_snap, lrs.SchedulingSuccess(
        integer.problem_id, integer.plan, None, None))
        if isinstance(integer, eip.IntegerOptimum) and same_problem else None)
    ilp_blocks["after"] = block_counts(ilp_sched)
    ref = build_state(spec, "adapted_vllm")
    proposal = driver.to_jsonable(ref._propose(time.monotonic()))
    candidates = {}
    if isinstance(integer, eip.IntegerOptimum):
        candidates["integer"] = (eip.plan_actions(integer.plan),
                                 integer.plan.objective)
    else:
        record["findings"].append("integer_not_optimal")
    if isinstance(lp_result, lrs.SchedulingSuccess):
        candidates["lp_plan"] = (eip.plan_actions(lp_result.plan),
                                 lp_result.plan.objective)
    else:
        record["findings"].append("lp_failure")
    candidates["reference"] = (reference_assignment(proposal), None)
    for alt, assignment in spec.get("alternatives", {}).items():
        candidates[f"alternative:{alt}"] = (assignment, None)
    for label, outputs in (("lp", lp_outputs), ("integer", ilp_outputs)):
        if isinstance(outputs, lrs.Failure):
            record["findings"].append(f"executor_rejected_{label}_plan")
    check = running_order_check(problem, candidates)
    record["findings"] += check["findings"]
    mapped_requests = {r.request_id: dict(
        resident=r.ownership == lsm.OWNERSHIP_RUNNING,
        prompt_tokens_remaining=r.prompt_tokens_remaining,
        output_len=r.output_len, logical_blocks=r.logical_block_count,
        physical_blocks=r.physical_block_count, decode_demand=r.decode_demand,
        prefill_fixed_charge=r.prefill_fixed_charge)
        for r in lp_snap.requests}
    integer_record = check["candidates"].get("integer")
    observed = dict(
        free_blocks=problem.m_free,
        decode_demand={rid: m["decode_demand"] for rid, m in
                       mapped_requests.items() if m["resident"]},
        physical={rid: m["physical_blocks"] for rid, m in
                  mapped_requests.items() if m["resident"]},
        weights=check["numerics"]["weights"],
        integer_actions=integer_record["actions"] if integer_record else None,
        integer_utility=(Fraction(integer_record["exact_utility"]["exact"])
                         if integer_record else None),
        reference_actions=check["candidates"]["reference"]["actions"])
    expectations = {}
    for kind in ("expect", "observe"):
        for key, want in spec.get(kind, {}).items():
            got = observed[key]
            if isinstance(want, dict) and isinstance(got, dict) and key in (
                    "decode_demand", "physical", "weights"):
                met = all(got.get(k) == v for k, v in want.items())
            else:
                met = got == want
            expectations[key] = dict(
                kind=kind, met=met, **{k: str(v) if isinstance(v, Fraction)
                                       else v for k, v in (("expected", want),
                                                           ("observed", got))})
            if kind == "expect" and not met:
                record["findings"].append(f"expectation_unmet:{key}")
    if name == "numerical_stress_r8":
        met = check["numerics"]["weights"].get("0") == STRESS_R8_HEAD_WEIGHT
        expectations["stated_head_weight"] = dict(
            kind="expect", expected=STRESS_R8_HEAD_WEIGHT,
            observed=check["numerics"]["weights"].get("0"), met=met)
        if not met:
            record["findings"].append("expectation_unmet:stated_head_weight")
    record.update(
        mapped_requests=mapped_requests,
        block_counts=dict(lp=lp_blocks, integer=ilp_blocks),
        coefficients={r.request_id: dict(
            decode=r.decode_utility, prefill_token=r.prefill_token_utility,
            penalty=r.preemption_penalty, service_rank=r.service_rank,
            victim_rank=r.victim_rank, upper_bound=r.prefill_upper_bound)
            for r in problem.requests},
        free_blocks=problem.m_free, watermark=problem.admission_watermark,
        problem=driver.to_jsonable(problem),
        lp=driver.to_jsonable(lp_result),
        lp_relaxed_values=(
            {d.request_id: [d.x, d.y, d.prefill_indicator, d.z]
             for d in lp_result.relaxed.decisions}
            if isinstance(lp_result, lrs.SchedulingSuccess) else None),
        integer=driver.to_jsonable(integer),
        lp_vs_integer=eip.compare_same_state(problem, lp_result, integer),
        reference_proposal=proposal,
        reference_vs_integer=compare_reference(problem, proposal, integer),
        executed=dict(lp=executed(lp_outputs), integer=executed(ilp_outputs)),
        running_order=check, expectations=expectations)
    return record


def write_json_or_gz(path, value):
    if path.suffix == ".gz":
        write_json_gz(path, value)
    else:
        path.write_text(json.dumps(value, indent=1))


def _victim_patterns(service):
    counts = {}
    for v in service["victims"]:
        counts[v["victim"]] = counts.get(v["victim"], 0) + 1
    return dict(
        non_tail_victims=sum(1 for v in service["victims"]
                             if v["running_position"] != v["running_count"] - 1),
        repeat_victims={k: n for k, n in counts.items() if n > 1})


def running_order_report(rows, divergences, diagnostics, status):
    def label(s):
        return f"{s['case']}_limit{s['resident_limit']}"
    text = queue_report(
        rows, divergences,
        title="Running-order batch (CPU, native central/worker replay)",
        label=label, reference_policy="running-order")
    lines = ["", "Running-order exact checks (decisions with running-order "
             "coefficients; the control LP is not included)"]
    for r in rows:
        ro = r.get("running_order")
        if not ro:
            continue
        lines.append(
            f"  {label(r['summary'])} {r['summary']['method']}: "
            f"{ro['decisions']} decisions, enumerated {ro['enumerated']} "
            f"(skipped {ro['enumeration_skipped']}); max weight "
            f"{ro['max_weight']}, max objective range "
            f"{ro['max_objective_range']}, min positive coefficient "
            f"{_fmt(ro['min_positive_coefficient'])}, max coefficient ratio "
            f"{_fmt(ro['max_coefficient_ratio'])}, min float margin "
            f"{_fmt(ro['min_float_margin'])}; integer vs other "
            f"{ro['integer_vs_other']}; findings {ro['findings']}")
        for finding, indices in sorted(ro["finding_indices"].items()):
            if not finding.startswith(("comparison_tol", "ulp_")):
                lines.append(f"      {finding}: decisions {indices}")
    lines += ["", "Victim patterns (non-tail victims; requests preempted "
              "more than once)"]
    for r in rows:
        p = _victim_patterns(r["service"])
        if r["service"]["victims"]:
            lines.append(f"  {label(r['summary'])} {r['summary']['method']}: "
                         f"{p}")
    lines += ["", "Single-state diagnostics (running-order coefficients)"]
    for d in diagnostics:
        if "error" in d:
            lines.append(f"  {d['name']}: HARNESS ERROR {d['error']['message']}")
            continue
        if "mapping_failure" in d:
            lines.append(f"  {d['name']}: mapping failure "
                         f"{d['mapping_failure']['reason']}")
            continue
        c = d["running_order"]["candidates"]
        n = d["running_order"]["numerics"]
        s = d["lp_vs_integer"]
        cells = "; ".join(
            f"{k} {v['actions']} vec {v['progress_vector']} exact "
            f"{v['exact_utility']['exact']} float {_fmt(v['float_objective'])}"
            f"{'' if v['feasible'] else ' INFEASIBLE ' + str(v['violated_rows'] + v['domain_violations'])}"
            for k, v in c.items())
        lines.append(f"  {d['name']}: weights {n['weights']}; free "
                     f"{d['free_blocks']}; LP relaxed "
                     f"{_fmt(s.get('lp_relaxed_objective'))} "
                     f"({s['attribution']}); {cells}")
        lines.append(f"      reference vs integer: "
                     f"{d['reference_vs_integer']['classification']}; "
                     f"enumeration "
                     f"{d['running_order'].get('enumeration')}; findings "
                     f"{d['findings']}")
        unmet = {k: v for k, v in d["expectations"].items() if not v["met"]}
        lines.append(f"      expectations met: "
                     f"{sorted(k for k, v in d['expectations'].items() if v['met'])}"
                     f"; not met: {unmet}")
        lines.append(f"      numerics: max coefficient "
                     f"{_fmt(n['max_coefficient'])}, min positive "
                     f"{_fmt(n['min_positive_coefficient'])}, ratio "
                     f"{_fmt(n['coefficient_ratio'])}, ulp at objective range "
                     f"{_fmt(n['ulp_at_objective_range'])}, min float margin "
                     f"{_fmt(n['min_float_margin'])}, recorded mismatches "
                     f"{n['recorded_mismatches']}")
    lines += ["", "Aggregate status", f"  {json.dumps(status, indent=1)}"]
    return text + "\n".join(lines) + "\n"


# Findings that are numerical contradictions or evidence problems (not
# experimental outcomes): any of them makes the batch FAIL.
CONTRADICTIONS = ("counterexample", "float_order_differs", "recorded_mismatch",
                  "float_margin_nonpositive", "enumeration_disagrees",
                  "dominance_violated", "integer_not_optimal",
                  "expectation_unmet", "mapping_failure", "lp_failure",
                  "executor_rejected", "separate_builds_differ",
                  "native_capacity_violated")


def running_order_main(out):
    import test_compare_scheduler_gpu as cpu
    cpu.setUpModule()
    record = dict(
        argv=sys.argv, batch="running_order", methods=RUNNING_ORDER_METHODS,
        workloads=[list(w) for w in RUNNING_ORDER_WORKLOADS],
        cases={case: driver.case_inputs(case) for case in dict.fromkeys(
            c for c, _ in RUNNING_ORDER_WORKLOADS)},
        running_order=driver.RUNNING_ORDER_FORMULA,
        reciprocal_square="w = 1 / (1 + r) ** 2; q = min(remaining, c_max, "
        "b_max); decode w, prefill token w / q, penalty w (control, unchanged)",
        reference_diagnostic_policy="running_order_utility",
        enumeration_limit=ENUMERATION_LIMIT,
        integer_solver=dict(solver=eip.SOLVER, options=eip.SOLVER_OPTIONS,
                            conversion_tol=eip.CONVERSION_TOL,
                            comparison_tol=eip.COMPARISON_TOL),
        queue96_input_checks={case: queue_input_checks(case)
                              for case in driver.RUNNING_ORDER_QUEUE_CASES},
        provenance=driver.provenance_record(),
        environment=driver.environment_record())
    problems = [f"input check {case}" for case, c in
                record["queue96_input_checks"].items() if not c["passed"]]
    rows, traces, harness_errors = [], {}, []
    for case, limit in RUNNING_ORDER_WORKLOADS:
        for method in RUNNING_ORDER_METHODS:
            key = f"{case}_limit{limit}"
            print(f"rollout {key} {method}", flush=True)
            run_dir = out / "rollouts" / key / method
            run_dir.mkdir(parents=True)
            try:
                summary, trace = run_queue_rollout(
                    cpu, method, case, run_dir, limit,
                    driver.running_order_policy
                    if method == "adapted_vllm" else None)
                new = method in RUNNING_ORDER_PROBLEM_METHODS
                analysis = analyse_run(trace, method, case,
                                       running_order_decision if new else None)
                if new:
                    analysis["running_order"] = running_order_summary(
                        analysis["records"])
            except Exception as err:  # harness error: kept, others continue
                harness_errors.append(dict(run=f"{key} {method}",
                                           type=type(err).__name__,
                                           message=str(err),
                                           traceback=traceback.format_exc()))
                print(f"  -> HARNESS ERROR {err!r}", flush=True)
                continue
            (run_dir / "summary.json").write_text(json.dumps(
                driver.to_jsonable(dict(
                    summary, coverage=analysis["coverage"],
                    service=dict(analysis["service"],
                                 **_victim_patterns(analysis["service"])),
                    comparisons=analysis["comparisons"],
                    running_order=analysis.get("running_order"))), indent=1))
            write_json_gz(run_dir / "comparisons.json.gz",
                          driver.to_jsonable(analysis["records"]))
            rows.append(dict(summary=summary, **analysis))
            traces[key, method] = trace
            print(f"  -> valid {summary['validation_passed']} outcome "
                  f"{summary['workload_outcome']} decisions "
                  f"{summary['decisions']} coverage complete "
                  f"{analysis['coverage']['complete']} findings "
                  f"{(analysis.get('running_order') or {}).get('findings')}",
                  flush=True)
    by_key = {(f"{r['summary']['case']}_limit{r['summary']['resident_limit']}",
               r["summary"]["method"]): r["summary"] for r in rows}
    divergences = {}
    for case, limit in RUNNING_ORDER_WORKLOADS:
        key = f"{case}_limit{limit}"
        for a, b in RUNNING_ORDER_PAIRS:
            if (key, a) in by_key and (key, b) in by_key:
                divergences[f"{key} {a} vs {b}"] = dict(
                    emitted=first_divergence(by_key[key, a], by_key[key, b],
                                             traces[key, a], traces[key, b]),
                    first_selection_difference=first_selection_divergence(
                        by_key[key, a], by_key[key, b]))
    (out / "divergences.json").write_text(json.dumps(
        driver.to_jsonable(divergences), indent=1))
    diagnostics = []
    for name, spec in RUNNING_ORDER_DIAGNOSTICS.items():
        print(f"diagnostic {name}", flush=True)
        try:
            diagnostics.append(driver.to_jsonable(
                run_running_order_diagnostic(name, spec)))
        except Exception as err:  # harness error: kept, others continue
            diagnostics.append(dict(name=name, error=dict(
                type=type(err).__name__, message=str(err),
                traceback=traceback.format_exc())))
        print(f"  -> findings {diagnostics[-1].get('findings')}", flush=True)
    (out / "diagnostics.json").write_text(json.dumps(
        driver.to_jsonable(diagnostics), indent=1))

    for r in rows:
        tag = f"{r['summary']['case']}_limit{r['summary']['resident_limit']} " \
              f"{r['summary']['method']}"
        if not r["summary"]["validation_passed"]:
            problems.append(f"{tag}: rollout failure "
                            f"{(r['summary']['failure'] or {}).get('category')}")
        if not r["coverage"]["complete"]:
            problems.append(f"{tag}: incomplete comparison coverage")
        if r["coverage"]["time_limits"] or r["coverage"]["other_failures"]:
            problems.append(f"{tag}: non-optimal integer solves")
        if not r["consistent"]:
            problems.append(f"{tag}: inconsistent same-state comparison")
        for finding in (r.get("running_order") or {}).get("findings", {}):
            if finding.startswith(CONTRADICTIONS):
                problems.append(f"{tag}: {finding}")
    for d in diagnostics:
        if "error" in d:
            problems.append(f"diagnostic {d['name']}: harness error")
        for finding in d.get("findings", []):
            if finding.startswith(CONTRADICTIONS):
                problems.append(f"diagnostic {d['name']}: {finding}")
    problems += [f"harness error: {e['run']}" for e in harness_errors]
    status = dict(runs_expected=len(RUNNING_ORDER_WORKLOADS)
                  * len(RUNNING_ORDER_METHODS), runs_completed=len(rows),
                  validation_passed=sum(r["summary"]["validation_passed"]
                                        for r in rows),
                  outcomes={o: sum(r["summary"]["workload_outcome"] == o
                                   for r in rows)
                            for o in (driver.DRAINED, driver.BOUNDED_UNFINISHED,
                                      driver.FAILURE)},
                  diagnostics=len(diagnostics), problems=problems)
    status["result"] = ("PASS" if not problems and len(rows)
                        == status["runs_expected"] else "FAIL")
    text = running_order_report(rows, divergences, diagnostics, status)
    (out / "comparison.txt").write_text(text)
    print(text, flush=True)
    record.update(all_checks_passed=status["result"] == "PASS", status=status,
                  harness_errors=harness_errors)
    (out / "run_record.json").write_text(json.dumps(
        driver.to_jsonable(record), indent=1))
    print("RESULT:", status["result"], flush=True)
    return 0 if status["result"] == "PASS" else 1


BATCHES = ("coordinated_policy", "queue_growth", "retained_runs",
           "running_order")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch", choices=BATCHES, default=BATCHES[0],
                        help="coordinated_policy (default): the coordinated "
                        "batch; queue_growth: the reciprocal-square "
                        "queue-growth batch; retained_runs: offline "
                        "comparisons of retained runs (--runs); "
                        "running_order: the running-order batch")
    parser.add_argument("--runs", nargs="+", default=[],
                        help="retained_runs: run directories holding "
                        "summary.json and decision_trace.json[.gz]")
    parser.add_argument("--cases", nargs="+", default=None,
                        choices=driver.QUEUE_GROWTH_CASES,
                        help="queue_growth: run only these cases (default: "
                        "all six)")
    parser.add_argument("--cpu-batch", default=None,
                        help="retained_runs: optional queue_growth output "
                        "directory whose rollouts the runs are checked "
                        "against")
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    out = Path(args.output_dir)
    if (out / "run_record.json").exists():
        parser.error(f"{out} already holds run artifacts")
    if (args.batch == "retained_runs") != bool(args.runs) or (
            args.cpu_batch and args.batch != "retained_runs"):
        parser.error("--runs (and --cpu-batch) go with --batch retained_runs "
                     "only, which requires --runs")
    if args.cases and args.batch != "queue_growth":
        parser.error("--cases goes with --batch queue_growth only")
    out.mkdir(parents=True, exist_ok=True)
    if args.batch == "queue_growth":
        return queue_growth_main(
            out, tuple(c for c in driver.QUEUE_GROWTH_CASES
                       if c in (args.cases or driver.QUEUE_GROWTH_CASES)))
    if args.batch == "retained_runs":
        return retained_runs_main(out, args.runs, args.cpu_batch)
    if args.batch == "running_order":
        return running_order_main(out)

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
