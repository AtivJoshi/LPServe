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
"""

import argparse
import contextlib
import functools
import io
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


def decision_comparisons(trace, method):
    """One same-state comparison record per reached decision of a retained
    (JSON) trace, plus coverage counts.

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


def queue_report(rows, divergences):
    lines = ["Queue-growth batch (CPU, native central/worker replay; "
             "resident limit 8)", "",
             "Rollouts"]
    head = ("case", "method", "valid", "outcome", "dec", "done", "preempt",
            "after_out", "adm+victim", "discarded", "replayed", "expanded",
            "events", "overgen", "res_omit", "max_streak", "adm_inv",
            "done_inv")
    lines.append(" | ".join(head))
    for r in rows:
        s, v = r["summary"], r["service"]
        lines.append(" | ".join(_fmt(x) for x in (
            s["case"], s["method"], s["validation_passed"],
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
            r["summary"]["case"], r["summary"]["method"],
            k["reached_decisions"], k["records"], k["valid_problems"],
            k["no_valid_problem"], k["attempted_solves"],
            k["reused_integer_results"], k["optimal"], k["time_limits"],
            k["other_failures"], k["comparison_errors"], k["missing_records"],
            k["round_trip_failures"], k["complete"])))
    lines += ["", "Same-state results (identical problems only; objectives "
              "of different policies are not compared)"]
    for r in rows:
        m, s = r["comparisons"], r["summary"]
        tag = f"  {s['case']} {s['method']}:"
        solver = (f" integer raw statuses {m['integer_raw_statuses']}, max "
                  f"integrality deviation "
                  f"{_fmt(m['max_integrality_deviation'])}")
        if s["method"] == "adapted_vllm":
            lines.append(f"{tag} reference vs reciprocal-square integer "
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
            f"  {s['case']} {s['method']}: max rank {m['max_service_rank']}, "
            f"min positive coefficient {_fmt(m['min_positive_coefficient'])}, "
            f"min gaps { {k: float(f'{v:.3g}') for k, v in m['min_gaps'].items()} }, "
            f"exact ties {m['exact_ties']}, max offline solve "
            f"{_fmt(m['solve_seconds_max'])} s")
    lines += ["", "Victims: [decision, victim, running position / count, "
              "had output, reasons, admitted in the same decision]"]
    for r in rows:
        victims = r["service"]["victims"]
        if victims:
            lines.append(f"  {r['summary']['case']} {r['summary']['method']}:")
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
        lines.append(f"  {r['summary']['case']} {r['summary']['method']}: "
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


def run_queue_rollout(cpu, method, case, run_dir):
    runtime = cpu.CpuRuntime(method, QUEUE_LIMIT, case)
    trace, failure = {}, None
    if method == "adapted_vllm":
        trace["reference_diagnostic_problems"] = (
            driver.capture_reference_problems(runtime, case))
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            outcome = driver.run_comparison(method, QUEUE_LIMIT, runtime, trace,
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
    summary = rollout_summary(method, case, QUEUE_LIMIT,
                              driver.case_inputs(case), trace, outcome, failure)
    return summary, trace


def analyse_run(trace, method, case):
    records, coverage = decision_comparisons(trace, method)
    return dict(records=records, coverage=coverage,
                comparisons=comparison_summary(records, method),
                service=service_record(trace, case),
                consistent=all(comparison_consistent(r) for r in records))


def queue_growth_main(out):
    import test_compare_scheduler_gpu as cpu
    cpu.setUpModule()
    record = dict(argv=sys.argv, batch="queue_growth",
                  runs=[list(r) for r in QUEUE_RUNS], resident_limit=QUEUE_LIMIT,
                  cases={case: driver.case_inputs(case)
                         for case in driver.QUEUE_GROWTH_CASES},
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
    for case, method in QUEUE_RUNS:
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
    for case in driver.QUEUE_GROWTH_CASES:
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


BATCHES = ("coordinated_policy", "queue_growth", "retained_runs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch", choices=BATCHES, default=BATCHES[0],
                        help="coordinated_policy (default): the coordinated "
                        "batch; queue_growth: the reciprocal-square "
                        "queue-growth batch; retained_runs: offline "
                        "comparisons of retained runs (--runs)")
    parser.add_argument("--runs", nargs="+", default=[],
                        help="retained_runs: run directories holding "
                        "summary.json and decision_trace.json[.gz]")
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
    out.mkdir(parents=True, exist_ok=True)
    if args.batch == "queue_growth":
        return queue_growth_main(out)
    if args.batch == "retained_runs":
        return retained_runs_main(out, args.runs, args.cpu_batch)

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
