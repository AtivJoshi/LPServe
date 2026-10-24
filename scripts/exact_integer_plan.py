"""CPU-only experimental integer optimization of the one-step scheduling ILP.

This is a comparison instrument, not a scheduler: it is not registered, has no
``EngineArgs`` configuration, and is never a fallback for the LP method. It
solves the integer program of the same ``LPProblem`` the LP method uses. The
objective, bounds, and every row come from ``lp_relaxation_scheduler._build_lp``
unchanged; only the domains differ: every variable, including prefill-token
counts, is integer, and the indicator, decode, and preemption variables keep
their 0-1 (or 0-0) bounds.

Solver: ``scipy.optimize.milp`` (HiGHS) with the visible provisional options
``SOLVER_OPTIONS`` (``mip_rel_gap`` 0, presolve on, ``time_limit`` 10 seconds
per solve). "Optimal" means solver-reported optimality within HiGHS's
numerical tolerances, not an exact-arithmetic certificate.

A solver vector becomes a plan only if the status is optimal, every value is
finite and within ``CONVERSION_TOL`` of an integer, the reported objective
agrees with the recomputed one, the rounded indicators equal 1{x > 0}, every
row holds exactly at the integer point, and the complete plan passes
``validate_integer_plan``. A materially fractional vector is never rounded
into a plan. A time limit or other non-optimal status is a failure; any
incumbent and bound are kept only as labeled diagnostics. No call retries.

``integer_decision`` drives one native decision from an integer optimum:
capture once (``map_scheduler_state``, same coefficients), solve once,
validate, and execute through ``lpserve_plan_execution.execute_plan``. No LP
is solved to construct its inputs.
"""

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

import lp_relaxation_scheduler as lrs

SOLVER = "scipy.optimize.milp (HiGHS)"
MIP_REL_GAP = 0.0
TIME_LIMIT_SECONDS = 10.0
SOLVER_OPTIONS = (("mip_rel_gap", MIP_REL_GAP), ("presolve", True),
                  ("time_limit", TIME_LIMIT_SECONDS))
# Largest accepted distance of a solver value from its integer (provisional;
# equal to HiGHS's default mip_feasibility_tolerance).
CONVERSION_TOL = 1e-6
# Tolerance for comparing objectives of different methods on one problem
# (provisional; equal to the numerical policy's feasibility tolerance, which
# exceeds the 1e-9 objective tolerances because the LP solve is itself only
# optimal to HiGHS's 1e-7 tolerances).
COMPARISON_TOL = 1e-7

STAGE_INTEGER_SOLVER = "integer_solver"
STAGE_INTEGER_CONVERSION = "integer_conversion"


@dataclass(frozen=True)
class IntegerOptimum:
    """A validated plan from a solver-reported integer optimum."""

    problem_id: str
    plan: lrs.IntegerPlan
    raw_vector: tuple


@dataclass(frozen=True)
class IntegerSolveFailure:
    """No accepted optimum. ``incumbent_vector`` and ``incumbent_objective``
    (maximization) are diagnostics only and never a plan."""

    problem_id: Optional[str]
    stage: str
    category: str
    reason: str
    raw_status: Optional[int] = None
    message: Optional[str] = None
    incumbent_vector: Optional[tuple] = None
    incumbent_objective: Optional[float] = None
    dual_bound: Optional[float] = None
    reported_gap: Optional[float] = None
    node_count: Optional[int] = None
    options: tuple = SOLVER_OPTIONS


def _maybe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _maybe_int(value):
    return int(value) if isinstance(value, (int, np.integer)) else None


def solve_integer_plan(problem):
    """Solve the ILP of ``problem`` once; ``IntegerOptimum`` or
    ``IntegerSolveFailure``. Never raises for solver or numerical outcomes."""
    pid = getattr(problem, "problem_id", None)
    canonical = lrs.validate_problem(problem)
    if isinstance(canonical, lrs.Failure):
        return IntegerSolveFailure(pid, canonical.stage, canonical.category,
                                   canonical.reason)
    c, rows, rhs, bounds, _ = lrs._build_lp(canonical)
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(rows))
            and np.all(np.isfinite(rhs))):
        return IntegerSolveFailure(pid, STAGE_INTEGER_SOLVER,
                                   lrs.CATEGORY_NONFINITE,
                                   "nonfinite model coefficient; not solved")
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([np.inf if b[1] is None else b[1] for b in bounds],
                     dtype=float)
    try:
        raw = milp(c, integrality=np.ones(c.size), bounds=Bounds(lower, upper),
                   constraints=LinearConstraint(rows, -np.inf, rhs),
                   options=dict(SOLVER_OPTIONS))
    except Exception as exc:  # reported, not retried
        return IntegerSolveFailure(pid, STAGE_INTEGER_SOLVER, "exception",
                                   f"solver raised {exc!r}")
    status = _maybe_int(getattr(raw, "status", None))
    message = str(getattr(raw, "message", None))
    fun = _maybe_float(getattr(raw, "fun", None))
    dual = _maybe_float(getattr(raw, "mip_dual_bound", None))
    gap = _maybe_float(getattr(raw, "mip_gap", None))
    nodes = _maybe_int(getattr(raw, "mip_node_count", None))
    x = getattr(raw, "x", None)
    vector = None
    if x is not None:
        arr = np.asarray(x, dtype=float)
        if arr.shape == c.shape and np.all(np.isfinite(arr)):
            vector = tuple(float(v) for v in arr)

    def fail(stage, category, reason):
        return IntegerSolveFailure(
            pid, stage, category, reason, status, message, vector,
            None if fun is None else -fun, None if dual is None else -dual,
            gap, nodes)

    if status != 0:
        category = {1: "solver_limit", 2: "integer_infeasible",
                    3: "unbounded"}.get(status, "solver_failure")
        return fail(STAGE_INTEGER_SOLVER, category,
                    f"solver status {status!r}: {message}")
    if vector is None or fun is None:
        return fail(STAGE_INTEGER_SOLVER, "malformed_output",
                    "optimal status without a finite vector and objective")
    policy = canonical.numerical_policy
    recomputed = math.fsum(float(ci) * v for ci, v in zip(c, vector))
    if not lrs._objective_agrees(fun, recomputed, policy):
        return fail(STAGE_INTEGER_SOLVER, "malformed_output",
                    f"reported objective {fun!r} disagrees with recomputed "
                    f"{recomputed!r}")
    deviation = max(abs(v - round(v)) for v in vector)
    if deviation > CONVERSION_TOL:
        return fail(STAGE_INTEGER_CONVERSION, "fractional_solution",
                    f"largest integrality deviation {deviation!r} exceeds "
                    f"{CONVERSION_TOL}; not rounded")

    n = len(canonical.requests)
    ints = [int(round(v)) for v in vector]
    x_i, y_i, ind_i, z_i = (ints[0:n], ints[n:2 * n], ints[2 * n:3 * n],
                            ints[3 * n:])
    if any(ind != (1 if xv > 0 else 0) for ind, xv in zip(ind_i, x_i)):
        return fail(STAGE_INTEGER_CONVERSION, "inconsistent_indicator",
                    f"indicators {ind_i} differ from 1{{x > 0}} for {x_i}")
    residuals = lrs._residuals(canonical, *[[float(v) for v in part] for part
                                            in (x_i, y_i, ind_i, z_i)])
    violated = sorted(k for k, v in residuals.items() if v > 0)
    if violated:
        return fail(STAGE_INTEGER_CONVERSION, "integer_infeasible_point",
                    f"converted point violates {violated}")
    objective = lrs._objective_max(canonical, x_i, y_i, z_i)
    allowance = (policy.objective_abs_tol
                 + policy.objective_rel_tol * max(1.0, abs(fun))
                 + CONVERSION_TOL * math.fsum(abs(float(ci)) for ci in c))
    if abs(objective - (-fun)) > allowance:
        return fail(STAGE_INTEGER_CONVERSION, "objective_mismatch",
                    f"integer objective {objective!r} vs reported {-fun!r}")

    reqs = canonical.requests
    tokens = sum(x_i) + sum(y_i)
    actions = sum(ind_i) + sum(y_i)
    residents = (canonical.resident_count - sum(z_i)
                 + sum(ind for r, ind in zip(reqs, ind_i) if not r.resident))
    free = (canonical.m_free
            + sum(r.preemption_recovery * z for r, z in zip(reqs, z_i))
            - sum(r.prefill_fixed_charge * ind for r, ind in zip(reqs, ind_i)))
    victim_order, execution_order = lrs._declared_orders(canonical, y_i, z_i,
                                                         x_i)
    plan = lrs.IntegerPlan(
        problem_id=canonical.problem_id,
        decisions=tuple(
            lrs.IntegerDecision(r.request_id, r.order_key, x_i[k], y_i[k],
                                z_i[k], ind_i[k])
            for k, r in enumerate(reqs)),
        victim_order=victim_order,
        execution_order=execution_order,
        token_use=tokens, action_count=actions, resident_count_after=residents,
        free_blocks_before_decodes=free,
        decode_demand=sum(r.decode_demand * y for r, y in zip(reqs, y_i)),
        numerical_policy=policy,
        objective=objective,
        extraction=lrs.DirectIntegerSolution(
            solver=SOLVER, scipy_version=scipy.__version__,
            options=SOLVER_OPTIONS, raw_status=status, message=message,
            objective=-fun, dual_bound=None if dual is None else -dual,
            reported_gap=gap, node_count=nodes,
            max_integrality_deviation=deviation,
            conversion_tol=CONVERSION_TOL),
    )
    validated = lrs.validate_integer_plan(canonical, plan)
    if isinstance(validated, lrs.Failure):
        return fail(validated.stage, validated.category, validated.reason)
    return IntegerOptimum(canonical.problem_id, validated, vector)


def plan_actions(plan):
    """``{request_id: "prefill:<n>" | "decode" | "preempt"}`` of a plan or
    of its JSON record."""
    decisions = (plan["decisions"] if isinstance(plan, dict)
                 else [dataclasses.asdict(d) for d in plan.decisions])
    out = {}
    for d in decisions:
        if d["prefill_tokens"]:
            out[d["request_id"]] = f"prefill:{d['prefill_tokens']}"
        elif d["decode"]:
            out[d["request_id"]] = "decode"
        elif d["preempt"]:
            out[d["request_id"]] = "preempt"
    return out


def problem_from_record(record):
    """The ``LPProblem`` of a retained JSON record of one (as written by the
    comparison drivers). Only the JSON container types are restored: tuple
    ``order_key``, frozenset legal victims, and the ``NumericalPolicy``.
    Callers validate the result and check that it serializes back to the
    same record."""
    requests = tuple(lrs.RequestInput(**dict(r, order_key=tuple(r["order_key"])))
                     for r in record["requests"])
    return lrs.LPProblem(**dict(
        record, requests=requests,
        legal_preemption_ids=frozenset(record["legal_preemption_ids"]),
        numerical_policy=lrs.NumericalPolicy(**record["numerical_policy"])))


def lp_summary(result_type, result):
    """Primitive values of an LP-method result (live, or its JSON record);
    ``None`` unless it is a ``SchedulingSuccess``."""
    if result_type != "SchedulingSuccess":
        return None
    if not isinstance(result, dict):
        result = dataclasses.asdict(result)
    return dict(relaxed_objective=result["relaxed"]["normalized_objective"],
                plan_objective=result["plan"]["objective"],
                actions=plan_actions(result["plan"]),
                fractional_request_count=(
                    result["plan"]["extraction"]["fractional_request_count"]))


def integer_summary(result_type, result):
    """Primitive record of one integer solve (live, or its JSON record):
    ``status`` "optimal" with the solver's provenance, or "failure" with the
    stage, category, status, and any incumbent, bound, and gap kept as
    diagnostics (an incumbent is never called optimal)."""
    if not isinstance(result, dict):
        result = dataclasses.asdict(result)
    if result_type == "IntegerOptimum":
        source = result["plan"]["extraction"]
        return dict(status="optimal", objective=result["plan"]["objective"],
                    actions=plan_actions(result["plan"]),
                    raw_status=source["raw_status"], message=source["message"],
                    dual_bound=source["dual_bound"],
                    reported_gap=source["reported_gap"],
                    node_count=source["node_count"],
                    max_integrality_deviation=(
                        source["max_integrality_deviation"]))
    return dict(status="failure", stage=result["stage"],
                category=result["category"], reason=result["reason"],
                raw_status=result["raw_status"], message=result["message"],
                incumbent_objective=result["incumbent_objective"],
                incumbent_vector=result["incumbent_vector"],
                dual_bound=result["dual_bound"],
                reported_gap=result["reported_gap"],
                node_count=result["node_count"])


def evaluate_assignment(problem, assignment):
    """Utility and LP-contract feasibility of an integer assignment
    ``{request_id: "prefill:<n>" | "decode" | "preempt"}`` (absent means no
    action) on canonical ``problem``. Diagnostic only; returns every domain
    and row violation, not just the first."""
    reqs = problem.requests
    known = {r.request_id for r in reqs}
    domain = [f"unknown request {rid}" for rid in sorted(set(assignment)
                                                         - known)]
    x, y, ind, z = ([0] * len(reqs) for _ in range(4))
    for k, r in enumerate(reqs):
        action = assignment.get(r.request_id)
        if action is None:
            continue
        if action.startswith("prefill:"):
            x[k], ind[k] = int(action.split(":")[1]), 1
            if not (r.prefill_eligible and 1 <= x[k] <= r.prefill_upper_bound):
                domain.append(f"{r.request_id}: prefill {x[k]} outside "
                              f"[1, {r.prefill_upper_bound}]")
        elif action == "decode":
            y[k] = 1
            if not r.decode_eligible:
                domain.append(f"{r.request_id}: decode not eligible")
        elif action == "preempt":
            z[k] = 1
            if r.request_id not in problem.legal_preemption_ids:
                domain.append(f"{r.request_id}: preemption not legal")
        else:
            domain.append(f"{r.request_id}: unknown action {action!r}")
    residuals = lrs._residuals(problem, *[[float(v) for v in part]
                                          for part in (x, y, ind, z)])
    violated = sorted(k for k, v in residuals.items() if v > 0)
    return dict(feasible=not domain and not violated,
                domain_violations=domain, violated_rows=violated,
                objective=lrs._objective_max(problem, x, y, z))


def compare_same_state(problem, lp_result, integer_result):
    """Compare the LP method and the integer optimum on one identical
    problem. Gaps are attributed only when both succeeded:
    ``relaxation_gap`` = LP optimum - integer optimum and
    ``extraction_loss`` = integer optimum - returned LP plan utility."""
    lp_type, integer_type = (type(lp_result).__name__,
                             type(integer_result).__name__)
    return compare_summaries(problem.problem_id, lp_type,
                             lp_summary(lp_type, lp_result), integer_type,
                             integer_summary(integer_type, integer_result))


def compare_summaries(problem_id, lp_status, lp, integer_status, integer):
    """``compare_same_state`` on primitive summaries (``lp_summary`` and
    ``integer_summary``) of results for the identical problem."""
    record = dict(problem_id=problem_id, comparison_tol=COMPARISON_TOL,
                  lp_status=lp_status, integer_status=integer_status,
                  integer=integer)
    if integer["status"] == "optimal":
        record.update(integer_objective=integer["objective"],
                      integer_dual_bound=integer["dual_bound"],
                      integer_reported_gap=integer["reported_gap"],
                      integer_node_count=integer["node_count"],
                      integer_actions=integer["actions"])
    else:
        record.update(integer_failure=dict(
            stage=integer["stage"], category=integer["category"],
            reason=integer["reason"],
            incumbent_objective_diagnostic=integer["incumbent_objective"],
            dual_bound=integer["dual_bound"]))
    if lp is not None:
        record.update(lp_relaxed_objective=lp["relaxed_objective"],
                      lp_plan_objective=lp["plan_objective"],
                      lp_actions=lp["actions"],
                      fractional_request_count=lp["fractional_request_count"])
    if lp is None or integer["status"] != "optimal":
        record["attribution"] = "not_comparable"
        return record
    lp_opt, ilp, lp_plan = (record["lp_relaxed_objective"],
                            record["integer_objective"],
                            record["lp_plan_objective"])

    def tol(*values):
        return COMPARISON_TOL * max(1.0, *(abs(v) for v in values))
    gap, loss = lp_opt - ilp, ilp - lp_plan
    record.update(relaxation_gap=gap, extraction_loss=loss,
                  same_actions=record["lp_actions"] == record["integer_actions"],
                  bounds_consistent=gap >= -tol(lp_opt, ilp)
                  and loss >= -tol(ilp, lp_plan))
    if record["same_actions"]:
        record["attribution"] = "identical_actions"
    elif abs(loss) <= tol(ilp, lp_plan):
        record["attribution"] = "tied_optimum_different_actions"
    else:
        record["attribution"] = "extraction_loss"
    record["relaxation_gap_present"] = gap > tol(lp_opt, ilp)
    return record


def integer_decision(scheduler):
    """One native decision driven by an integer optimum, in the live LP
    coordinator's order: ordinary idle check, one mapping with the
    scheduler's coefficient policy and configured caps, one integer solve,
    then the unchanged executor. A failed mapping, solve, or precommit check
    raises ``LPSchedulingError`` before mutation; nothing falls back."""
    import lpserve_plan_execution as lpe
    import lpserve_state_mapping as lsm
    from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
    from sarathi.core.scheduler.lp_scheduler import LPSchedulingError

    now = time.monotonic()
    if scheduler._is_idle(now):
        return SchedulerOutputs(scheduler._iteration_id, ignored_seq_ids=[],
                                preempted_seq_ids=[],
                                scheduled_seq_metadata_list=[])
    config = scheduler.scheduler_config
    snapshot = lsm.map_scheduler_state(
        scheduler, snapshot_time=now, b_max=config.b_max, c_max=config.c_max,
        s_max=config.s_max, coefficient_policy=scheduler._coefficient_policy(),
        rank_profile=lsm.vllm_oriented_ranks,
        numerical_policy=config.numerical_policy)
    if isinstance(snapshot, lsm.MappingFailure):
        raise LPSchedulingError(snapshot)
    # Module-global lookup, so a validation observer may wrap it.
    result = solve_integer_plan(snapshot.lp_problem)
    if isinstance(result, IntegerSolveFailure):
        raise LPSchedulingError(result)
    outputs = lpe.execute_plan(
        scheduler, snapshot,
        lrs.SchedulingSuccess(result.problem_id, result.plan, None, None))
    if isinstance(outputs, lrs.Failure):
        raise LPSchedulingError(outputs)
    return outputs
