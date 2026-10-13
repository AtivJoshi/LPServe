"""Standalone, CPU-only LP-relaxation scheduling layer.

Builds the one-step myopic scheduling LP (docs/lp_scheduler_design.md section 7)
from immutable inputs, solves it once with SciPy/HiGHS, validates the relaxed
solution, constructs whole integer actions guided by that solution (design
section 10), and validates the complete integer plan exactly (design section
11.3).

The module reads no clocks, global state, or randomness, and imports nothing
from the serving framework. Every failure is returned as a structured
``Failure`` that carries no plan. ``Failure.outcome`` distinguishes
solver-reported LP infeasibility, exhaustive absence of legal useful integer
work, solver or numerical failure, and implementation or validation failure.

Variable layout for ``n`` requests sorted by ascending ``order_key``: the LP
vector is ``[x_0..x_{n-1}, y_0.., I_0.., z_0..]`` where ``I`` is the prefill
indicator. Throughout, ``H`` is the free-block count after selected
preemptions and prefills and before any decode:
``H = m_free + sum(recovery * z) - sum(prefill_fixed_charge * I)``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy
from scipy.optimize import linprog

STAGE_INPUT = "input_validation"
STAGE_SOLVER = "solver"
STAGE_RELAXED = "relaxed_validation"
STAGE_EXTRACTION = "extraction"
STAGE_PLAN = "plan_validation"

OUTCOME_LP_INFEASIBLE = "solver_reported_lp_infeasibility"
OUTCOME_NO_USEFUL_WORK = "no_legal_useful_work"
OUTCOME_NUMERICAL = "solver_or_numerical_failure"
OUTCOME_INVALID = "implementation_or_validation_failure"

CATEGORY_OPTIMAL = "optimal_candidate"
CATEGORY_LP_INFEASIBLE = "solver_reported_infeasibility"
CATEGORY_NO_USEFUL_WORK = "no_legal_useful_work"
CATEGORY_NONFINITE = "nonfinite_arithmetic"
_STATUS_CATEGORIES = {
    1: "solver_limit",
    2: CATEGORY_LP_INFEASIBLE,
    3: "unbounded",
    4: "numerical_difficulty",
}

SOLVER_METHOD = "highs-ds"
SOLVER_OPTIONS = (("presolve", True),)

Rank = Union[int, float]


# ---------------------------------------------------------------------------
# Immutable records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalPolicy:
    """Provisional numerical policy; every value is externally visible.

    ``integrality_tol`` is retained as a configured value only; it has no
    classification or decision role in extraction.
    """

    policy_id: str = "lp_relaxation_mvp_v1"
    feasibility_tol: float = 1e-7
    integrality_tol: float = 1e-6
    objective_abs_tol: float = 1e-9
    objective_rel_tol: float = 1e-9


@dataclass(frozen=True)
class RequestInput:
    """One request of the planning universe.

    ``resident`` selects membership of the resident set (allocated cache) or
    the waiting set (no allocation). ``decode_demand`` is the exact zero-or-one
    block allocated by one decode. ``service_rank`` is required for prefill- or
    decode-eligible requests and ``victim_rank`` for preemption-eligible ones;
    smaller ranks come first and ``order_key`` breaks remaining ties.
    """

    request_id: str
    order_key: tuple
    resident: bool
    prompt_tokens_remaining: int
    prefill_upper_bound: int
    prefill_eligible: bool
    decode_eligible: bool
    preemption_eligible: bool
    prefill_fixed_charge: int
    decode_demand: int
    preemption_recovery: int
    decode_utility: float
    prefill_token_utility: float
    preemption_penalty: float
    service_rank: Optional[Rank]
    victim_rank: Optional[Rank]


@dataclass(frozen=True)
class LPProblem:
    requests: tuple
    legal_preemption_ids: frozenset
    b_max: int
    c_max: int
    s_max: int
    resident_limit: int
    resident_count: int
    m_free: int
    admission_watermark: int
    problem_id: str
    numerical_policy: NumericalPolicy


@dataclass(frozen=True)
class SolverDiagnostics:
    solver: str
    scipy_version: str
    method: str
    options: tuple
    raw_status: Optional[int]
    raw_success: Optional[bool]
    message: Optional[str]
    iterations: Optional[int]
    vector_present: bool
    vector_length: Optional[int]
    expected_vector_length: int


@dataclass(frozen=True)
class SolverResult:
    """Project-owned normalization of one solver attempt."""

    category: str
    reason: str
    diagnostics: SolverDiagnostics
    raw_vector: Optional[tuple]
    reported_objective_min: Optional[float]
    recomputed_objective_min: Optional[float]


@dataclass(frozen=True)
class RelaxedDecision:
    request_id: str
    x: float
    y: float
    prefill_indicator: float
    z: float


@dataclass(frozen=True)
class RelaxedSolution:
    decisions: tuple
    raw_vector: tuple
    projection_count: int
    worst_raw_violation: float
    worst_normalized_violation: float
    normalized_objective: float


@dataclass(frozen=True)
class ExtractionDiagnostics:
    """Observations of one whole-action construction.

    ``candidate_order`` and ``victim_candidate_order`` are the scan orders.
    For the no-useful-work outcome, ``candidate_order`` is the exhausted
    candidate set. ``fractional_request_count`` counts requests with some
    relaxed y, I, or z strictly between 0 and 1; it is descriptive only.
    """

    candidate_order: tuple
    victim_candidate_order: tuple
    accepted_order: tuple
    trial_count: int
    failed_trial_count: int
    declined_optional_trial_count: int
    skipped_candidate_count: int
    fractional_request_count: int


@dataclass(frozen=True)
class IntegerDecision:
    request_id: str
    order_key: tuple
    prefill_tokens: int
    decode: int
    preempt: int
    prefill_indicator: int


@dataclass(frozen=True)
class IntegerPlan:
    """A complete integer plan.

    ``victim_order`` lists preempted requests by (victim rank, order_key).
    ``execution_order`` lists prefills, then zero-demand decodes, then
    one-block-demand decodes, each by (service rank, order_key). The totals
    are final local totals: ``free_blocks_before_decodes`` is H.
    """

    problem_id: str
    decisions: tuple
    victim_order: tuple
    execution_order: tuple
    token_use: int
    action_count: int
    resident_count_after: int
    free_blocks_before_decodes: int
    decode_demand: int
    numerical_policy: NumericalPolicy
    objective: float
    extraction: ExtractionDiagnostics


@dataclass(frozen=True)
class Failure:
    problem_id: Optional[str]
    stage: str
    outcome: str
    category: str
    reason: str
    solver: Optional[SolverResult] = None
    extraction: Optional[ExtractionDiagnostics] = None


@dataclass(frozen=True)
class SchedulingSuccess:
    problem_id: str
    plan: IntegerPlan
    relaxed: RelaxedSolution
    solver: SolverResult


def _fail(problem, stage, outcome, category, reason, solver=None,
          extraction=None):
    problem_id = getattr(problem, "problem_id", None)
    if not isinstance(problem_id, str):
        problem_id = None
    return Failure(
        problem_id, stage, outcome, category, reason, solver, extraction
    )


class _InputError(Exception):
    def __init__(self, category, reason):
        super().__init__(reason)
        self.category = category
        self.reason = reason


class _NumericalError(Exception):
    pass


# ---------------------------------------------------------------------------
# Checked arithmetic
# ---------------------------------------------------------------------------


def _checked_sum(values, what):
    """Stable sum that raises ``_NumericalError`` unless the result is finite.

    ``math.fsum`` raises on intermediate overflow and on ``inf + -inf``; both
    are reported as numerical failures, never as unexplained exceptions.
    """
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise _NumericalError(f"{what}: nonfinite summation ({exc})") from None
    if not math.isfinite(total):
        raise _NumericalError(f"{what}: nonfinite sum {total!r}")
    return total


def _checked_product(a, b, what):
    value = a * b
    if not math.isfinite(value):
        raise _NumericalError(f"{what}: nonfinite product {a!r} * {b!r}")
    return value


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_count(value, minimum):
    """Integer at least ``minimum`` that converts to a finite float."""
    return _is_int(value) and value >= minimum and _is_real(value)


def _require(condition, category, reason):
    if not condition:
        raise _InputError(category, reason)


def _check_policy(policy):
    _require(
        isinstance(policy, NumericalPolicy), "malformed_value",
        "numerical_policy must be a NumericalPolicy",
    )
    _require(
        isinstance(policy.policy_id, str) and policy.policy_id,
        "malformed_value", "numerical policy_id must be a non-empty string",
    )
    for name in (
        "feasibility_tol", "integrality_tol",
        "objective_abs_tol", "objective_rel_tol",
    ):
        value = getattr(policy, name)
        _require(
            _is_real(value) and value > 0, "malformed_value",
            f"numerical policy {name} must be a finite positive number",
        )


def _check_request(r, c_max, legal_ids):
    _require(
        isinstance(r.request_id, str) and r.request_id,
        "malformed_value", "request_id must be a non-empty string",
    )
    rid = r.request_id
    _require(
        isinstance(r.order_key, tuple) and r.order_key
        and all(_is_int(k) for k in r.order_key),
        "malformed_value",
        f"request {rid!r}: order_key must be a non-empty tuple of integers",
    )
    for name in (
        "prompt_tokens_remaining", "prefill_upper_bound",
        "prefill_fixed_charge", "decode_demand", "preemption_recovery",
    ):
        _require(
            _is_count(getattr(r, name), 0), "malformed_value",
            f"request {rid!r}: {name} must be a non-negative integer "
            "representable as a finite float",
        )
    for name in (
        "decode_utility", "prefill_token_utility", "preemption_penalty",
    ):
        value = getattr(r, name)
        _require(
            _is_real(value) and value >= 0, "malformed_value",
            f"request {rid!r}: {name} must be a finite non-negative number",
        )
    for name in (
        "resident", "prefill_eligible", "decode_eligible",
        "preemption_eligible",
    ):
        _require(
            isinstance(getattr(r, name), bool), "malformed_value",
            f"request {rid!r}: {name} must be a bool",
        )
    _require(
        r.decode_demand in (0, 1), "inconsistent_resources",
        f"request {rid!r}: decode_demand must be exactly 0 or 1",
    )
    if r.prefill_eligible:
        _require(
            r.prompt_tokens_remaining > 0
            and r.prefill_upper_bound
            == min(r.prompt_tokens_remaining, c_max),
            "inconsistent_bounds",
            f"request {rid!r}: prefill-eligible requires "
            "prefill_upper_bound == min(prompt_tokens_remaining, C_max) > 0",
        )
    else:
        _require(
            r.prefill_upper_bound == 0, "inconsistent_bounds",
            f"request {rid!r}: prefill-ineligible requires "
            "prefill_upper_bound == 0",
        )
    if r.decode_eligible:
        _require(
            r.prompt_tokens_remaining == 0, "inconsistent_eligibility",
            f"request {rid!r}: decode-eligible requires "
            "prompt_tokens_remaining == 0",
        )
    if r.resident:
        _require(
            r.prefill_fixed_charge == 0, "inconsistent_resources",
            f"request {rid!r}: a resident request has prefill_fixed_charge 0",
        )
    else:
        _require(
            not r.decode_eligible and not r.preemption_eligible,
            "inconsistent_membership",
            f"request {rid!r}: a waiting request can only prefill",
        )
    _require(
        r.preemption_eligible == (rid in legal_ids),
        "inconsistent_preemption_membership",
        f"request {rid!r}: preemption_eligible disagrees with "
        "legal_preemption_ids",
    )
    if not r.preemption_eligible:
        _require(
            r.preemption_recovery == 0, "inconsistent_preemption_membership",
            f"request {rid!r}: preemption_recovery must be 0 outside the "
            "legal-preemption set",
        )
    for name, required in (
        ("service_rank", r.prefill_eligible or r.decode_eligible),
        ("victim_rank", r.preemption_eligible),
    ):
        value = getattr(r, name)
        if required:
            _require(
                value is not None, "missing_rank",
                f"request {rid!r}: {name} is required",
            )
        _require(
            value is None or _is_real(value), "malformed_value",
            f"request {rid!r}: {name} must be a finite number",
        )


def _validate_problem_or_raise(problem):
    _require(
        isinstance(problem, LPProblem), "malformed_value",
        "problem must be an LPProblem",
    )
    _require(
        isinstance(problem.problem_id, str) and problem.problem_id,
        "malformed_value", "problem_id must be a non-empty string",
    )
    _check_policy(problem.numerical_policy)
    for name in ("b_max", "c_max", "s_max", "resident_limit"):
        _require(
            _is_count(getattr(problem, name), 1), "malformed_value",
            f"{name} must be a positive integer representable as a "
            "finite float",
        )
    for name in ("m_free", "admission_watermark", "resident_count"):
        _require(
            _is_count(getattr(problem, name), 0), "malformed_value",
            f"{name} must be a non-negative integer representable as a "
            "finite float",
        )
    _require(
        isinstance(problem.requests, tuple) and problem.requests
        and all(isinstance(r, RequestInput) for r in problem.requests),
        "malformed_value",
        "requests must be a non-empty tuple of RequestInput",
    )
    legal = problem.legal_preemption_ids
    _require(
        isinstance(legal, frozenset)
        and all(isinstance(i, str) for i in legal),
        "malformed_value",
        "legal_preemption_ids must be a frozenset of request IDs",
    )
    ids = [r.request_id for r in problem.requests]
    _require(
        all(isinstance(i, str) for i in ids), "malformed_value",
        "request_id must be a string",
    )
    _require(len(set(ids)) == len(ids), "duplicate_id", "duplicate request_id")
    _require(
        legal <= set(ids), "inconsistent_preemption_membership",
        "legal_preemption_ids contains an unknown request ID",
    )
    for r in problem.requests:
        _check_request(r, problem.c_max, legal)
    keys = [r.order_key for r in problem.requests]
    _require(
        len(set(keys)) == len(keys), "duplicate_order_key",
        "duplicate order_key",
    )
    residents = sum(1 for r in problem.requests if r.resident)
    _require(
        problem.resident_count == residents, "inconsistent_membership",
        f"resident_count {problem.resident_count} differs from the "
        f"{residents} resident requests",
    )
    _require(
        problem.resident_count <= problem.resident_limit,
        "inconsistent_membership",
        f"resident_count {problem.resident_count} exceeds resident_limit "
        f"{problem.resident_limit}",
    )
    ordered = tuple(sorted(problem.requests, key=lambda r: r.order_key))
    return dataclasses.replace(problem, requests=ordered)


def validate_problem(problem):
    """Return the canonical problem (requests sorted ascending by order_key)
    or a ``Failure``. Later stages require this canonical form."""
    try:
        return _validate_problem_or_raise(problem)
    except _InputError as err:
        return _fail(problem, STAGE_INPUT, OUTCOME_INVALID, err.category,
                     err.reason)


# ---------------------------------------------------------------------------
# LP construction and solver adapter
# ---------------------------------------------------------------------------


def _watermark_rows_needed(problem):
    """Waiting requests that receive a watermark row (none when h == 0)."""
    if problem.admission_watermark == 0:
        return []
    return [i for i, r in enumerate(problem.requests) if not r.resident]


def _build_lp(problem):
    """Return (c, A_ub, b_ub, bounds, row_labels) in SciPy minimization form.

    Every row is ``A_ub[k] @ v <= b_ub[k]``; ``row_labels[k]`` names it with
    the same labels as ``_residuals``.
    """
    reqs = problem.requests
    n = len(reqs)
    s_max = float(problem.s_max)
    h = float(problem.admission_watermark)
    c = np.zeros(4 * n)
    bounds = []
    for i, r in enumerate(reqs):
        c[i] = -float(r.prefill_token_utility)
        c[n + i] = -float(r.decode_utility)
        if r.preemption_eligible:
            c[3 * n + i] = float(r.preemption_penalty)
    bounds += [(0.0, None)] * n
    bounds += [(0.0, 1.0 if r.decode_eligible else 0.0) for r in reqs]
    bounds += [(0.0, 1.0 if r.prefill_eligible else 0.0) for r in reqs]
    bounds += [(0.0, 1.0 if r.preemption_eligible else 0.0) for r in reqs]

    # Coefficients of -H on (I, z); -H <= -m_free is the generic H row.
    neg_h = np.zeros(4 * n)
    for i, r in enumerate(reqs):
        neg_h[2 * n + i] = float(r.prefill_fixed_charge)
        if r.resident:
            neg_h[3 * n + i] = -float(r.preemption_recovery)
    m_free = float(problem.m_free)

    rows, rhs, labels = [], [], []

    def add(label, row, bound):
        rows.append(row)
        rhs.append(bound)
        labels.append(label)

    row = np.zeros(4 * n)
    row[0:n] = 1.0
    row[n:2 * n] = 1.0
    add("token_volume", row, float(problem.b_max))

    row = np.zeros(4 * n)
    row[n:2 * n] = 1.0
    row[2 * n:3 * n] = 1.0
    add("action_width", row, s_max)

    # H - sum(d y) >= 0  <=>  -H + sum(d y) <= 0
    row = neg_h.copy()
    for i, r in enumerate(reqs):
        row[n + i] = float(r.decode_demand)
    add("actual_memory", row, m_free)

    # sum(y) <= S_max * H  <=>  sum(y) - S_max * H <= 0. Overflow yields a
    # nonfinite entry, which solve_relaxation reports before solving.
    with np.errstate(over="ignore"):
        row = s_max * neg_h
    row[n:2 * n] += 1.0
    add("decode_prerequisite", row, s_max * m_free)

    # h * I_i <= H
    for i in _watermark_rows_needed(problem):
        row = neg_h.copy()
        row[2 * n + i] += h
        add(f"admission_watermark[{reqs[i].request_id}]", row, m_free)

    # |R| - sum_R z + sum_Q I <= N_max
    row = np.zeros(4 * n)
    for i, r in enumerate(reqs):
        if r.resident:
            row[3 * n + i] = -1.0
        else:
            row[2 * n + i] = 1.0
    add("resident_capacity", row,
        float(problem.resident_limit) - float(problem.resident_count))

    # sum(I + y) >= 1
    row = np.zeros(4 * n)
    row[n:3 * n] = -1.0
    add("useful_execution", row, -1.0)

    for i, r in enumerate(reqs):
        rid = r.request_id
        row = np.zeros(4 * n)  # I - x <= 0
        row[2 * n + i] = 1.0
        row[i] = -1.0
        add(f"prefill_linkage_lower[{rid}]", row, 0.0)
        row = np.zeros(4 * n)  # x - U * I <= 0
        row[i] = 1.0
        row[2 * n + i] = -float(r.prefill_upper_bound)
        add(f"prefill_linkage_upper[{rid}]", row, 0.0)
        row = np.zeros(4 * n)  # I + y + z <= 1
        row[n + i] = 1.0
        row[2 * n + i] = 1.0
        row[3 * n + i] = 1.0
        add(f"mutual_exclusion[{rid}]", row, 1.0)
    return c, np.array(rows), np.array(rhs), bounds, labels


def _objective_agrees(reported, recomputed, policy):
    allowance = policy.objective_abs_tol + policy.objective_rel_tol * max(
        1.0, abs(reported), abs(recomputed)
    )
    return abs(reported - recomputed) <= allowance


def _diagnostics(expected_length, status=None, success=None, message=None,
                 iterations=None, x=None):
    length = None
    if x is not None:
        try:
            length = int(np.asarray(x).size)
        except Exception:
            length = None
    return SolverDiagnostics(
        solver="scipy.optimize.linprog",
        scipy_version=scipy.__version__,
        method=SOLVER_METHOD,
        options=SOLVER_OPTIONS,
        raw_status=int(status) if _is_int(status)
        or isinstance(status, np.integer) else None,
        raw_success=success if isinstance(success, bool) else None,
        message=None if message is None else str(message),
        iterations=iterations if _is_int(iterations) else None,
        vector_present=x is not None,
        vector_length=length,
        expected_vector_length=expected_length,
    )


def _usable_vector(x, expected_length):
    if x is None:
        return None
    try:
        vector = np.asarray(x, dtype=float)
    except Exception:
        return None
    if vector.shape != (expected_length,) or not np.all(np.isfinite(vector)):
        return None
    return tuple(float(v) for v in vector)


def _normalize_scipy_result(raw, c, policy):
    expected = int(c.size)
    status = getattr(raw, "status", None)
    success = getattr(raw, "success", None)
    x = getattr(raw, "x", None)
    fun = getattr(raw, "fun", None)
    diag = _diagnostics(
        expected, status, success, getattr(raw, "message", None),
        getattr(raw, "nit", None), x,
    )
    vector = _usable_vector(x, expected)

    def result(category, reason, reported=None, recomputed=None):
        return SolverResult(
            category, reason, diag, vector, reported, recomputed
        )

    if not _is_int(status) and not isinstance(status, np.integer):
        return result("malformed_output", f"unknown solver status {status!r}")
    status = int(status)
    if status in _STATUS_CATEGORIES:
        return result(
            _STATUS_CATEGORIES[status], f"solver reported status {status}"
        )
    if status != 0:
        return result("malformed_output", f"unknown solver status {status}")
    if success is not True:
        return result("malformed_output", "status 0 without success is True")
    if vector is None:
        return result(
            "malformed_output", "status 0 without a usable finite primal vector"
        )
    if not _is_real(fun):
        return result(
            "malformed_output", "status 0 without a finite reported objective"
        )
    reported = float(fun)
    try:
        recomputed = _checked_sum(
            (float(ci) * v for ci, v in zip(c, vector)), "solver objective"
        )
    except _NumericalError as err:
        return result(CATEGORY_NONFINITE, str(err), reported)
    if not _objective_agrees(reported, recomputed, policy):
        return result(
            "malformed_output",
            f"reported objective {reported!r} disagrees with recomputed "
            f"{recomputed!r}",
            reported, recomputed,
        )
    return result(CATEGORY_OPTIMAL, "optimal candidate", reported, recomputed)


def solve_relaxation(problem):
    """Solve the LP once for a canonical problem; never raises for solver or
    numerical outcomes."""
    c, rows, rhs, bounds, _ = _build_lp(problem)
    expected = int(c.size)
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(rows))
            and np.all(np.isfinite(rhs))):
        return SolverResult(
            CATEGORY_NONFINITE,
            "LP construction produced a nonfinite coefficient; the solver "
            "was not called",
            _diagnostics(expected), None, None, None,
        )
    try:
        raw = linprog(
            c, A_ub=rows, b_ub=rhs, bounds=bounds,
            method=SOLVER_METHOD, options=dict(SOLVER_OPTIONS),
        )
    except KeyboardInterrupt as exc:
        return SolverResult(
            "interruption", f"solver interrupted: {exc!r}",
            _diagnostics(expected), None, None, None,
        )
    except Exception as exc:
        return SolverResult(
            "exception", f"solver raised {exc!r}",
            _diagnostics(expected), None, None, None,
        )
    return _normalize_scipy_result(raw, c, problem.numerical_policy)


def _solver_outcome(category):
    if category == CATEGORY_LP_INFEASIBLE:
        return OUTCOME_LP_INFEASIBLE
    return OUTCOME_NUMERICAL


# ---------------------------------------------------------------------------
# Relaxed-solution validation
# ---------------------------------------------------------------------------


def _split(vector, n):
    return (
        list(vector[0:n]), list(vector[n:2 * n]),
        list(vector[2 * n:3 * n]), list(vector[3 * n:4 * n]),
    )


def _residuals(problem, x, y, ind, z):
    """Recompute every constraint residual (lhs - rhs, feasible when <= 0)
    directly from project data, independently of the LP matrix.

    Raises ``_NumericalError`` when the arithmetic is not finite.
    """
    reqs = problem.requests
    rng = range(len(reqs))

    def prod(a, b):
        return _checked_product(float(a), b, "relaxed residual")

    h_value = _checked_sum(
        [float(problem.m_free)]
        + [prod(r.preemption_recovery, z[i])
           for i, r in enumerate(reqs) if r.resident]
        + [-prod(r.prefill_fixed_charge, ind[i]) for i, r in enumerate(reqs)],
        "H",
    )
    decodes = _checked_sum(y, "decode count")
    res = {
        "token_volume": _checked_sum(
            list(x) + list(y) + [-float(problem.b_max)], "token volume"),
        "action_width": _checked_sum(
            list(ind) + list(y) + [-float(problem.s_max)], "action width"),
        "actual_memory": _checked_sum(
            [prod(r.decode_demand, y[i]) for i, r in enumerate(reqs)]
            + [-h_value], "actual memory"),
        "decode_prerequisite": _checked_sum(
            [decodes, -prod(problem.s_max, h_value)], "decode prerequisite"),
    }
    for i in _watermark_rows_needed(problem):
        res[f"admission_watermark[{reqs[i].request_id}]"] = _checked_sum(
            [prod(problem.admission_watermark, ind[i]), -h_value],
            "admission watermark",
        )
    res["resident_capacity"] = _checked_sum(
        [float(problem.resident_count), -float(problem.resident_limit)]
        + [-z[i] if r.resident else ind[i] for i, r in enumerate(reqs)],
        "resident capacity",
    )
    res["useful_execution"] = _checked_sum(
        [1.0] + [-v for v in ind] + [-v for v in y], "useful execution")
    for i in rng:
        r = reqs[i]
        rid = r.request_id
        res[f"prefill_linkage_lower[{rid}]"] = _checked_sum(
            [ind[i], -x[i]], "prefill linkage")
        res[f"prefill_linkage_upper[{rid}]"] = _checked_sum(
            [x[i], -prod(r.prefill_upper_bound, ind[i])], "prefill linkage")
        res[f"mutual_exclusion[{rid}]"] = _checked_sum(
            [ind[i], y[i], z[i], -1.0], "mutual exclusion")
    return res


def _project(value, upper, eps):
    """Single-variable projection within tolerance.

    Returns ``(projected, changed)`` or ``None`` if outside tolerance.
    """
    if upper == 0.0:
        if abs(value) <= eps:
            return 0.0, value != 0.0
        return None
    if value < 0.0:
        return (0.0, True) if value >= -eps else None
    if upper is not None and value > upper:
        return (upper, True) if value <= upper + eps else None
    return value, False


def _objective_max(problem, x, y, z):
    """Maximization objective with checked stable summation."""
    terms = []
    for i, r in enumerate(problem.requests):
        terms.append(_checked_product(
            float(r.decode_utility), y[i], "objective"))
        terms.append(_checked_product(
            float(r.prefill_token_utility), x[i], "objective"))
        if r.preemption_eligible:
            terms.append(-_checked_product(
                float(r.preemption_penalty), z[i], "objective"))
    return _checked_sum(terms, "objective")


def validate_relaxed_solution(problem, raw_vector):
    """Independently validate a raw primal vector for a canonical problem.

    Returns a ``RelaxedSolution`` (normalized vector plus preserved raw
    vector) or a ``Failure``.
    """
    reqs = problem.requests
    n = len(reqs)
    eps = problem.numerical_policy.feasibility_tol

    def bad(category, reason):
        return _fail(problem, STAGE_RELAXED, OUTCOME_NUMERICAL, category,
                     reason)

    try:
        raw = tuple(float(v) for v in raw_vector)
    except (TypeError, ValueError):
        return bad("malformed_vector", "raw vector is not a sequence of numbers")
    if len(raw) != 4 * n or not all(math.isfinite(v) for v in raw):
        return bad(
            "malformed_vector",
            f"raw vector must be finite with length {4 * n}, got {len(raw)}",
        )

    uppers = (
        [None] * n
        + [1.0 if r.decode_eligible else 0.0 for r in reqs]
        + [1.0 if r.prefill_eligible else 0.0 for r in reqs]
        + [1.0 if r.preemption_eligible else 0.0 for r in reqs]
    )
    normalized = []
    projection_count = 0
    for k, (value, upper) in enumerate(zip(raw, uppers)):
        projected = _project(value, upper, eps)
        if projected is None:
            block = ("x", "y", "prefill_indicator", "z")[k // n]
            return bad(
                "bound_violation",
                f"{block} of request {reqs[k % n].request_id!r} = {value!r} "
                f"is outside its bounds beyond tolerance {eps}",
            )
        normalized.append(projected[0])
        projection_count += projected[1]

    try:
        raw_res = _residuals(problem, *_split(raw, n))
        norm_res = _residuals(problem, *_split(normalized, n))
        x, y, ind, z = _split(normalized, n)
        objective = _objective_max(problem, x, y, z)
    except _NumericalError as err:
        return bad(CATEGORY_NONFINITE, str(err))
    for label, res in (("raw", raw_res), ("normalized", norm_res)):
        for name, value in res.items():
            if not value <= eps:
                return bad(
                    "constraint_violation",
                    f"{label} {name} residual {value!r} exceeds tolerance "
                    f"{eps}",
                )
    decisions = tuple(
        RelaxedDecision(r.request_id, x[i], y[i], ind[i], z[i])
        for i, r in enumerate(reqs)
    )
    return RelaxedSolution(
        decisions=decisions,
        raw_vector=raw,
        projection_count=projection_count,
        worst_raw_violation=max(0.0, *raw_res.values()),
        worst_normalized_violation=max(0.0, *norm_res.values()),
        normalized_objective=objective,
    )


# ---------------------------------------------------------------------------
# Whole-action construction
# ---------------------------------------------------------------------------


@dataclass
class _Totals:
    """Local exact integer totals of a partial plan."""

    tokens: int
    actions: int
    residents: int
    free_before_decodes: int
    decode_demand: int
    admissions: int
    decodes: int

    def legal(self, problem):
        h = self.free_before_decodes
        return (
            self.tokens <= problem.b_max
            and self.actions <= problem.s_max
            and self.residents <= problem.resident_limit
            and h >= self.decode_demand
            and (self.admissions == 0 or h >= problem.admission_watermark)
            and (self.decodes == 0 or h >= 1)
        )


def _execution_key(r):
    return (r.service_rank, r.order_key)


def _victim_key(r):
    return (r.victim_rank, r.order_key)


def _declared_orders(problem, decode, preempt, prefill_tokens):
    reqs = problem.requests
    n = len(reqs)
    victims = sorted(
        (i for i in range(n) if preempt[i]), key=lambda i: _victim_key(reqs[i])
    )

    def group(i):
        if prefill_tokens[i] > 0:
            return 0
        return 1 + reqs[i].decode_demand

    executions = sorted(
        (i for i in range(n) if prefill_tokens[i] > 0 or decode[i]),
        key=lambda i: (group(i),) + _execution_key(reqs[i]),
    )
    return (
        tuple(reqs[i].request_id for i in victims),
        tuple(reqs[i].request_id for i in executions),
    )


def extract_integer_plan(problem, relaxed):
    """Construct whole actions guided by a validated relaxed solution.

    Returns an unvalidated ``IntegerPlan`` or a ``Failure``. Callers must pass
    a plan through ``validate_integer_plan``.
    """
    reqs = problem.requests
    n = len(reqs)
    policy = problem.numerical_policy
    if not isinstance(relaxed, RelaxedSolution) or len(
        relaxed.decisions
    ) != n or any(
        d.request_id != r.request_id for d, r in zip(relaxed.decisions, reqs)
    ):
        return _fail(
            problem, STAGE_EXTRACTION, OUTCOME_INVALID, "association_mismatch",
            "relaxed decisions do not match the ordered problem requests",
        )
    lp = relaxed.decisions
    if not all(
        _is_real(v) and v >= 0
        for d in lp for v in (d.x, d.y, d.prefill_indicator, d.z)
    ):
        return _fail(
            problem, STAGE_EXTRACTION, OUTCOME_INVALID, "unvalidated_relaxed",
            "relaxed values must be finite and non-negative",
        )

    candidates = sorted(
        (i for i in range(n) if reqs[i].prefill_eligible
         or reqs[i].decode_eligible),
        key=lambda i: (-(lp[i].prefill_indicator + lp[i].y),)
        + _execution_key(reqs[i]),
    )
    victim_list = sorted(
        (i for i in range(n) if reqs[i].preemption_eligible),
        key=lambda i: (-lp[i].z,) + _victim_key(reqs[i]),
    )

    x_hat, y_hat, z_hat = [0] * n, [0] * n, [0] * n
    accepted, accepted_order, victims = set(), [], set()
    totals = _Totals(
        tokens=0, actions=0, residents=problem.resident_count,
        free_before_decodes=problem.m_free, decode_demand=0,
        admissions=0, decodes=0,
    )
    trials = failed = declined = skipped = 0

    def diagnostics():
        return ExtractionDiagnostics(
            candidate_order=tuple(reqs[i].request_id for i in candidates),
            victim_candidate_order=tuple(
                reqs[i].request_id for i in victim_list),
            accepted_order=tuple(reqs[i].request_id for i in accepted_order),
            trial_count=trials,
            failed_trial_count=failed,
            declined_optional_trial_count=declined,
            skipped_candidate_count=skipped,
            fractional_request_count=sum(
                1 for d in lp
                if any(0.0 < v < 1.0 for v in (d.y, d.prefill_indicator, d.z))
            ),
        )

    try:
        for i in candidates:
            r = reqs[i]
            b_rem = problem.b_max - totals.tokens
            s_rem = problem.s_max - totals.actions
            if i in victims or b_rem < 1 or s_rem < 1:
                skipped += 1
                continue
            trials += 1
            trial = dataclasses.replace(
                totals, tokens=totals.tokens + 1, actions=totals.actions + 1
            )
            chunk = 0
            if r.decode_eligible:
                trial.decode_demand += r.decode_demand
                trial.decodes += 1
                reward = float(r.decode_utility)
            else:
                chunk = min(
                    r.prefill_upper_bound, b_rem, max(1, math.floor(lp[i].x))
                )
                trial.tokens = totals.tokens + chunk
                trial.free_before_decodes -= r.prefill_fixed_charge
                if not r.resident:
                    trial.residents += 1
                    trial.admissions += 1
                reward = _checked_product(
                    float(r.prefill_token_utility), chunk, "trial reward"
                )
            new_victims = []
            for k in victim_list:
                if trial.legal(problem):
                    break
                if k == i or k in accepted or k in victims:
                    continue
                new_victims.append(k)
                trial.free_before_decodes += reqs[k].preemption_recovery
                trial.residents -= 1
            if not trial.legal(problem):
                failed += 1
                continue
            penalty = _checked_sum(
                (float(reqs[k].preemption_penalty) for k in new_victims),
                "trial penalty",
            )
            if accepted:
                if new_victims:
                    gain = reward - penalty
                    tau = policy.objective_abs_tol + policy.objective_rel_tol \
                        * max(abs(reward), abs(penalty))
                    if not (math.isfinite(gain) and math.isfinite(tau)):
                        raise _NumericalError(
                            f"trial gain {gain!r} or threshold {tau!r} is "
                            "nonfinite"
                        )
                    if not gain > tau:
                        declined += 1
                        continue
                elif not reward >= 0.0:
                    declined += 1
                    continue
            totals = trial
            accepted.add(i)
            accepted_order.append(i)
            if r.decode_eligible:
                y_hat[i] = 1
            else:
                x_hat[i] = chunk
            for k in new_victims:
                victims.add(k)
                z_hat[k] = 1
    except _NumericalError as err:
        return _fail(problem, STAGE_EXTRACTION, OUTCOME_NUMERICAL,
                     CATEGORY_NONFINITE, str(err))

    if not accepted:
        return _fail(
            problem, STAGE_EXTRACTION, OUTCOME_NO_USEFUL_WORK,
            CATEGORY_NO_USEFUL_WORK,
            f"none of the {len(candidates)} eligible execution candidates "
            "has a legal whole action even with every other legal victim",
            extraction=diagnostics(),
        )

    for i in sorted(
        (i for i in accepted if x_hat[i] > 0),
        key=lambda i: _execution_key(reqs[i]),
    ):
        extra = min(
            reqs[i].prefill_upper_bound - x_hat[i],
            problem.b_max - totals.tokens,
        )
        x_hat[i] += extra
        totals.tokens += extra

    try:
        objective = _objective_max(problem, x_hat, y_hat, z_hat)
    except _NumericalError as err:
        return _fail(problem, STAGE_EXTRACTION, OUTCOME_NUMERICAL,
                     CATEGORY_NONFINITE, str(err))
    victim_order, execution_order = _declared_orders(
        problem, y_hat, z_hat, x_hat
    )
    decisions = tuple(
        IntegerDecision(
            reqs[i].request_id, reqs[i].order_key, x_hat[i], y_hat[i],
            z_hat[i], 1 if x_hat[i] > 0 else 0,
        )
        for i in range(n)
    )
    return IntegerPlan(
        problem_id=problem.problem_id,
        decisions=decisions,
        victim_order=victim_order,
        execution_order=execution_order,
        token_use=totals.tokens,
        action_count=totals.actions,
        resident_count_after=totals.residents,
        free_blocks_before_decodes=totals.free_before_decodes,
        decode_demand=totals.decode_demand,
        numerical_policy=policy,
        objective=objective,
        extraction=diagnostics(),
    )


# ---------------------------------------------------------------------------
# Integer-plan validation
# ---------------------------------------------------------------------------


def validate_integer_plan(problem, plan):
    """Exactly validate a complete integer plan; return it or a ``Failure``.

    Recomputes every total from the per-request decisions with exact integer
    arithmetic and does not trust the plan's reported totals or orders. A
    rejection is an implementation or validation failure; it never proves
    that useful work is impossible.
    """
    reqs = problem.requests

    def bad(category, reason, outcome=OUTCOME_INVALID):
        return _fail(problem, STAGE_PLAN, outcome, category, reason)

    if not isinstance(plan, IntegerPlan):
        return bad("malformed_plan", "plan must be an IntegerPlan")
    if plan.problem_id != problem.problem_id:
        return bad("snapshot_mismatch", "plan problem_id differs from problem")
    if plan.numerical_policy != problem.numerical_policy:
        return bad("policy_mismatch", "plan numerical policy differs")
    if not isinstance(plan.extraction, ExtractionDiagnostics):
        return bad("malformed_plan", "plan extraction diagnostics missing")
    if not isinstance(plan.decisions, tuple) or len(plan.decisions) != len(
        reqs
    ):
        return bad("decision_count", "plan must have one decision per request")

    tokens = actions = demand = admissions = decodes = 0
    free = problem.m_free
    residents = problem.resident_count
    for r, d in zip(reqs, plan.decisions):
        rid = r.request_id
        if not (
            isinstance(d, IntegerDecision) and d.request_id == rid
            and d.order_key == r.order_key
        ):
            return bad("association_mismatch", f"decision order for {rid!r}")
        if not all(
            _is_int(v) for v in (
                d.prefill_tokens, d.decode, d.preempt, d.prefill_indicator
            )
        ):
            return bad("domain_violation", f"{rid!r}: non-integer decision")
        if d.prefill_tokens < 0 or any(
            v not in (0, 1) for v in (d.decode, d.preempt, d.prefill_indicator)
        ):
            return bad("domain_violation", f"{rid!r}: value outside domain")
        if d.prefill_indicator != (1 if d.prefill_tokens > 0 else 0):
            return bad("canonical_indicator", f"{rid!r}: indicator != 1{{x>0}}")
        if d.prefill_indicator + d.decode + d.preempt > 1:
            return bad("mutual_exclusion", f"{rid!r}: more than one action")
        if d.prefill_tokens > 0 and not (
            r.prefill_eligible and d.prefill_tokens <= r.prefill_upper_bound
        ):
            return bad("ineligible_action", f"{rid!r}: illegal prefill chunk")
        if d.decode and not r.decode_eligible:
            return bad("ineligible_action", f"{rid!r}: decode not eligible")
        if d.preempt and not (
            r.preemption_eligible and r.resident
            and rid in problem.legal_preemption_ids
        ):
            return bad("ineligible_action", f"{rid!r}: preemption not legal")
        tokens += d.prefill_tokens + d.decode
        actions += d.prefill_indicator + d.decode
        demand += r.decode_demand * d.decode
        decodes += d.decode
        free += r.preemption_recovery * d.preempt
        free -= r.prefill_fixed_charge * d.prefill_indicator
        residents -= d.preempt
        if not r.resident:
            residents += d.prefill_indicator
            admissions += d.prefill_indicator

    checks = (
        ("token_volume", tokens <= problem.b_max),
        ("action_width", actions <= problem.s_max),
        ("resident_capacity", residents <= problem.resident_limit),
        ("actual_memory", free >= demand),
        ("decode_prerequisite", decodes == 0 or free >= 1),
        ("admission_watermark",
         admissions == 0 or free >= problem.admission_watermark),
        ("useful_execution", actions >= 1),
    )
    for name, ok in checks:
        if not ok:
            return bad(
                "constraint_violation",
                f"{name} fails: tokens={tokens} actions={actions} "
                f"residents_after={residents} H={free} decode_demand={demand} "
                f"admissions={admissions} decodes={decodes}",
            )
    reported = (
        plan.token_use, plan.action_count, plan.resident_count_after,
        plan.free_blocks_before_decodes, plan.decode_demand,
    )
    recomputed = (tokens, actions, residents, free, demand)
    if reported != recomputed:
        return bad(
            "total_mismatch",
            f"reported totals {reported} disagree with recomputed "
            f"{recomputed}",
        )
    victim_order, execution_order = _declared_orders(
        problem,
        [d.decode for d in plan.decisions],
        [d.preempt for d in plan.decisions],
        [d.prefill_tokens for d in plan.decisions],
    )
    if plan.victim_order != victim_order:
        return bad("order_mismatch", f"victim order must be {victim_order}")
    if plan.execution_order != execution_order:
        return bad(
            "order_mismatch", f"execution order must be {execution_order}"
        )
    try:
        objective = _objective_max(
            problem,
            [d.prefill_tokens for d in plan.decisions],
            [d.decode for d in plan.decisions],
            [d.preempt for d in plan.decisions],
        )
    except _NumericalError as err:
        return bad(CATEGORY_NONFINITE, str(err), OUTCOME_NUMERICAL)
    if plan.objective != objective:
        return bad(
            "objective_mismatch",
            f"reported objective {plan.objective!r} differs from "
            f"recomputed {objective!r}",
        )
    return plan


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------


def solve_and_extract(problem):
    """Validate, solve once, validate the relaxation, construct whole
    actions, and validate the complete integer plan."""
    canonical = validate_problem(problem)
    if isinstance(canonical, Failure):
        return canonical
    solver = solve_relaxation(canonical)
    if solver.category != CATEGORY_OPTIMAL:
        return _fail(
            canonical, STAGE_SOLVER, _solver_outcome(solver.category),
            solver.category, solver.reason, solver,
        )
    relaxed = validate_relaxed_solution(canonical, solver.raw_vector)
    if isinstance(relaxed, Failure):
        return dataclasses.replace(relaxed, solver=solver)
    plan = extract_integer_plan(canonical, relaxed)
    if isinstance(plan, Failure):
        return dataclasses.replace(plan, solver=solver)
    plan = validate_integer_plan(canonical, plan)
    if isinstance(plan, Failure):
        return dataclasses.replace(plan, solver=solver)
    return SchedulingSuccess(canonical.problem_id, plan, relaxed, solver)


# ---------------------------------------------------------------------------
# Hard-coded smoke case
# ---------------------------------------------------------------------------


def _smoke_problem():
    # One waiting prefill (admission charge 1) and one resident zero-demand
    # decode. Two free blocks leave H = 1 after the admission, which the
    # decode prerequisite requires.
    return LPProblem(
        requests=(
            RequestInput(
                request_id="smoke-prefill", order_key=(0, 10), resident=False,
                prompt_tokens_remaining=4, prefill_upper_bound=4,
                prefill_eligible=True, decode_eligible=False,
                preemption_eligible=False,
                prefill_fixed_charge=1, decode_demand=0,
                preemption_recovery=0,
                decode_utility=0.0, prefill_token_utility=1.0,
                preemption_penalty=0.0,
                service_rank=1, victim_rank=None,
            ),
            RequestInput(
                request_id="smoke-decode", order_key=(0, 20), resident=True,
                prompt_tokens_remaining=0, prefill_upper_bound=0,
                prefill_eligible=False, decode_eligible=True,
                preemption_eligible=False,
                prefill_fixed_charge=0, decode_demand=0,
                preemption_recovery=0,
                decode_utility=10.0, prefill_token_utility=0.0,
                preemption_penalty=0.0,
                service_rank=0, victim_rank=None,
            ),
        ),
        legal_preemption_ids=frozenset(),
        b_max=3, c_max=4, s_max=2, resident_limit=2, resident_count=1,
        m_free=2, admission_watermark=0,
        problem_id="smoke-problem",
        numerical_policy=NumericalPolicy(),
    )


def format_result(result):
    if isinstance(result, Failure):
        return (
            f"lp_relaxation FAILURE problem_id={result.problem_id} "
            f"stage={result.stage} outcome={result.outcome} "
            f"category={result.category}\n"
            f"  reason: {result.reason}"
        )
    plan, diag = result.plan, result.solver.diagnostics
    lines = [
        f"lp_relaxation SUCCESS problem_id={result.problem_id}",
        f"  numerical_policy: {plan.numerical_policy.policy_id}",
        f"  solver: {diag.method} scipy={diag.scipy_version} "
        f"status={diag.raw_status} iterations={diag.iterations}",
        f"  relaxed objective (max): {result.relaxed.normalized_objective}",
        f"  integer objective (max): {plan.objective}",
        "  plan (ascending order_key):",
    ]
    for d in plan.decisions:
        action = (
            "prefill" if d.prefill_tokens else
            "decode" if d.decode else
            "preempt" if d.preempt else "none"
        )
        lines.append(
            f"    {d.request_id} order_key={d.order_key} action={action} "
            f"prefill_tokens={d.prefill_tokens} decode={d.decode} "
            f"preempt={d.preempt} prefill_indicator={d.prefill_indicator}"
        )
    lines += [
        "  totals: "
        f"prefill_tokens={sum(d.prefill_tokens for d in plan.decisions)} "
        f"decode_actions={sum(d.decode for d in plan.decisions)} "
        f"preemptions={sum(d.preempt for d in plan.decisions)}",
        "  final totals: "
        f"tokens={plan.token_use} actions={plan.action_count} "
        f"residents_after={plan.resident_count_after} "
        f"H={plan.free_blocks_before_decodes} "
        f"decode_demand={plan.decode_demand}",
        f"  victim_order: {plan.victim_order}",
        f"  execution_order: {plan.execution_order}",
        f"  candidate_order: {plan.extraction.candidate_order}",
        f"  fractional_request_count (observation): "
        f"{plan.extraction.fractional_request_count}",
    ]
    return "\n".join(lines)


def main():
    result = solve_and_extract(_smoke_problem())
    print(format_result(result))
    return 1 if isinstance(result, Failure) else 0


if __name__ == "__main__":
    raise SystemExit(main())
