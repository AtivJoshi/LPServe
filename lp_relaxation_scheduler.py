"""Standalone, CPU-only LP-relaxation scheduling layer.

Builds the one-step myopic scheduling LP (docs/lp_scheduler_design.md section 7)
from immutable inputs, solves it with SciPy/HiGHS, validates the relaxed
solution, deterministically extracts an integer plan (design section 10), and
validates that plan exactly (design section 11.3).

The module reads no clocks, global state, or randomness, and imports nothing
from the serving framework. Every failure is returned as a structured
``Failure`` that carries no plan.

Variable layout for ``n`` requests sorted by ascending ``order_key``: the LP
vector is ``[x_0..x_{n-1}, y_0.., I_0.., z_0..]`` where ``I`` is the prefill
indicator.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
from scipy.optimize import linprog

STAGE_INPUT = "input_validation"
STAGE_SOLVER = "solver"
STAGE_RELAXED = "relaxed_validation"
STAGE_EXTRACTION = "extraction"
STAGE_PLAN = "plan_validation"

CATEGORY_OPTIMAL = "optimal_candidate"
_STATUS_CATEGORIES = {
    1: "solver_limit",
    2: "infeasible",
    3: "unbounded",
    4: "numerical_difficulty",
}

SOLVER_METHOD = "highs-ds"
SOLVER_OPTIONS = (("presolve", True),)


# ---------------------------------------------------------------------------
# Immutable records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalPolicy:
    """Provisional numerical policy; every value is externally visible."""

    policy_id: str = "lp_relaxation_mvp_v1"
    feasibility_tol: float = 1e-7
    integrality_tol: float = 1e-6
    objective_abs_tol: float = 1e-9
    objective_rel_tol: float = 1e-9


@dataclass(frozen=True)
class RequestInput:
    request_id: str
    order_key: tuple
    prompt_tokens_remaining: int
    prefill_upper_bound: int
    prefill_eligible: bool
    decode_eligible: bool
    preemption_eligible: bool
    prefill_fixed_charge: int
    decode_charge: int
    preemption_recovery: int
    decode_utility: float
    prefill_token_utility: float
    preemption_penalty: float


@dataclass(frozen=True)
class LPProblem:
    requests: tuple
    legal_preemption_ids: frozenset
    b_max: int
    c_max: int
    s_max: int
    m_free: int
    w: int
    problem_id: str
    decode_memory_policy_id: str
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
class IntegerDecision:
    request_id: str
    order_key: tuple
    prefill_tokens: int
    decode: int
    preempt: int
    prefill_indicator: int


@dataclass(frozen=True)
class IntegerPlan:
    problem_id: str
    decisions: tuple
    residual_token_capacity: int
    residual_action_capacity: int
    residual_memory_capacity: int
    fractional_request_count: int
    dominant_preemption_ids: tuple
    safety_preemption_ids: tuple
    numerical_policy: NumericalPolicy
    objective: float


@dataclass(frozen=True)
class Failure:
    problem_id: Optional[str]
    stage: str
    category: str
    reason: str
    solver: Optional[SolverResult] = None


@dataclass(frozen=True)
class SchedulingSuccess:
    problem_id: str
    plan: IntegerPlan
    relaxed: RelaxedSolution
    solver: SolverResult


def _fail(problem, stage, category, reason, solver=None):
    problem_id = getattr(problem, "problem_id", None)
    if not isinstance(problem_id, str):
        problem_id = None
    return Failure(problem_id, stage, category, reason, solver)


class _InputError(Exception):
    def __init__(self, category, reason):
        super().__init__(reason)
        self.category = category
        self.reason = reason


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


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
        "prefill_fixed_charge", "decode_charge", "preemption_recovery",
    ):
        value = getattr(r, name)
        _require(
            _is_int(value) and value >= 0, "malformed_value",
            f"request {rid!r}: {name} must be a non-negative integer",
        )
    for name in (
        "decode_utility", "prefill_token_utility", "preemption_penalty",
    ):
        _require(
            _is_real(getattr(r, name)), "malformed_value",
            f"request {rid!r}: {name} must be a finite number",
        )
    for name in ("prefill_eligible", "decode_eligible", "preemption_eligible"):
        _require(
            isinstance(getattr(r, name), bool), "malformed_value",
            f"request {rid!r}: {name} must be a bool",
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


def _validate_problem_or_raise(problem):
    _require(
        isinstance(problem, LPProblem), "malformed_value",
        "problem must be an LPProblem",
    )
    for name in ("problem_id", "decode_memory_policy_id"):
        value = getattr(problem, name)
        _require(
            isinstance(value, str) and value, "malformed_value",
            f"{name} must be a non-empty string",
        )
    _check_policy(problem.numerical_policy)
    for name in ("b_max", "s_max", "m_free", "w"):
        value = getattr(problem, name)
        _require(
            _is_int(value) and value >= 0, "malformed_value",
            f"{name} must be a non-negative integer",
        )
    _require(
        _is_int(problem.c_max) and problem.c_max > 0, "malformed_value",
        "c_max must be a positive integer",
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
    ordered = tuple(sorted(problem.requests, key=lambda r: r.order_key))
    return dataclasses.replace(problem, requests=ordered)


def validate_problem(problem):
    """Return the canonical problem (requests sorted ascending by order_key)
    or a ``Failure``. Later stages require this canonical form."""
    try:
        return _validate_problem_or_raise(problem)
    except _InputError as err:
        return _fail(problem, STAGE_INPUT, err.category, err.reason)


# ---------------------------------------------------------------------------
# LP construction and solver adapter
# ---------------------------------------------------------------------------


def _build_lp(problem):
    """Return (c, A_ub, b_ub, bounds) in SciPy minimization form."""
    reqs = problem.requests
    n = len(reqs)
    c = np.zeros(4 * n)
    bounds = []
    for i, r in enumerate(reqs):
        c[i] = -r.prefill_token_utility
        c[n + i] = -r.decode_utility
        if r.preemption_eligible:
            c[3 * n + i] = r.preemption_penalty
    bounds += [(0.0, None)] * n
    bounds += [(0.0, 1.0 if r.decode_eligible else 0.0) for r in reqs]
    bounds += [(0.0, 1.0 if r.prefill_eligible else 0.0) for r in reqs]
    bounds += [(0.0, 1.0 if r.preemption_eligible else 0.0) for r in reqs]

    rows = np.zeros((3 + 3 * n, 4 * n))
    rhs = np.zeros(3 + 3 * n)
    for i, r in enumerate(reqs):
        rows[0, i] = 1.0
        rows[0, n + i] = 1.0
        rows[1, 2 * n + i] = 1.0
        rows[1, n + i] = 1.0
        rows[2, 2 * n + i] = r.prefill_fixed_charge
        rows[2, n + i] = r.decode_charge
        rows[2, 3 * n + i] = -r.preemption_recovery
    rhs[0] = problem.b_max
    rhs[1] = problem.s_max
    rhs[2] = problem.m_free - problem.w
    for i, r in enumerate(reqs):
        link_lo, link_hi, mutex = 3 + 3 * i, 4 + 3 * i, 5 + 3 * i
        rows[link_lo, 2 * n + i] = 1.0  # I - x <= 0
        rows[link_lo, i] = -1.0
        rows[link_hi, i] = 1.0  # x - U*I <= 0
        rows[link_hi, 2 * n + i] = -r.prefill_upper_bound
        rows[mutex, 2 * n + i] = 1.0  # I + y + z <= 1
        rows[mutex, n + i] = 1.0
        rows[mutex, 3 * n + i] = 1.0
        rhs[mutex] = 1.0
    return c, rows, rhs, bounds


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
        raw_status=status if _is_int(status) else None,
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
    recomputed = math.fsum(float(ci) * v for ci, v in zip(c, vector))
    reported = float(fun)
    if not _objective_agrees(reported, recomputed, policy):
        return result(
            "malformed_output",
            f"reported objective {reported!r} disagrees with recomputed "
            f"{recomputed!r}",
            reported, recomputed,
        )
    return result(CATEGORY_OPTIMAL, "optimal candidate", reported, recomputed)


def solve_relaxation(problem):
    """Solve the LP for a canonical problem; never raises for solver outcomes."""
    c, rows, rhs, bounds = _build_lp(problem)
    expected = int(c.size)
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


# ---------------------------------------------------------------------------
# Relaxed-solution validation
# ---------------------------------------------------------------------------


def _split(vector, n):
    return (
        list(vector[0:n]), list(vector[n:2 * n]),
        list(vector[2 * n:3 * n]), list(vector[3 * n:4 * n]),
    )


def _residuals(problem, x, y, ind, z):
    """Recompute every constraint residual (lhs - rhs) from project data."""
    reqs = problem.requests
    n = len(reqs)
    rng = range(n)
    return {
        "token_volume": math.fsum(x[i] + y[i] for i in rng) - problem.b_max,
        "action_width": math.fsum(ind[i] + y[i] for i in rng) - problem.s_max,
        "planning_memory": math.fsum(
            reqs[i].prefill_fixed_charge * ind[i]
            + reqs[i].decode_charge * y[i]
            - reqs[i].preemption_recovery * z[i]
            for i in rng
        ) - (problem.m_free - problem.w),
        "prefill_linkage_lower": max(ind[i] - x[i] for i in rng),
        "prefill_linkage_upper": max(
            x[i] - reqs[i].prefill_upper_bound * ind[i] for i in rng
        ),
        "decode_causality": max(
            y[i] - (1.0 if reqs[i].prompt_tokens_remaining == 0 else 0.0)
            for i in rng
        ),
        "mutual_exclusion": max(ind[i] + y[i] + z[i] - 1.0 for i in rng),
    }


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
    reqs = problem.requests
    return math.fsum(
        r.decode_utility * y[i] + r.prefill_token_utility * x[i]
        - (r.preemption_penalty * z[i] if r.preemption_eligible else 0.0)
        for i, r in enumerate(reqs)
    )


def validate_relaxed_solution(problem, raw_vector):
    """Independently validate a raw primal vector for a canonical problem.

    Returns a ``RelaxedSolution`` (normalized vector plus preserved raw
    vector) or a ``Failure``.
    """
    reqs = problem.requests
    n = len(reqs)
    eps = problem.numerical_policy.feasibility_tol
    try:
        raw = tuple(float(v) for v in raw_vector)
    except (TypeError, ValueError):
        return _fail(
            problem, STAGE_RELAXED, "malformed_vector",
            "raw vector is not a sequence of numbers",
        )
    if len(raw) != 4 * n or not all(math.isfinite(v) for v in raw):
        return _fail(
            problem, STAGE_RELAXED, "malformed_vector",
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
            return _fail(
                problem, STAGE_RELAXED, "bound_violation",
                f"{block} of request {reqs[k % n].request_id!r} = {value!r} "
                f"is outside its bounds beyond tolerance {eps}",
            )
        normalized.append(projected[0])
        projection_count += projected[1]

    raw_res = _residuals(problem, *_split(raw, n))
    norm_res = _residuals(problem, *_split(normalized, n))
    for label, res in (("raw", raw_res), ("normalized", norm_res)):
        for name, value in res.items():
            if value > eps:
                return _fail(
                    problem, STAGE_RELAXED, "constraint_violation",
                    f"{label} {name} residual {value!r} exceeds tolerance "
                    f"{eps}",
                )
    x, y, ind, z = _split(normalized, n)
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
        normalized_objective=_objective_max(problem, x, y, z),
    )


# ---------------------------------------------------------------------------
# Deterministic extraction
# ---------------------------------------------------------------------------


def _classify(value, eps):
    """0, 1, or None (fractional) for a validated indicator in [0, 1]."""
    if value <= eps:
        return 0
    if value >= 1.0 - eps:
        return 1
    return None


def extract_integer_plan(problem, relaxed):
    """Deterministic integer extraction; returns an unvalidated plan or a
    ``Failure``. Callers must pass the result through ``validate_integer_plan``.
    """
    reqs = problem.requests
    n = len(reqs)
    eps = problem.numerical_policy.integrality_tol
    if len(relaxed.decisions) != n or any(
        d.request_id != r.request_id for d, r in zip(relaxed.decisions, reqs)
    ):
        return _fail(
            problem, STAGE_EXTRACTION, "association_mismatch",
            "relaxed decisions do not match the ordered problem requests",
        )

    x_hat, y_hat, z_hat, i_hat = [0] * n, [0] * n, [0] * n, [0] * n
    frac = []
    for i, (r, d) in enumerate(zip(reqs, relaxed.decisions)):
        cy = _classify(d.y, eps)
        ci = _classify(d.prefill_indicator, eps)
        cz = _classify(d.z, eps)
        if cy is None or ci is None or cz is None:
            frac.append(i)
            continue
        floor_x = math.floor(d.x)
        if (ci == 1 and floor_x < 1) or (ci == 0 and floor_x > 0):
            return _fail(
                problem, STAGE_EXTRACTION, "indicator_inconsistent_with_x",
                f"request {r.request_id!r}: locked prefill indicator {ci} "
                f"is inconsistent with unchanged x={d.x!r}",
            )
        x_hat[i], y_hat[i], z_hat[i], i_hat[i] = floor_x, cy, cz, ci

    frac_set = set(frac)
    locked = [i for i in range(n) if i not in frac_set]
    b_curr = problem.b_max - sum(x_hat[i] + y_hat[i] for i in locked)
    s_curr = problem.s_max - sum(i_hat[i] + y_hat[i] for i in locked)
    m_curr = (problem.m_free - problem.w) - sum(
        reqs[i].prefill_fixed_charge * i_hat[i]
        + reqs[i].decode_charge * y_hat[i]
        - reqs[i].preemption_recovery * z_hat[i]
        for i in locked
    )

    dominant = []
    for i in frac:
        r, d = reqs[i], relaxed.decisions[i]
        if (
            r.preemption_eligible and r.preemption_recovery > 0
            and d.z > d.y and d.z > d.prefill_indicator
        ):
            z_hat[i] = 1
            m_curr += r.preemption_recovery
            dominant.append(i)

    safety = []
    while m_curr < 0:
        candidates = [
            i for i in frac
            if reqs[i].preemption_eligible and z_hat[i] == 0
            and reqs[i].preemption_recovery > 0
        ]
        if not candidates:
            return _fail(
                problem, STAGE_EXTRACTION, "memory_repair_exhausted",
                f"planning memory {m_curr} is negative and no unused legal "
                "preemption candidate with positive recovery remains",
            )
        k = min(
            candidates, key=lambda i: (-relaxed.decisions[i].z,
                                       reqs[i].order_key),
        )
        z_hat[k] = 1
        m_curr += reqs[k].preemption_recovery
        safety.append(k)

    for i in frac:
        if z_hat[i] == 1:
            continue
        r, d = reqs[i], relaxed.decisions[i]
        if d.y == 0.0 and d.prefill_indicator == 0.0:
            continue
        if d.y >= d.prefill_indicator:
            if b_curr >= 1 and s_curr >= 1 and m_curr >= r.decode_charge:
                y_hat[i] = 1
                b_curr -= 1
                s_curr -= 1
                m_curr -= r.decode_charge
        elif b_curr >= 1 and s_curr >= 1 and m_curr >= r.prefill_fixed_charge:
            chunk = min(r.prompt_tokens_remaining, problem.c_max, b_curr)
            if chunk > 0:
                x_hat[i] = chunk
                i_hat[i] = 1
                b_curr -= chunk
                s_curr -= 1
                m_curr -= r.prefill_fixed_charge

    for i in range(n):
        if i_hat[i] != (1 if x_hat[i] > 0 else 0):
            return _fail(
                problem, STAGE_EXTRACTION, "noncanonical_prefill",
                f"request {reqs[i].request_id!r}: indicator {i_hat[i]} "
                f"disagrees with prefill tokens {x_hat[i]}",
            )

    def by_key(indices):
        return tuple(
            reqs[i].request_id
            for i in sorted(indices, key=lambda j: reqs[j].order_key)
        )

    decisions = tuple(
        IntegerDecision(
            reqs[i].request_id, reqs[i].order_key, x_hat[i], y_hat[i],
            z_hat[i], i_hat[i],
        )
        for i in range(n)
    )
    return IntegerPlan(
        problem_id=problem.problem_id,
        decisions=decisions,
        residual_token_capacity=b_curr,
        residual_action_capacity=s_curr,
        residual_memory_capacity=m_curr,
        fractional_request_count=len(frac),
        dominant_preemption_ids=by_key(dominant),
        safety_preemption_ids=by_key(safety),
        numerical_policy=problem.numerical_policy,
        objective=_objective_max(
            problem, x_hat, y_hat, [float(v) for v in z_hat]
        ),
    )


# ---------------------------------------------------------------------------
# Integer-plan validation
# ---------------------------------------------------------------------------


def validate_integer_plan(problem, plan):
    """Exactly validate a complete integer plan; return it or a ``Failure``."""
    reqs = problem.requests

    def bad(category, reason):
        return _fail(problem, STAGE_PLAN, category, reason)

    if not isinstance(plan, IntegerPlan):
        return bad("malformed_plan", "plan must be an IntegerPlan")
    if plan.problem_id != problem.problem_id:
        return bad("snapshot_mismatch", "plan problem_id differs from problem")
    if plan.numerical_policy != problem.numerical_policy:
        return bad("policy_mismatch", "plan numerical policy differs")
    if len(plan.decisions) != len(reqs):
        return bad("decision_count", "plan must have one decision per request")

    tokens = actions = memory = 0
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
            r.preemption_eligible and rid in problem.legal_preemption_ids
        ):
            return bad("ineligible_action", f"{rid!r}: preemption not legal")
        tokens += d.prefill_tokens + d.decode
        actions += d.prefill_indicator + d.decode
        memory += (
            r.prefill_fixed_charge * d.prefill_indicator
            + r.decode_charge * d.decode
            - r.preemption_recovery * d.preempt
        )

    residuals = (
        problem.b_max - tokens,
        problem.s_max - actions,
        (problem.m_free - problem.w) - memory,
    )
    if any(v < 0 for v in residuals):
        return bad(
            "capacity_violation",
            f"residual token/action/memory capacities {residuals} are "
            "negative",
        )
    if residuals != (
        plan.residual_token_capacity, plan.residual_action_capacity,
        plan.residual_memory_capacity,
    ):
        return bad(
            "residual_mismatch",
            f"reported residuals disagree with recomputed {residuals}",
        )
    preempted = {d.request_id for d in plan.decisions if d.preempt}
    for label, ids in (
        ("dominant", plan.dominant_preemption_ids),
        ("safety", plan.safety_preemption_ids),
    ):
        if not set(ids) <= preempted:
            return bad("preemption_ids", f"{label} IDs are not preempted")
    if set(plan.dominant_preemption_ids) & set(plan.safety_preemption_ids):
        return bad("preemption_ids", "dominant and safety IDs overlap")
    if not (
        _is_int(plan.fractional_request_count)
        and 0 <= plan.fractional_request_count <= len(reqs)
    ):
        return bad("malformed_plan", "invalid fractional request count")
    return plan


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------


def solve_and_extract(problem):
    """Validate, solve, validate the relaxation, extract, validate the plan."""
    canonical = validate_problem(problem)
    if isinstance(canonical, Failure):
        return canonical
    solver = solve_relaxation(canonical)
    if solver.category != CATEGORY_OPTIMAL:
        return _fail(
            canonical, STAGE_SOLVER, solver.category, solver.reason, solver
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
    return LPProblem(
        requests=(
            RequestInput(
                request_id="smoke-prefill", order_key=(0, 10),
                prompt_tokens_remaining=4, prefill_upper_bound=4,
                prefill_eligible=True, decode_eligible=False,
                preemption_eligible=False,
                prefill_fixed_charge=1, decode_charge=0,
                preemption_recovery=0,
                decode_utility=0.0, prefill_token_utility=1.0,
                preemption_penalty=0.0,
            ),
            RequestInput(
                request_id="smoke-decode", order_key=(0, 20),
                prompt_tokens_remaining=0, prefill_upper_bound=0,
                prefill_eligible=False, decode_eligible=True,
                preemption_eligible=False,
                prefill_fixed_charge=0, decode_charge=0,
                preemption_recovery=0,
                decode_utility=10.0, prefill_token_utility=0.0,
                preemption_penalty=0.0,
            ),
        ),
        legal_preemption_ids=frozenset(),
        b_max=3, c_max=4, s_max=2, m_free=1, w=0,
        problem_id="smoke-problem",
        decode_memory_policy_id="fixture_supplied_charge_v1",
        numerical_policy=NumericalPolicy(),
    )


def format_result(result):
    if isinstance(result, Failure):
        return (
            f"lp_relaxation FAILURE problem_id={result.problem_id} "
            f"stage={result.stage} category={result.category}\n"
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
        "  residual capacities: "
        f"tokens={plan.residual_token_capacity} "
        f"actions={plan.residual_action_capacity} "
        f"memory={plan.residual_memory_capacity}",
        f"  fractional_request_count: {plan.fractional_request_count}",
        f"  dominant_preemption_ids: {plan.dominant_preemption_ids}",
        f"  safety_preemption_ids: {plan.safety_preemption_ids}",
    ]
    return "\n".join(lines)


def main():
    result = solve_and_extract(_smoke_problem())
    print(format_result(result))
    return 1 if isinstance(result, Failure) else 0


if __name__ == "__main__":
    raise SystemExit(main())
