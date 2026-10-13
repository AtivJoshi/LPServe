"""Focused CPU checks for the corrected LP-relaxation planning layer.

Fixture domains. Each fixture is labelled either ABSTRACT (abstract
formulation: valid model input, no claim that a validated runtime state has
this shape) or RUNTIME (admission-validated runtime: pool and watermark are
consistent, checked by ``runtime_pool_consistent``). The module itself applies
no total-pool admission restriction.

Dispositions of the superseded expectations (earlier revision of this file):

- ``test_main_smoke``: retained; the smoke inputs were corrected (two free
  blocks, exact zero decode demand, resident membership, ranks) so the decode
  prerequisite holds after the admission. Same printed totals.
- ``test_mixed_case_and_tied_extraction``: superseded. Its residual,
  dominant-preemption, and safety-preemption fields are retired with locking
  extraction. Replaced by ``test_declared_orders_and_totals`` and
  ``test_lp_values_primary_over_ranks_then_identity``.
- ``test_fractional_prefill``: superseded. It used a decode charge of 2,
  which the exact zero-or-one demand contract rejects. Fractional guidance is
  covered by ``test_chunk_floor_and_minimum_one`` and the nine-waiting
  watermark construction.
- ``test_zero_zero_execution_tie_is_no_action``: superseded and inverted.
  Zero-support candidates are now always tried
  (``test_integral_idle_victim_and_zero_support_candidate``); its fixture also
  gave a resident prefill a nonzero charge, now invalid input.
- ``test_visible_infeasibility``: superseded. It relied on the removed
  planning reserve; ``test_solver_reported_infeasibility_real_status_2``
  reaches status 2 through the actual-memory and useful-execution rows.
"""

import contextlib
import dataclasses
import io
import itertools
import math
import random
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import lp_relaxation_scheduler as lrs  # noqa: E402

ABSTRACT = "abstract formulation"
RUNTIME = "admission-validated runtime"
EPS = lrs.NumericalPolicy().feasibility_tol


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _spec(rid, key, **kw):
    spec = dict(
        request_id=rid, order_key=(key,), resident=True,
        prompt_tokens_remaining=0, prefill_eligible=False,
        decode_eligible=False, preemption_eligible=False,
        prefill_fixed_charge=0, decode_demand=0, preemption_recovery=0,
        decode_utility=0.0, prefill_token_utility=0.0, preemption_penalty=0.0,
        service_rank=None, victim_rank=None,
    )
    spec.update(kw)
    return spec


def waiting(rid, key, rem, charge, beta=0.0, service=None):
    return _spec(rid, key, resident=False, prompt_tokens_remaining=rem,
                 prefill_eligible=True, prefill_fixed_charge=charge,
                 prefill_token_utility=beta,
                 service_rank=key if service is None else service)


def decoder(rid, key, demand, blocks, alpha=0.0, gamma=0.0, service=None,
            victim=None, preemptible=True):
    return _spec(rid, key, decode_eligible=True, decode_demand=demand,
                 preemption_eligible=preemptible,
                 preemption_recovery=blocks if preemptible else 0,
                 decode_utility=alpha, preemption_penalty=gamma,
                 service_rank=key if service is None else service,
                 victim_rank=(key if victim is None else victim)
                 if preemptible else None)


def resident_prefill(rid, key, rem, blocks, beta=0.0, gamma=0.0, service=None,
                     victim=None, preemptible=True):
    return _spec(rid, key, prompt_tokens_remaining=rem, prefill_eligible=True,
                 preemption_eligible=preemptible,
                 preemption_recovery=blocks if preemptible else 0,
                 prefill_token_utility=beta, preemption_penalty=gamma,
                 service_rank=key if service is None else service,
                 victim_rank=(key if victim is None else victim)
                 if preemptible else None)


def idle(rid, key, blocks, gamma=0.0, victim=None):
    """Resident holding blocks, not eligible to execute, legally preemptible."""
    return _spec(rid, key, preemption_eligible=True,
                 preemption_recovery=blocks, preemption_penalty=gamma,
                 victim_rank=key if victim is None else victim)


def problem(specs, b, c, s, n, free, h, pid="test"):
    reqs = []
    for spec in specs:
        ub = (min(spec["prompt_tokens_remaining"], c)
              if spec["prefill_eligible"] else 0)
        reqs.append(lrs.RequestInput(prefill_upper_bound=ub, **spec))
    return lrs.LPProblem(
        requests=tuple(reqs),
        legal_preemption_ids=frozenset(
            r.request_id for r in reqs if r.preemption_eligible),
        b_max=b, c_max=c, s_max=s, resident_limit=n,
        resident_count=sum(1 for r in reqs if r.resident),
        m_free=free, admission_watermark=h, problem_id=pid,
        numerical_policy=lrs.NumericalPolicy(),
    )


def runtime_pool_consistent(prob):
    """Test-side label check: every resident has the live mapper's shape (a
    preemptible partial prefill or decode), pool T = free + all resident
    blocks, default watermark h = floor(0.01 T), and A_i + h <= T for each
    waiting prefill. This is only the resident-shape and pool/watermark part
    of initial admission validation."""
    if any(r.resident and not (
            r.preemption_eligible
            and (r.prefill_eligible or r.decode_eligible))
           for r in prob.requests):
        return False
    total = prob.m_free + sum(
        r.preemption_recovery for r in prob.requests if r.resident)
    return (
        prob.admission_watermark == total // 100
        and all(r.prefill_fixed_charge + prob.admission_watermark <= total
                for r in prob.requests if not r.resident)
        and prob.resident_count <= prob.resident_limit
    )


def canon(testcase, prob):
    canonical = lrs.validate_problem(prob)
    testcase.assertIsInstance(canonical, lrs.LPProblem, canonical)
    return canonical


def vector(canonical, values):
    """Layout [x.., y.., I.., z..] in canonical order from {id: {x,y,I,z}}."""
    n = len(canonical.requests)
    v = [0.0] * (4 * n)
    for i, r in enumerate(canonical.requests):
        d = values.get(r.request_id, {})
        v[i], v[n + i] = d.get("x", 0.0), d.get("y", 0.0)
        v[2 * n + i], v[3 * n + i] = d.get("I", 0.0), d.get("z", 0.0)
    return tuple(v)


def actions(plan):
    out = {}
    for d in plan.decisions:
        if d.prefill_tokens:
            out[d.request_id] = f"prefill:{d.prefill_tokens}"
        elif d.decode:
            out[d.request_id] = "decode"
        elif d.preempt:
            out[d.request_id] = "preempt"
    return out


# ---------------------------------------------------------------------------
# Independent oracle (test code only; does not call production validators)
# ---------------------------------------------------------------------------


def oracle_options(r):
    opts = [None]
    if r.decode_eligible:
        opts.append(("decode",))
    if r.prefill_eligible:
        opts += [("prefill", c) for c in range(1, r.prefill_upper_bound + 1)]
    if r.preemption_eligible:
        opts.append(("preempt",))
    return opts


def oracle_simulate(prob, choice):
    """Simulate the native allocation checks in the declared execution
    order: free all victims; admit each waiting prefill requiring
    free - A >= h; resident prefills allocate nothing; each zero-demand decode
    requires one free block; each one-block decode requires one free block
    and then consumes it. Returns (legal, useful, utility)."""
    reqs = prob.requests
    tokens = acts = 0
    residents = prob.resident_count
    free = prob.m_free
    util = []
    for r, c in zip(reqs, choice):
        if c is None:
            continue
        if c[0] == "preempt":
            free += r.preemption_recovery
            residents -= 1
            util.append(-r.preemption_penalty)
        elif c[0] == "decode":
            tokens += 1
            acts += 1
            util.append(r.decode_utility)
        else:
            tokens += c[1]
            acts += 1
            util.append(r.prefill_token_utility * c[1])
    legal = (tokens <= prob.b_max and acts <= prob.s_max)
    for r, c in zip(reqs, choice):
        if c is not None and c[0] == "prefill" and not r.resident:
            free -= r.prefill_fixed_charge
            residents += 1
            legal = legal and free >= prob.admission_watermark
    legal = legal and residents <= prob.resident_limit
    for demand in (0, 1):
        for r, c in zip(reqs, choice):
            if c is not None and c[0] == "decode" and r.decode_demand == demand:
                legal = legal and free >= 1
                free -= demand
    return legal, acts >= 1, math.fsum(util)


def oracle_enumerate(prob, limit=50000):
    """All complete integer plans of a small canonical problem."""
    option_lists = [oracle_options(r) for r in prob.requests]
    size = math.prod(len(o) for o in option_lists)
    assert size <= limit, f"enumeration size {size} exceeds bound {limit}"
    return itertools.product(*option_lists)


def oracle_best(prob):
    """(useful legal plan exists, best utility among them)."""
    best = None
    for choice in oracle_enumerate(prob):
        legal, useful, util = oracle_simulate(prob, choice)
        if legal and useful and (best is None or util > best):
            best = util
    return best is not None, best


def oracle_choice(prob, decisions):
    """Independently check a returned decision tuple and convert it to an
    action choice. Raises AssertionError on any violation of request
    association, integer domains, canonical indicators, action exclusivity,
    eligibility, or chunk bounds (including the request's own remainder).
    Resources are checked separately by ``oracle_simulate``."""
    reqs = prob.requests
    assert isinstance(decisions, tuple) and len(decisions) == len(reqs), \
        "one decision per request"
    out = []
    for r, d in zip(reqs, decisions):
        rid = r.request_id
        assert (d.request_id, d.order_key) == (rid, r.order_key), \
            f"{rid}: association"
        x, y, z, ind = d.prefill_tokens, d.decode, d.preempt, d.prefill_indicator
        assert all(type(v) is int for v in (x, y, z, ind)), f"{rid}: domain"
        assert x >= 0 and {y, z, ind} <= {0, 1}, f"{rid}: domain"
        assert ind == (1 if x > 0 else 0), f"{rid}: canonical indicator"
        assert ind + y + z <= 1, f"{rid}: conflicting action flags"
        if x:
            assert r.prefill_eligible, f"{rid}: prefill not eligible"
            assert x <= r.prompt_tokens_remaining, f"{rid}: exceeds remainder"
            assert x <= prob.c_max, f"{rid}: exceeds chunk cap"
            out.append(("prefill", x))
        elif y:
            assert r.decode_eligible, f"{rid}: decode not eligible"
            out.append(("decode",))
        elif z:
            assert r.preemption_eligible and r.resident \
                and rid in prob.legal_preemption_ids, f"{rid}: illegal victim"
            out.append(("preempt",))
        else:
            out.append(None)
    return tuple(out)


def plan_choice(plan, prob):
    return oracle_choice(prob, plan.decisions)


def choice_vector(prob, choice):
    n = len(prob.requests)
    v = [0.0] * (4 * n)
    for i, c in enumerate(choice):
        if c is None:
            continue
        if c[0] == "decode":
            v[n + i] = 1.0
        elif c[0] == "prefill":
            v[i], v[2 * n + i] = float(c[1]), 1.0
        else:
            v[3 * n + i] = 1.0
    return v


# ---------------------------------------------------------------------------
# Named fixtures from the manuscript and discussion
# ---------------------------------------------------------------------------


def abstract_lp_feasible_integer_infeasible():
    # ABSTRACT: F=3, h=2, two waiting requests needing two blocks each.
    return problem(
        [waiting("w1", 1, 1, 2, beta=1.0), waiting("w2", 2, 1, 2, beta=1.0)],
        b=2, c=1, s=2, n=2, free=3, h=2,
    )


def nine_waiting_watermark(n_waiting=9):
    # RUNTIME: background resident with 98 blocks, pool 100, h = 1.
    specs = [decoder("b", 100, 0, 98, alpha=0.0, gamma=100.0, victim=0)]
    specs += [waiting(f"w{j}", j, 1, 1, beta=1.0) for j in range(1, n_waiting + 1)]
    return problem(specs, b=2, c=1, s=2, n=n_waiting + 1, free=2, h=1)


def avoidable_negative_first():
    # RUNTIME: pool 12 (A 10, B 1, C 1, free 0), h = 0.
    return problem(
        [
            decoder("A", 1, 0, 10, alpha=0.0, gamma=8.0, service=2, victim=0),
            decoder("B", 2, 0, 1, alpha=5.0, gamma=100.0, service=0, victim=1),
            resident_prefill("C", 3, 1, 1, beta=0.0, gamma=100.0, service=1,
                             victim=2),
        ],
        b=1, c=1, s=2, n=3, free=0, h=0,
    )


def mass_preemption(positive):
    # RUNTIME: pool 111 (A 10, B 1, 100 others 1), free 0, h = 1.
    small_alpha, small_gamma = (0.001, 0.1) if positive else (0.0, 0.0)
    specs = [
        decoder("A", 1, 0, 10, alpha=small_alpha,
                gamma=1.0 if positive else 0.0, service=1, victim=1),
        decoder("B", 2, 0, 1, alpha=100.0, gamma=small_gamma, service=2,
                victim=2),
        waiting("W", 3, 1, 109, beta=10.5 if positive else 0.0, service=3),
    ]
    specs += [
        decoder(f"R{j:03d}", 10 + j, 0, 1, alpha=small_alpha,
                gamma=small_gamma)
        for j in range(100)
    ]
    return problem(specs, b=2, c=1, s=2, n=102, free=0, h=1)


MASS_POINT = {"B": {"y": 1.0}, "A": {"z": 0.05}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class LPRelaxationSchedulerTest(unittest.TestCase):

    def extract(self, prob, values):
        """Validate the supplied relaxed point, extract, validate the plan."""
        canonical = canon(self, prob)
        relaxed = lrs.validate_relaxed_solution(
            canonical, vector(canonical, values))
        self.assertIsInstance(relaxed, lrs.RelaxedSolution, relaxed)
        plan = lrs.extract_integer_plan(canonical, relaxed)
        if isinstance(plan, lrs.IntegerPlan):
            self.assertIs(lrs.validate_integer_plan(canonical, plan), plan)
            legal, useful, util = oracle_simulate(
                canonical, plan_choice(plan, canonical))
            self.assertTrue(legal and useful)
            self.assertAlmostEqual(util, plan.objective, places=9)
        return canonical, relaxed, plan

    # -- smoke and interface ------------------------------------------------

    def test_main_smoke(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = lrs.main()
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("lp_relaxation SUCCESS", text)
        self.assertIn("totals: prefill_tokens=2 decode_actions=1", text)
        self.assertNotIn("sarathi", sys.modules)

        result = lrs.solve_and_extract(lrs._smoke_problem())
        self.assertIsInstance(result, lrs.SchedulingSuccess)
        self.assertEqual(result.solver.diagnostics.raw_status, 0)
        self.assertEqual(result.solver.diagnostics.method, "highs-ds")
        self.assertEqual(
            actions(result.plan),
            {"smoke-prefill": "prefill:2", "smoke-decode": "decode"})
        self.assertEqual(result.plan.free_blocks_before_decodes, 1)

    def test_obsolete_fields_removed_without_compatibility(self):
        problem_fields = {f.name for f in dataclasses.fields(lrs.LPProblem)}
        request_fields = {f.name for f in dataclasses.fields(lrs.RequestInput)}
        plan_fields = {f.name for f in dataclasses.fields(lrs.IntegerPlan)}
        self.assertFalse({"w", "decode_memory_policy_id"} & problem_fields)
        self.assertNotIn("decode_charge", request_fields)
        self.assertFalse({
            "residual_token_capacity", "residual_action_capacity",
            "residual_memory_capacity", "dominant_preemption_ids",
            "safety_preemption_ids", "fractional_request_count",
        } & plan_fields)
        spec = waiting("a", 1, 1, 0)
        spec["decode_charge"] = spec.pop("decode_demand")
        with self.assertRaises(TypeError):
            lrs.RequestInput(prefill_upper_bound=1, **spec)
        prob = lrs._smoke_problem()
        with self.assertRaises(TypeError):
            lrs.LPProblem(**{f.name: getattr(prob, f.name)
                             for f in dataclasses.fields(prob)}, w=0)

    def test_input_validation(self):
        base = problem(
            [decoder("d", 1, 1, 2, alpha=1.0), waiting("w", 2, 3, 1, beta=1.0),
             resident_prefill("p", 3, 2, 1)],
            b=4, c=2, s=2, n=3, free=1, h=0,
        )
        self.assertIsInstance(lrs.validate_problem(base), lrs.LPProblem)

        def with_req(rid, **kw):
            reqs = tuple(dataclasses.replace(r, **kw)
                         if r.request_id == rid else r for r in base.requests)
            return dataclasses.replace(base, requests=reqs)

        cases = [
            ("b_max zero", dataclasses.replace(base, b_max=0)),
            ("s_max zero", dataclasses.replace(base, s_max=0)),
            ("c_max zero", dataclasses.replace(base, c_max=0)),
            ("resident limit zero", dataclasses.replace(base, resident_limit=0)),
            ("negative free", dataclasses.replace(base, m_free=-1)),
            ("negative watermark",
             dataclasses.replace(base, admission_watermark=-1)),
            ("bool capacity", dataclasses.replace(base, b_max=True)),
            ("float free", dataclasses.replace(base, m_free=1.0)),
            ("resident count mismatch",
             dataclasses.replace(base, resident_count=1)),
            ("resident count over limit", dataclasses.replace(
                base, resident_limit=1, resident_count=2,
                requests=base.requests[:2])),
            ("resident prefill charge", with_req("p", prefill_fixed_charge=1)),
            ("decode demand two", with_req("d", decode_demand=2)),
            ("waiting decode", with_req("w", decode_eligible=True)),
            ("waiting preemptible", with_req("w", preemption_eligible=True)),
            ("waiting recovery", with_req("w", preemption_recovery=1)),
            ("negative beta", with_req("w", prefill_token_utility=-1.0)),
            ("negative gamma", with_req("d", preemption_penalty=-0.5)),
            ("nan alpha", with_req("d", decode_utility=float("nan"))),
            ("inf beta", with_req("w", prefill_token_utility=float("inf"))),
            ("huge int beta", with_req("w", prefill_token_utility=10 ** 400)),
            ("huge int charge", with_req("w", prefill_fixed_charge=10 ** 400)),
            ("missing service rank", with_req("w", service_rank=None)),
            ("nan service rank", with_req("d", service_rank=float("nan"))),
            ("missing victim rank", with_req("d", victim_rank=None)),
            ("inf victim rank", with_req("p", victim_rank=float("inf"))),
            ("string rank", with_req("d", service_rank="0")),
            ("legal set mismatch", dataclasses.replace(
                base, legal_preemption_ids=frozenset({"d"}))),
            ("bad upper bound", with_req("w", prefill_upper_bound=3)),
            ("decode with prompt", with_req("d", prompt_tokens_remaining=1)),
            ("non-bool membership", with_req("d", resident=1)),
            ("duplicate order key", with_req("w", order_key=(1,))),
            ("empty", dataclasses.replace(base, requests=())),
        ]
        for name, bad in cases:
            with self.subTest(name):
                result = lrs.validate_problem(bad)
                self.assertIsInstance(result, lrs.Failure)
                self.assertEqual(result.stage, lrs.STAGE_INPUT)
                self.assertEqual(result.outcome, lrs.OUTCOME_INVALID)
                solved = lrs.solve_and_extract(bad)
                self.assertIsInstance(solved, lrs.Failure)
                self.assertIsNone(solved.solver)

    # -- LP rows and relaxed validation -------------------------------------

    def row_fixture(self, **over):
        # ABSTRACT fixture used for individual-row checks.
        params = dict(b=4, c=2, s=3, n=4, free=2, h=1)
        params.update(over)
        return problem(
            [decoder("d0", 1, 1, 2, alpha=1.0), decoder("d1", 2, 0, 1),
             resident_prefill("p0", 3, 3, 1),
             waiting("w0", 4, 2, 2, beta=1.0), waiting("w1", 5, 2, 1)],
            **params,
        )

    def test_lp_matrix_matches_independent_residuals(self):
        rng = random.Random(7)
        for h in (0, 1):
            prob = canon(self, self.row_fixture(h=h))
            c, a_ub, b_ub, bounds, labels = lrs._build_lp(prob)
            watermark = [lb for lb in labels if lb.startswith("admission")]
            self.assertEqual(
                watermark,
                [] if h == 0 else ["admission_watermark[w0]",
                                   "admission_watermark[w1]"])
            self.assertEqual(len(labels), len(set(labels)))
            for _ in range(20):
                v = [rng.uniform(-0.5, 2.0) for _ in range(c.size)]
                n = len(prob.requests)
                res = lrs._residuals(prob, *lrs._split(v, n))
                self.assertEqual(sorted(res), sorted(labels))
                matrix = a_ub @ np.array(v) - b_ub
                for label, value in zip(labels, matrix):
                    self.assertAlmostEqual(value, res[label], places=9,
                                           msg=label)

        # Hand-computed residuals for one point (h = 1).
        prob = canon(self, self.row_fixture())
        point = {"d0": {"y": 0.5}, "d1": {"y": 1.0, "z": 0.0},
                 "p0": {"x": 1.0, "I": 0.5, "z": 0.25},
                 "w0": {"x": 1.0, "I": 0.5}, "w1": {"z": 0.0}}
        v = vector(prob, point)
        res = lrs._residuals(prob, *lrs._split(v, len(prob.requests)))
        h_value = 2 + 1 * 0.25 - 2 * 0.5
        expected = {
            "token_volume": 1.0 + 1.0 + 0.5 + 1.0 - 4,
            "action_width": 0.5 + 1.0 + 0.5 + 0.5 - 3,
            "actual_memory": 1 * 0.5 - h_value,
            "decode_prerequisite": 1.5 - 3 * h_value,
            "admission_watermark[w0]": 1 * 0.5 - h_value,
            "admission_watermark[w1]": 0.0 - h_value,
            "resident_capacity": 3 - 0.25 + 0.5 - 4,
            "useful_execution": 1 - 2.5,
            "prefill_linkage_lower[p0]": 0.5 - 1.0,
            "prefill_linkage_upper[p0]": 1.0 - 2 * 0.5,
            "mutual_exclusion[p0]": 0.5 + 0.25 - 1,
            "prefill_linkage_upper[w0]": 1.0 - 2 * 0.5,
        }
        for label, value in expected.items():
            self.assertAlmostEqual(res[label], value, places=12, msg=label)

    def test_relaxed_validation_rejects_each_row(self):
        default = canon(self, self.row_fixture())
        ok = lrs.validate_relaxed_solution(
            default, vector(default, {"d1": {"y": 1.0}}))
        self.assertIsInstance(ok, lrs.RelaxedSolution)
        cases = [
            ("token_volume", dict(b=1),
             {"d1": {"y": 1.0}, "p0": {"x": 0.5, "I": 0.25}}),
            ("action_width", dict(s=1),
             {"d1": {"y": 1.0}, "p0": {"x": 0.5, "I": 0.5}}),
            ("actual_memory", dict(free=0),
             {"d0": {"y": 1.0}, "p0": {"z": 0.5}}),
            ("decode_prerequisite", dict(free=0),
             {"d1": {"y": 1.0}, "p0": {"z": 0.2}}),
            ("admission_watermark[w0]", {}, {"w0": {"x": 1.0, "I": 1.0}}),
            ("resident_capacity", dict(n=3), {"w1": {"x": 1.0, "I": 1.0}}),
            ("useful_execution", {}, {"d1": {"y": 0.5}}),
            ("prefill_linkage_lower[p0]", {},
             {"d1": {"y": 1.0}, "p0": {"x": 0.2, "I": 0.5}}),
            ("prefill_linkage_upper[p0]", {},
             {"d1": {"y": 1.0}, "p0": {"x": 1.5, "I": 0.5}}),
            ("mutual_exclusion[d0]", {},
             {"d1": {"y": 1.0}, "d0": {"y": 0.6, "z": 0.6}}),
        ]
        for label, over, point in cases:
            with self.subTest(label):
                prob = canon(self, self.row_fixture(**over))
                n = len(prob.requests)
                v = vector(prob, point)
                res = lrs._residuals(prob, *lrs._split(v, n))
                violated = [k for k, val in res.items() if val > EPS]
                self.assertEqual(violated, [label])
                result = lrs.validate_relaxed_solution(prob, v)
                self.assertIsInstance(result, lrs.Failure)
                self.assertEqual(result.category, "constraint_violation")
                self.assertEqual(result.outcome, lrs.OUTCOME_NUMERICAL)
                self.assertIn(f"raw {label} residual", result.reason)

    def test_relaxed_bounds_projection_and_raw_versus_normalized(self):
        prob = canon(self, self.row_fixture())
        bad = lrs.validate_relaxed_solution(
            prob, vector(prob, {"d1": {"y": 1.0}, "w0": {"y": 0.5}}))
        self.assertEqual(bad.category, "bound_violation")
        bad = lrs.validate_relaxed_solution(
            prob, vector(prob, {"d1": {"y": 1.0}, "w0": {"z": 1e-6}}))
        self.assertEqual(bad.category, "bound_violation")
        ok = lrs.validate_relaxed_solution(
            prob, vector(prob, {"d1": {"y": 1.0 + 5e-8}, "w0": {"z": 5e-8},
                                "d0": {"y": 0.5}}))
        self.assertIsInstance(ok, lrs.RelaxedSolution)
        self.assertEqual(ok.projection_count, 2)
        self.assertEqual(ok.decisions[1].y, 1.0)  # d1
        self.assertEqual(ok.decisions[0].y, 0.5)  # interior value untouched

        # ABSTRACT: raw residual exceeds tolerance, normalized does not.
        prob = canon(self, problem(
            [decoder("d", 1, 0, 0, preemptible=False),
             resident_prefill("p", 2, 2, 0, preemptible=False)],
            b=2, c=2, s=2, n=2, free=5, h=0))
        raw_bad = lrs.validate_relaxed_solution(prob, vector(
            prob, {"d": {"y": 1 + 0.9e-7}, "p": {"x": 1 + 0.5e-7, "I": 1.0}}))
        self.assertIsInstance(raw_bad, lrs.Failure)
        self.assertIn("raw token_volume", raw_bad.reason)

        # ABSTRACT: raw residual within tolerance, normalized exceeds it.
        prob = canon(self, problem(
            [decoder("d", 1, 0, 0, preemptible=False)]
            + [resident_prefill(p, k, 2, 0, preemptible=False)
               for p, k in (("p1", 2), ("p2", 3), ("p3", 4))],
            b=2, c=2, s=4, n=4, free=5, h=0))
        point = {"d": {"y": 1.0}, "p1": {"x": -0.9e-7}, "p2": {"x": -0.9e-7},
                 "p3": {"x": 1 + 2e-7, "I": 1.0}}
        norm_bad = lrs.validate_relaxed_solution(prob, vector(prob, point))
        self.assertIsInstance(norm_bad, lrs.Failure)
        self.assertIn("normalized token_volume", norm_bad.reason)

    def test_malformed_vectors(self):
        prob = canon(self, self.row_fixture())
        good = list(vector(prob, {"d1": {"y": 1.0}}))
        for name, raw in (
            ("short", good[:-1]), ("nan", good[:-1] + [float("nan")]),
            ("inf", [float("inf")] + good[1:]), ("text", ["a"] * len(good)),
            ("none", None),
        ):
            with self.subTest(name):
                result = lrs.validate_relaxed_solution(prob, raw)
                self.assertIsInstance(result, lrs.Failure)
                self.assertEqual(result.category, "malformed_vector")
                self.assertEqual(result.outcome, lrs.OUTCOME_NUMERICAL)

    # -- watermark, resident capacity, and fixed charge ---------------------

    def test_watermark_zero_positive_and_shared(self):
        # ABSTRACT (pool 2 implies h = 0 at runtime; h = 1 is formulation-only).
        # The two admissions are symmetric, so for h = 1 only the count of
        # admissions is asserted, not which one (no optimizer tie outcome).
        specs = [waiting("w1", 1, 1, 1, beta=1.0), waiting("w2", 2, 1, 1, beta=1.0)]
        for h, admitted in ((0, 2), (1, 1)):
            with self.subTest(h=h):
                result = lrs.solve_and_extract(
                    problem(specs, b=2, c=1, s=2, n=2, free=2, h=h))
                self.assertIsInstance(result, lrs.SchedulingSuccess, result)
                self.assertEqual(len(actions(result.plan)), admitted)
                self.assertEqual(result.plan.free_blocks_before_decodes,
                                 2 - admitted)
                if h == 1:  # unique LP optimum I = 2/3 for both
                    for d in result.relaxed.decisions:
                        self.assertAlmostEqual(d.prefill_indicator, 2 / 3,
                                               places=6)
                    self.assertEqual(
                        result.plan.extraction.failed_trial_count, 1)

        # ABSTRACT: two admissions share one watermark: H = 5 - 4 = 1 >= h.
        result = lrs.solve_and_extract(problem(
            [waiting("w1", 1, 1, 2, beta=1.0), waiting("w2", 2, 1, 2, beta=1.0)],
            b=2, c=1, s=2, n=2, free=5, h=1))
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(actions(result.plan),
                         {"w1": "prefill:1", "w2": "prefill:1"})
        self.assertEqual(result.plan.free_blocks_before_decodes, 1)

    def test_fixed_admission_charge_despite_short_chunk(self):
        # ABSTRACT: chunk is 1 token (token budget), charge stays 3 blocks.
        result = lrs.solve_and_extract(problem(
            [waiting("w", 1, 4, 3, beta=1.0)], b=1, c=4, s=1, n=1, free=3,
            h=0))
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(actions(result.plan), {"w": "prefill:1"})
        self.assertEqual(result.plan.free_blocks_before_decodes, 0)
        short = lrs.solve_and_extract(problem(
            [waiting("w", 1, 4, 3, beta=1.0)], b=1, c=4, s=1, n=1, free=2,
            h=0))
        self.assertIsInstance(short, lrs.Failure)
        self.assertEqual(short.outcome, lrs.OUTCOME_LP_INFEASIBLE)

    def test_omitted_resident_counts_and_zero_recovery_replacement(self):
        # ABSTRACT: the omitted resident holds no blocks; preempting it only
        # releases a resident place.
        prob = problem(
            [decoder("r_exec", 1, 0, 0, alpha=1.0, preemptible=False,
                     service=0),
             decoder("r_omit", 2, 0, 0, alpha=0.0, gamma=0.5, service=2),
             waiting("w", 3, 1, 1, beta=1.0, service=1)],
            b=3, c=1, s=3, n=2, free=5, h=0)
        _, _, plan = self.extract(prob, {
            "r_exec": {"y": 1.0}, "w": {"x": 1.0, "I": 1.0},
            "r_omit": {"z": 1.0}})
        self.assertEqual(actions(plan), {
            "r_exec": "decode", "w": "prefill:1", "r_omit": "preempt"})
        self.assertEqual(plan.resident_count_after, 2)
        self.assertEqual(plan.free_blocks_before_decodes, 4)

        # Without the omitted resident's preemption the admission cannot fit.
        prob_locked = problem(
            [decoder("r_exec", 1, 0, 0, alpha=1.0, preemptible=False),
             decoder("r_omit", 2, 0, 0, preemptible=False),
             waiting("w", 3, 1, 1, beta=1.0)],
            b=3, c=1, s=3, n=2, free=5, h=0)
        exists_choice = (None, None, ("prefill", 1))
        legal, _, _ = oracle_simulate(canon(self, prob_locked), exists_choice)
        self.assertFalse(legal)

    # -- decode demand and free blocks --------------------------------------

    def test_zero_free_blocks_zero_demand_decode(self):
        # ABSTRACT: a zero-demand decode still needs one free block.
        alone = lrs.solve_and_extract(problem(
            [decoder("d", 1, 0, 3, alpha=1.0)], b=1, c=1, s=1, n=1, free=0,
            h=0))
        self.assertIsInstance(alone, lrs.Failure)
        self.assertEqual(alone.outcome, lrs.OUTCOME_LP_INFEASIBLE)
        self.assertFalse(oracle_best(canon(self, problem(
            [decoder("d", 1, 0, 3, alpha=1.0)], b=1, c=1, s=1, n=1, free=0,
            h=0)))[0])
        # RUNTIME (pool 4, h = 0): another resident can be preempted.
        prob = problem(
            [decoder("d", 1, 0, 3, alpha=5.0, gamma=1.0),
             decoder("v", 2, 0, 1, alpha=0.0, gamma=1.0)],
            b=1, c=1, s=1, n=2, free=0, h=0)
        self.assertTrue(runtime_pool_consistent(prob))
        result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(actions(result.plan), {"d": "decode", "v": "preempt"})

    def test_one_free_block_demand_one_then_zero(self):
        # ABSTRACT: A consumes the only block; B needs none and runs first.
        result = lrs.solve_and_extract(problem(
            [decoder("A", 1, 1, 0, alpha=1.0, preemptible=False, service=0),
             decoder("B", 2, 0, 0, alpha=1.0, preemptible=False, service=1)],
            b=2, c=1, s=2, n=2, free=1, h=0))
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(actions(result.plan), {"A": "decode", "B": "decode"})
        self.assertEqual(result.plan.execution_order, ("B", "A"))
        self.assertEqual(result.plan.decode_demand, 1)
        self.assertEqual(result.plan.free_blocks_before_decodes, 1)

    def test_free_blocks_versus_one_block_decodes(self):
        specs = [decoder("O1", 1, 1, 0, alpha=1.0, preemptible=False),
                 decoder("O2", 2, 1, 0, alpha=1.0, preemptible=False)]
        # ABSTRACT, F = 1: only one of two one-block decodes fits.
        _, _, plan = self.extract(
            problem(specs, b=2, c=1, s=2, n=2, free=1, h=0),
            {"O1": {"y": 0.5}, "O2": {"y": 0.5}})
        self.assertEqual(actions(plan), {"O1": "decode"})
        self.assertEqual(plan.extraction.failed_trial_count, 1)
        # ABSTRACT, F = 2: both fit (unique LP optimum).
        result = lrs.solve_and_extract(
            problem(specs, b=2, c=1, s=2, n=2, free=2, h=0))
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(actions(result.plan), {"O1": "decode", "O2": "decode"})

    def test_mixed_decode_demand_groups(self):
        # ABSTRACT: F = 1, two zero-demand and two one-block decodes.
        prob = problem(
            [decoder("O1", 1, 1, 0, alpha=1.0, preemptible=False, service=0),
             decoder("Z1", 2, 0, 0, alpha=1.0, preemptible=False, service=1),
             decoder("Z2", 3, 0, 0, alpha=1.0, preemptible=False, service=2),
             decoder("O2", 4, 1, 0, alpha=1.0, preemptible=False, service=3)],
            b=4, c=1, s=4, n=4, free=1, h=0)
        _, _, plan = self.extract(prob, {
            "O1": {"y": 0.5}, "O2": {"y": 0.5}, "Z1": {"y": 1.0},
            "Z2": {"y": 1.0}})
        self.assertEqual(set(actions(plan)), {"Z1", "Z2", "O1"})
        self.assertEqual(plan.execution_order, ("Z1", "Z2", "O1"))
        self.assertEqual(plan.extraction.candidate_order,
                         ("Z1", "Z2", "O1", "O2"))

    # -- fractional structure and integral-idle victims ---------------------

    def test_nine_waiting_watermark_construction(self):
        prob = nine_waiting_watermark()
        self.assertTrue(runtime_pool_consistent(prob), RUNTIME)
        canonical, relaxed, plan = self.extract(
            prob, {f"w{j}": {"x": 0.2, "I": 0.2} for j in range(1, 10)})
        self.assertAlmostEqual(relaxed.normalized_objective, 1.8)
        self.assertEqual(plan.extraction.fractional_request_count, 9)
        self.assertEqual(actions(plan), {"w1": "prefill:1", "b": "decode"})
        self.assertEqual(plan.extraction.declined_optional_trial_count, 8)
        self.assertEqual(oracle_best(canonical), (True, 1.0))

        # Executed observation: the solver's I values equal 2/(n+1), the
        # unique optimal I (y_b is not unique, so it is not asserted).
        for n_waiting in (2, 5, 9):
            with self.subTest(n=n_waiting):
                result = lrs.solve_and_extract(nine_waiting_watermark(n_waiting))
                self.assertIsInstance(result, lrs.SchedulingSuccess, result)
                self.assertAlmostEqual(result.relaxed.normalized_objective,
                                       2 * n_waiting / (n_waiting + 1),
                                       places=6)
                for d in result.relaxed.decisions:
                    if d.request_id != "b":
                        self.assertAlmostEqual(d.prefill_indicator,
                                               2 / (n_waiting + 1), places=6)
                        self.assertAlmostEqual(d.x, d.prefill_indicator,
                                               places=6)
                acts = actions(result.plan)
                self.assertEqual(acts.pop("b"), "decode")
                self.assertEqual(list(acts.values()), ["prefill:1"])

    def test_integral_idle_victim_and_zero_support_candidate(self):
        # ABSTRACT: pool 200 and h = 2 are consistent, but the background
        # resident is preemptible with no eligible execution action, a shape
        # the live mapper never produces. It is integral idle in the LP.
        prob = problem(
            [waiting("w1", 1, 1, 2, beta=1.0), waiting("w2", 2, 1, 2, beta=1.0),
             idle("bg", 3, 197, gamma=1000.0)],
            b=2, c=1, s=2, n=3, free=3, h=2)
        self.assertFalse(runtime_pool_consistent(prob), ABSTRACT)
        canonical, relaxed, plan = self.extract(prob, {
            "w1": {"x": 0.5, "I": 0.5}, "w2": {"x": 0.5, "I": 0.5}})
        solved = lrs.solve_relaxation(canonical)
        self.assertAlmostEqual(-solved.recomputed_objective_min,
                               relaxed.normalized_objective, places=7)
        self.assertEqual(actions(plan), {
            "w1": "prefill:1", "w2": "prefill:1", "bg": "preempt"})
        self.assertEqual(plan.objective, 2.0 - 1000.0)
        self.assertEqual(oracle_best(canonical), (True, -998.0))

    # -- negative utility -----------------------------------------------------

    def test_necessary_negative_utility_first_work(self):
        # ABSTRACT (B holds no blocks). The only useful plan preempts A.
        prob = problem(
            [idle("A", 1, 1, gamma=5.0), decoder("B", 2, 0, 0, alpha=1.0,
                                                 preemptible=False)],
            b=1, c=1, s=1, n=2, free=0, h=0)
        result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(actions(result.plan), {"A": "preempt", "B": "decode"})
        self.assertEqual(result.plan.objective, -4.0)
        self.assertEqual(oracle_best(canon(self, prob)), (True, -4.0))

    def test_avoidable_negative_first_selection(self):
        prob = avoidable_negative_first()
        self.assertTrue(runtime_pool_consistent(prob), RUNTIME)
        point = {"B": {"y": 1.0}, "A": {"z": 0.05}}
        canonical, relaxed, plan = self.extract(prob, point)
        # The supplied point attains the LP optimum (analytically unique).
        solved = lrs.solve_relaxation(canonical)
        self.assertEqual(solved.category, lrs.CATEGORY_OPTIMAL)
        self.assertAlmostEqual(-solved.recomputed_objective_min, 4.6, places=7)
        self.assertAlmostEqual(relaxed.normalized_objective, 4.6, places=12)
        self.assertEqual(actions(plan), {"B": "decode", "A": "preempt"})
        self.assertEqual(plan.objective, -3.0)
        # C alone is legal with utility 0, the integer optimum.
        c_alone = tuple(("prefill", 1) if r.request_id == "C" else None
                        for r in canonical.requests)
        self.assertEqual(oracle_simulate(canonical, c_alone), (True, True, 0.0))
        self.assertEqual(oracle_best(canonical), (True, 0.0))
        # End to end, the unique LP optimum gives the same accepted limitation.
        result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(result.plan.objective, -3.0)

    # -- no legal useful plan --------------------------------------------------

    def test_lp_feasible_integer_infeasible_abstract_fixture(self):
        prob = abstract_lp_feasible_integer_infeasible()
        self.assertFalse(runtime_pool_consistent(prob), ABSTRACT)
        canonical = canon(self, prob)
        relaxed = lrs.validate_relaxed_solution(canonical, vector(
            canonical, {"w1": {"x": 0.5, "I": 0.5}, "w2": {"x": 0.5, "I": 0.5}}))
        self.assertIsInstance(relaxed, lrs.RelaxedSolution)
        failure = lrs.extract_integer_plan(canonical, relaxed)
        self.assertIsInstance(failure, lrs.Failure)
        self.assertEqual(failure.outcome, lrs.OUTCOME_NO_USEFUL_WORK)
        self.assertEqual(failure.extraction.candidate_order, ("w1", "w2"))
        self.assertEqual(failure.extraction.failed_trial_count, 2)
        self.assertFalse(oracle_best(canonical)[0])

        result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.Failure)
        self.assertEqual(result.stage, lrs.STAGE_EXTRACTION)
        self.assertEqual(result.outcome, lrs.OUTCOME_NO_USEFUL_WORK)
        self.assertEqual(result.solver.diagnostics.raw_status, 0)
        self.assertFalse(hasattr(result, "plan"))

    # -- construction mechanics ------------------------------------------------

    def test_self_victim_exclusion(self):
        # ABSTRACT: d is first in victim order but cannot be its own victim.
        prob = problem(
            [decoder("d", 1, 1, 2, alpha=1.0, service=0, victim=0),
             decoder("e", 2, 0, 0, alpha=1.0, preemptible=False, service=1),
             idle("v", 3, 1, victim=1)],
            b=2, c=1, s=2, n=3, free=0, h=0)
        _, _, plan = self.extract(prob, {
            "d": {"y": 0.5, "z": 0.5}, "e": {"y": 0.5}})
        self.assertEqual(plan.extraction.victim_candidate_order,
                         ("d", "v"))
        self.assertEqual(actions(plan), {
            "d": "decode", "e": "decode", "v": "preempt"})

    def test_failed_trial_cleanup_and_preservation(self):
        # ABSTRACT: W fails even with every victim; its tentative victims are
        # discarded and are later decoded.
        prob = problem(
            [decoder("d1", 1, 0, 0, alpha=1.0, preemptible=False, service=0),
             waiting("W", 2, 1, 100, beta=5.0, service=1),
             decoder("r1", 3, 0, 1, service=2),
             decoder("r2", 4, 0, 1, service=3)],
            b=3, c=1, s=3, n=4, free=1, h=0)
        _, _, plan = self.extract(prob, {"d1": {"y": 1.0}})
        self.assertEqual(actions(plan), {
            "d1": "decode", "r1": "decode", "r2": "decode"})
        self.assertEqual(plan.extraction.failed_trial_count, 1)
        self.assertEqual(plan.extraction.accepted_order, ("d1", "r1", "r2"))

    def test_zero_coefficient_mass_preemption_rejected(self):
        prob = mass_preemption(positive=False)
        self.assertTrue(runtime_pool_consistent(prob), RUNTIME)
        canonical, relaxed, plan = self.extract(prob, MASS_POINT)
        self.assertEqual(relaxed.normalized_objective, 100.0)
        solved = lrs.solve_relaxation(canonical)
        self.assertAlmostEqual(-solved.recomputed_objective_min, 100.0,
                               places=7)
        self.assertEqual(actions(plan), {
            "B": "decode", "A": "preempt", "R000": "decode"})
        self.assertEqual(plan.extraction.declined_optional_trial_count, 1)
        self.assertEqual(plan.extraction.accepted_order, ("B", "R000"))
        self.assertEqual(plan.victim_order, ("A",))

    def test_positive_coefficient_mass_preemption_accepted_limitation(self):
        prob = mass_preemption(positive=True)
        self.assertTrue(runtime_pool_consistent(prob), RUNTIME)
        canonical, relaxed, plan = self.extract(prob, MASS_POINT)
        self.assertAlmostEqual(relaxed.normalized_objective, 99.95, places=12)
        solved = lrs.solve_relaxation(canonical)
        self.assertAlmostEqual(-solved.recomputed_objective_min, 99.95,
                               places=7)
        self.assertEqual(sum(d.preempt for d in plan.decisions), 101)
        self.assertEqual(actions(plan)["W"], "prefill:1")
        self.assertEqual(actions(plan)["B"], "decode")
        self.assertAlmostEqual(plan.objective, 99.5, places=9)
        self.assertEqual(plan.free_blocks_before_decodes, 1)
        self.assertEqual(plan.resident_count_after, 2)
        # A cheaper legal plan exists: decode B after one small victim.
        cheap = tuple(
            ("decode",) if r.request_id == "B" else
            ("preempt",) if r.request_id == "R000" else None
            for r in canonical.requests)
        legal, useful, util = oracle_simulate(canonical, cheap)
        self.assertTrue(legal and useful)
        self.assertAlmostEqual(util, 99.9, places=9)

    def test_chunk_floor_and_minimum_one(self):
        # ABSTRACT: chunk = floor(1.7) = 1; the decode then uses the last token.
        prob = problem(
            [resident_prefill("p", 1, 3, 0, beta=1.0, preemptible=False),
             decoder("d", 2, 0, 0, alpha=1.0, preemptible=False)],
            b=2, c=3, s=2, n=2, free=1, h=0)
        _, _, plan = self.extract(prob, {
            "p": {"x": 1.7, "I": 1.0}, "d": {"y": 0.3}})
        self.assertEqual(actions(plan), {"p": "prefill:1", "d": "decode"})
        # Minimum one token for a small and for a zero-support prefill.
        for x_value in (0.3, 0.0):
            with self.subTest(x=x_value):
                prob = problem(
                    [decoder("d", 1, 0, 0, alpha=1.0, preemptible=False),
                     resident_prefill("p", 2, 3, 0, beta=1.0,
                                      preemptible=False)],
                    b=2, c=3, s=2, n=2, free=1, h=0)
                values = {"d": {"y": 1.0}}
                if x_value:
                    values["p"] = {"x": x_value, "I": x_value}
                _, _, plan = self.extract(prob, values)
                self.assertEqual(actions(plan), {"d": "decode",
                                                 "p": "prefill:1"})

    def test_extension_in_service_rank_order(self):
        # ABSTRACT: p2 has the smaller service rank and is extended first.
        prob = problem(
            [resident_prefill("p1", 1, 3, 0, beta=1.0, preemptible=False,
                              service=1),
             resident_prefill("p2", 2, 3, 0, beta=1.0, preemptible=False,
                              service=0)],
            b=5, c=3, s=2, n=2, free=0, h=0)
        _, _, plan = self.extract(prob, {
            "p1": {"x": 1.0, "I": 1.0}, "p2": {"x": 1.0, "I": 1.0}})
        self.assertEqual(actions(plan), {"p1": "prefill:2", "p2": "prefill:3"})
        self.assertEqual(plan.extraction.accepted_order, ("p2", "p1"))
        self.assertEqual(plan.token_use, 5)

    def test_strict_improvement_threshold(self):
        # ABSTRACT: after d1, d2 needs victim v (penalty 1.0).
        def run(alpha2):
            prob = problem(
                [decoder("d1", 1, 1, 0, alpha=10.0, preemptible=False),
                 decoder("d2", 2, 1, 0, alpha=alpha2, preemptible=False),
                 decoder("d3", 3, 0, 0, alpha=0.0, preemptible=False),
                 idle("v", 4, 1, gamma=1.0)],
                b=3, c=1, s=3, n=4, free=1, h=0)
            return self.extract(prob, {"d1": {"y": 1.0}})[2]

        policy = lrs.NumericalPolicy()
        for alpha2, accepted in ((1.0 + 3e-9, True), (1.0 + 1.5e-9, False),
                                 (1.0, False)):
            with self.subTest(alpha2=alpha2):
                gain = alpha2 - 1.0
                tau = policy.objective_abs_tol + policy.objective_rel_tol * max(
                    alpha2, 1.0)
                self.assertEqual(gain > tau, accepted)
                self.assertGreaterEqual(gain, 0.0)
                plan = run(alpha2)
                acts = actions(plan)
                if accepted:
                    self.assertEqual(acts, {"d1": "decode", "d2": "decode",
                                            "v": "preempt", "d3": "decode"})
                else:
                    # d2 declined; zero-gain d3 without new victims accepted.
                    self.assertEqual(acts, {"d1": "decode", "d3": "decode"})
                    self.assertEqual(
                        plan.extraction.declined_optional_trial_count, 1)

    def test_lp_values_primary_over_ranks_then_identity(self):
        # ABSTRACT.
        prob = problem(
            [decoder("A", 1, 0, 0, preemptible=False, service=5),
             decoder("B", 2, 0, 0, preemptible=False, service=0),
             decoder("C", 3, 0, 0, preemptible=False, service=1),
             decoder("D", 4, 0, 0, preemptible=False, service=1),
             idle("V1", 5, 0, victim=9), idle("V2", 6, 0, victim=0),
             idle("V3", 7, 0, victim=2), idle("V4", 8, 0, victim=2)],
            b=4, c=1, s=4, n=8, free=1, h=0)
        values = {"A": {"y": 0.5}, "B": {"y": 0.4}, "C": {"y": 0.5},
                  "D": {"y": 0.5}, "V1": {"z": 0.3}, "V2": {"z": 0.2},
                  "V3": {"z": 0.3}, "V4": {"z": 0.3}}
        _, relaxed, plan = self.extract(prob, values)
        self.assertEqual(plan.extraction.candidate_order, ("C", "D", "A", "B"))
        self.assertEqual(plan.extraction.victim_candidate_order,
                         ("V3", "V4", "V1", "V2"))
        canonical = canon(self, prob)
        again = lrs.extract_integer_plan(canonical, relaxed)
        self.assertEqual(again, plan)

    def test_exact_lp_comparison_precedes_rank(self):
        # ABSTRACT. Observed solver output for the symmetric h = 1 watermark
        # fixture differs by one unit in the last place. Exact comparison
        # (no tolerance comparator) orders by that difference; ranks break
        # only exactly equal values.
        specs = [waiting("w1", 1, 1, 1, beta=1.0), waiting("w2", 2, 1, 1, beta=1.0)]
        prob = problem(specs, b=2, c=1, s=2, n=2, free=2, h=1)
        low, high = 0.6666666666666666, 0.6666666666666667
        self.assertLess(low, high)
        for values, first in (
            ({"w1": {"x": low, "I": low}, "w2": {"x": high, "I": high}}, "w2"),
            ({"w1": {"x": low, "I": low}, "w2": {"x": low, "I": low}}, "w1"),
        ):
            with self.subTest(first=first):
                _, _, plan = self.extract(prob, values)
                self.assertEqual(plan.extraction.candidate_order[0], first)
                self.assertEqual(actions(plan), {first: "prefill:1"})

    def test_declared_orders_and_totals(self):
        # ABSTRACT: victims by victim rank; prefills, zero-demand decodes,
        # then one-block decodes, each by service rank.
        prob = problem(
            [idle("v1", 1, 3, victim=5), idle("v2", 2, 1, victim=1),
             decoder("z0", 3, 0, 0, alpha=1.0, preemptible=False, service=3),
             decoder("o1", 4, 1, 0, alpha=1.0, preemptible=False, service=0),
             resident_prefill("rp", 5, 3, 0, beta=1.0, preemptible=False,
                              service=4),
             waiting("w", 6, 2, 3, beta=1.0, service=2)],
            b=10, c=10, s=10, n=10, free=0, h=0)
        _, _, plan = self.extract(prob, {
            "v1": {"z": 1.0}, "v2": {"z": 1.0}, "z0": {"y": 1.0},
            "o1": {"y": 1.0}, "rp": {"x": 1.0, "I": 1.0},
            "w": {"x": 1.0, "I": 1.0}})
        self.assertEqual(plan.victim_order, ("v2", "v1"))
        self.assertEqual(plan.execution_order, ("w", "rp", "z0", "o1"))
        self.assertEqual(actions(plan), {
            "v1": "preempt", "v2": "preempt", "z0": "decode", "o1": "decode",
            "rp": "prefill:3", "w": "prefill:2"})
        self.assertEqual(
            (plan.token_use, plan.action_count, plan.resident_count_after,
             plan.free_blocks_before_decodes, plan.decode_demand),
            (7, 4, 4, 1, 1))
        self.assertEqual([d.request_id for d in plan.decisions],
                         ["v1", "v2", "z0", "o1", "rp", "w"])

    # -- numerical failures -----------------------------------------------------

    def test_nonfinite_aggregate_arithmetic(self):
        # Overflow of 1e308 times a two-token chunk after extension.
        prob = canon(self, problem(
            [resident_prefill("p", 1, 2, 0, beta=1e308, preemptible=False)],
            b=2, c=2, s=1, n=1, free=0, h=0))
        relaxed = lrs.validate_relaxed_solution(
            prob, vector(prob, {"p": {"x": 1.5, "I": 1.0}}))
        self.assertIsInstance(relaxed, lrs.RelaxedSolution)
        failure = lrs.extract_integer_plan(prob, relaxed)
        self.assertEqual((failure.stage, failure.outcome, failure.category),
                         (lrs.STAGE_EXTRACTION, lrs.OUTCOME_NUMERICAL,
                          lrs.CATEGORY_NONFINITE))

        # Overflowing penalty sum of two tentative victims.
        prob = canon(self, problem(
            [decoder("d1", 1, 1, 0, alpha=1.0, preemptible=False),
             decoder("d2", 2, 1, 0, alpha=1.0, preemptible=False),
             idle("v0", 3, 0, gamma=1e308), idle("v1", 4, 1, gamma=1e308)],
            b=2, c=1, s=2, n=4, free=1, h=0))
        relaxed = lrs.validate_relaxed_solution(
            prob, vector(prob, {"d1": {"y": 1.0}}))
        failure = lrs.extract_integer_plan(prob, relaxed)
        self.assertEqual(failure.outcome, lrs.OUTCOME_NUMERICAL)
        self.assertIn("trial penalty", failure.reason)

        # Relaxed residual overflow is a numerical failure, not an exception.
        big = 10 ** 308
        prob = canon(self, problem(
            [resident_prefill("p1", 1, big, 0, preemptible=False),
             resident_prefill("p2", 2, big, 0, preemptible=False)],
            b=big, c=big, s=2, n=2, free=0, h=0))
        failure = lrs.validate_relaxed_solution(prob, vector(prob, {
            "p1": {"x": 1.5e308, "I": 1.0}, "p2": {"x": 1.5e308, "I": 1.0}}))
        self.assertEqual((failure.category, failure.outcome),
                         (lrs.CATEGORY_NONFINITE, lrs.OUTCOME_NUMERICAL))

        # LP construction overflow (S_max * charge): solver not called.
        prob = problem([waiting("w", 1, 1, big, beta=1.0)], b=1, c=1, s=2,
                       n=1, free=big, h=0)
        with mock.patch.object(lrs, "linprog") as patched:
            failure = lrs.solve_and_extract(prob)
        patched.assert_not_called()
        self.assertEqual((failure.stage, failure.outcome, failure.category),
                         (lrs.STAGE_SOLVER, lrs.OUTCOME_NUMERICAL,
                          lrs.CATEGORY_NONFINITE))

        # Objective recomputation overflow in solver-result normalization.
        fake = types.SimpleNamespace(status=0, success=True, x=np.array([2.0, 2.0]),
                                     fun=-1.0, message="ok", nit=1)
        result = lrs._normalize_scipy_result(
            fake, np.array([-1e308, 1e308]), lrs.NumericalPolicy())
        self.assertEqual(result.category, lrs.CATEGORY_NONFINITE)

        # Large finite coefficients end to end return a structured result.
        result = lrs.solve_and_extract(problem(
            [resident_prefill("p", 1, 2, 0, beta=1e308, preemptible=False)],
            b=2, c=2, s=1, n=1, free=0, h=0))
        self.assertIsInstance(result, (lrs.Failure, lrs.SchedulingSuccess))
        if isinstance(result, lrs.Failure):
            self.assertNotEqual(result.outcome, lrs.OUTCOME_NO_USEFUL_WORK)

    # -- solver outcomes and diagnostics ---------------------------------------

    def test_solver_reported_infeasibility_real_status_2(self):
        # ABSTRACT: an admission needs 3 blocks and only 2 are free.
        result = lrs.solve_and_extract(problem(
            [waiting("w", 1, 1, 3, beta=1.0)], b=1, c=1, s=1, n=1, free=2,
            h=0))
        self.assertIsInstance(result, lrs.Failure)
        self.assertEqual(result.stage, lrs.STAGE_SOLVER)
        self.assertEqual(result.outcome, lrs.OUTCOME_LP_INFEASIBLE)
        self.assertEqual(result.category, lrs.CATEGORY_LP_INFEASIBLE)
        self.assertEqual(result.solver.diagnostics.raw_status, 2)
        self.assertTrue(result.solver.diagnostics.message)
        self.assertFalse(hasattr(result, "plan"))

    def test_solver_outcome_distinctions_with_fake_results(self):
        prob = self.row_fixture()
        good = canon(self, prob)
        expected_len = 4 * len(good.requests)

        def fake(status, success=False, x=None, fun=None, message="m"):
            return types.SimpleNamespace(status=status, success=success, x=x,
                                         fun=fun, message=message, nit=3)

        cases = [
            (fake(1, message="limit"), "solver_limit", lrs.OUTCOME_NUMERICAL),
            (fake(2, message="infeasible"), lrs.CATEGORY_LP_INFEASIBLE,
             lrs.OUTCOME_LP_INFEASIBLE),
            (fake(3), "unbounded", lrs.OUTCOME_NUMERICAL),
            (fake(4), "numerical_difficulty", lrs.OUTCOME_NUMERICAL),
            (fake(9), "malformed_output", lrs.OUTCOME_NUMERICAL),
            (fake(0, success=False, x=np.zeros(expected_len), fun=0.0),
             "malformed_output", lrs.OUTCOME_NUMERICAL),
            (fake(0, success=True, x=None, fun=0.0), "malformed_output",
             lrs.OUTCOME_NUMERICAL),
            (fake(0, success=True, x=np.zeros(expected_len), fun=5.0),
             "malformed_output", lrs.OUTCOME_NUMERICAL),
        ]
        for raw, category, outcome in cases:
            with self.subTest(category=category, status=raw.status):
                with mock.patch.object(lrs, "linprog", return_value=raw):
                    result = lrs.solve_and_extract(prob)
                self.assertIsInstance(result, lrs.Failure)
                self.assertEqual((result.category, result.outcome),
                                 (category, outcome))
                self.assertEqual(result.solver.diagnostics.raw_status,
                                 raw.status)
                self.assertEqual(result.solver.diagnostics.message,
                                 raw.message)
                self.assertEqual(result.solver.diagnostics.iterations, 3)
        with mock.patch.object(lrs, "linprog",
                               side_effect=RuntimeError("boom")):
            result = lrs.solve_and_extract(prob)
        self.assertEqual((result.category, result.outcome),
                         ("exception", lrs.OUTCOME_NUMERICAL))
        # A status-0 vector failing relaxed validation keeps diagnostics.
        with mock.patch.object(lrs, "linprog", return_value=fake(
                0, success=True, x=np.zeros(expected_len), fun=0.0)):
            result = lrs.solve_and_extract(prob)
        self.assertEqual((result.stage, result.outcome),
                         (lrs.STAGE_RELAXED, lrs.OUTCOME_NUMERICAL))
        self.assertIn("useful_execution", result.reason)
        self.assertEqual(result.solver.diagnostics.raw_status, 0)

    def test_single_solver_invocation_on_success(self):
        prob = mass_preemption(positive=True)
        with mock.patch.object(lrs, "linprog", wraps=lrs.linprog) as spy:
            result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.SchedulingSuccess, result)
        self.assertEqual(spy.call_count, 1)
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs["method"], "highs-ds")
        self.assertEqual(kwargs["options"], {"presolve": True})
        self.assertEqual(set(kwargs), {"A_ub", "b_ub", "bounds", "method",
                                       "options"})

    def test_final_plan_rejection_is_not_impossibility(self):
        prob = mass_preemption(positive=True)
        canonical, _, plan = self.extract(prob, MASS_POINT)
        zero = tuple(dataclasses.replace(d, prefill_tokens=0, decode=0,
                                         preempt=0, prefill_indicator=0)
                     for d in plan.decisions)
        preempt_only = tuple(
            dataclasses.replace(d, prefill_tokens=0, decode=0,
                                prefill_indicator=0)
            for d in plan.decisions)
        over_chunk = tuple(
            dataclasses.replace(d, prefill_tokens=2) if d.request_id == "W"
            else d for d in plan.decisions)
        tampered = [
            ("totals", dataclasses.replace(plan, token_use=1)),
            ("victim order", dataclasses.replace(
                plan, victim_order=plan.victim_order[::-1])),
            ("execution order", dataclasses.replace(
                plan, execution_order=plan.execution_order[::-1])),
            ("objective", dataclasses.replace(plan, objective=0.0)),
            ("empty", dataclasses.replace(
                plan, decisions=zero, victim_order=(), execution_order=())),
            ("preempt only", dataclasses.replace(
                plan, decisions=preempt_only, execution_order=())),
            ("chunk bound", dataclasses.replace(plan, decisions=over_chunk)),
            ("problem id", dataclasses.replace(plan, problem_id="other")),
        ]
        for name, bad in tampered:
            with self.subTest(name):
                result = lrs.validate_integer_plan(canonical, bad)
                self.assertIsInstance(result, lrs.Failure)
                self.assertEqual(result.stage, lrs.STAGE_PLAN)
                self.assertEqual(result.outcome, lrs.OUTCOME_INVALID)

        # Memory violation: W admitted without its 100 small victims.
        partial = tuple(
            dataclasses.replace(d, preempt=0) if d.request_id.startswith("R")
            else d for d in plan.decisions)
        result = lrs.validate_integer_plan(canonical, dataclasses.replace(
            plan, decisions=partial))
        self.assertEqual(result.category, "constraint_violation")

        # Through the entry point, a rejected plan keeps solver diagnostics
        # and is not reported as no useful work.
        with mock.patch.object(lrs, "extract_integer_plan",
                               return_value=dataclasses.replace(
                                   plan, token_use=0)):
            result = lrs.solve_and_extract(prob)
        self.assertEqual((result.stage, result.outcome),
                         (lrs.STAGE_PLAN, lrs.OUTCOME_INVALID))
        self.assertEqual(result.solver.diagnostics.raw_status, 0)

    def test_oversized_vector_entry_is_structured_failure(self):
        prob = canon(self, self.row_fixture())
        raw = list(vector(prob, {"d1": {"y": 1.0}}))
        raw[0] = 10 ** 400
        result = lrs.validate_relaxed_solution(prob, raw)
        self.assertIsInstance(result, lrs.Failure)
        self.assertEqual((result.category, result.outcome),
                         ("malformed_vector", lrs.OUTCOME_NUMERICAL))

    def test_independent_oracle_rejects_malformed_decisions(self):
        prob = mass_preemption(positive=True)
        canonical, _, plan = self.extract(prob, MASS_POINT)
        self.assertEqual(len(oracle_choice(canonical, plan.decisions)),
                         len(canonical.requests))

        def tampered(rid, **kw):
            return tuple(dataclasses.replace(d, **kw) if d.request_id == rid
                         else d for d in plan.decisions)

        cases = {
            "conflicting decode and preempt": tampered("B", preempt=1),
            "conflicting prefill and decode": tampered("W", decode=1),
            "prefill beyond remainder": tampered("W", prefill_tokens=2),
            "noncanonical indicator": tampered("R000", prefill_indicator=1),
            "bool flag": tampered("B", decode=True),
            "decode not eligible": tampered("W", prefill_tokens=0,
                                            prefill_indicator=0, decode=1),
            "association": tuple(reversed(plan.decisions)),
            "missing decision": plan.decisions[:-1],
        }
        for name, decisions in cases.items():
            with self.subTest(name):
                with self.assertRaises(AssertionError):
                    oracle_choice(canonical, decisions)
                bad = lrs.validate_integer_plan(canonical, dataclasses.replace(
                    plan, decisions=decisions))
                self.assertIsInstance(bad, lrs.Failure)
                self.assertEqual(bad.outcome, lrs.OUTCOME_INVALID)

    # -- independent bounded enumeration ----------------------------------------

    def random_state(self, rng, index):
        specs, key = [], 0
        n_res, n_wait = rng.randint(0, 3), rng.randint(0, 3)
        if n_res + n_wait == 0:
            n_wait = 1
        c_max = rng.randint(1, 3)
        coef = (0.0, 0.5, 1.0, 2.0, 7.0)
        for _ in range(n_res):
            key += 1
            preemptible = rng.random() < 0.8
            blocks = rng.randint(0, 3)
            kind = rng.choice(("decode", "decode", "prefill", "idle"))
            service, victim = rng.randint(0, 2), rng.randint(0, 2)
            if kind == "decode":
                specs.append(decoder(
                    f"r{key}", key, rng.randint(0, 1), blocks,
                    alpha=rng.choice(coef), gamma=rng.choice(coef),
                    service=service, victim=victim, preemptible=preemptible))
            elif kind == "prefill":
                specs.append(resident_prefill(
                    f"r{key}", key, rng.randint(1, 3), blocks,
                    beta=rng.choice(coef), gamma=rng.choice(coef),
                    service=service, victim=victim, preemptible=preemptible))
            elif preemptible:
                specs.append(idle(f"r{key}", key, blocks,
                                  gamma=rng.choice(coef), victim=victim))
            else:
                specs.append(_spec(f"r{key}", key))
        for _ in range(n_wait):
            key += 1
            specs.append(waiting(f"w{key}", key, rng.randint(1, 3),
                                 rng.randint(0, 4), beta=rng.choice(coef),
                                 service=rng.randint(0, 2)))
        return problem(
            specs, b=rng.randint(1, 4), c=c_max, s=rng.randint(1, 3),
            n=rng.randint(max(1, n_res), n_res + 2), free=rng.randint(0, 4),
            h=rng.randint(0, 3), pid=f"random-{index}")

    def test_bounded_enumeration_against_constructor(self):
        # ABSTRACT random states (seeded). The oracle simulates native-order
        # allocation; it never calls the production legality validator.
        rng = random.Random(20261013)
        counts = {"success": 0, lrs.OUTCOME_LP_INFEASIBLE: 0,
                  lrs.OUTCOME_NO_USEFUL_WORK: 0, "guided_success": 0,
                  "guided_none": 0, "plans_enumerated": 0}
        for index in range(150):
            prob = canon(self, self.random_state(rng, index))
            n = len(prob.requests)
            c, a_ub, b_ub, _, labels = lrs._build_lp(prob)
            exists, best = False, None
            for choice in oracle_enumerate(prob):
                counts["plans_enumerated"] += 1
                legal, useful, util = oracle_simulate(prob, choice)
                v = choice_vector(prob, choice)
                res = lrs._residuals(prob, *lrs._split(v, n))
                lp_ok = all(val <= 1e-9 for val in res.values())
                matrix_ok = bool(np.all(a_ub @ np.array(v) - b_ub <= 1e-9))
                # Integer points satisfy every LP row exactly when legal+useful.
                self.assertEqual(lp_ok, legal and useful, (index, choice))
                self.assertEqual(matrix_ok, lp_ok, (index, choice))
                if legal and useful:
                    exists = True
                    best = util if best is None else max(best, util)

            result = lrs.solve_and_extract(prob)
            if isinstance(result, lrs.SchedulingSuccess):
                counts["success"] += 1
                self.assertTrue(exists, index)
                legal, useful, util = oracle_simulate(
                    prob, plan_choice(result.plan, prob))
                self.assertTrue(legal and useful, index)
                self.assertLessEqual(util, best + 1e-9)
                self.assertGreaterEqual(
                    result.relaxed.normalized_objective, best - 1e-6, index)
            else:
                self.assertIn(result.outcome, (lrs.OUTCOME_LP_INFEASIBLE,
                                               lrs.OUTCOME_NO_USEFUL_WORK),
                              (index, result))
                counts[result.outcome] += 1
                self.assertFalse(exists, index)

            # Progress does not depend on the guidance values: arbitrary
            # finite guidance (bypassing relaxed validation on purpose).
            guided = lrs.RelaxedSolution(
                decisions=tuple(
                    lrs.RelaxedDecision(
                        r.request_id, rng.uniform(0, r.prefill_upper_bound),
                        rng.random(), rng.random(), rng.random())
                    for r in prob.requests),
                raw_vector=(), projection_count=0, worst_raw_violation=0.0,
                worst_normalized_violation=0.0, normalized_objective=0.0)
            plan = lrs.extract_integer_plan(prob, guided)
            if isinstance(plan, lrs.IntegerPlan):
                counts["guided_success"] += 1
                self.assertTrue(exists, index)
                self.assertIs(lrs.validate_integer_plan(prob, plan), plan)
                legal, useful, _ = oracle_simulate(
                    prob, plan_choice(plan, prob))
                self.assertTrue(legal and useful, index)
            else:
                counts["guided_none"] += 1
                self.assertEqual(plan.outcome, lrs.OUTCOME_NO_USEFUL_WORK)
                self.assertFalse(exists, index)
        self.assertGreater(counts["success"], 0)
        self.assertGreater(counts[lrs.OUTCOME_LP_INFEASIBLE]
                           + counts[lrs.OUTCOME_NO_USEFUL_WORK], 0)
        print(f"\n[enumeration counts] {counts}", file=sys.stderr)

    def test_enumeration_on_named_fixtures(self):
        named = [
            ("abstract F=3 h=2", abstract_lp_feasible_integer_infeasible(),
             False),
            ("nine waiting", nine_waiting_watermark(), True),
            ("avoidable negative", avoidable_negative_first(), True),
        ]
        for name, prob, expect in named:
            with self.subTest(name):
                canonical = canon(self, prob)
                exists, _ = oracle_best(canonical)
                self.assertEqual(exists, expect)
                result = lrs.solve_and_extract(prob)
                self.assertEqual(
                    isinstance(result, lrs.SchedulingSuccess), expect)


if __name__ == "__main__":
    unittest.main()
