"""CPU checks for the experimental integer optimization helper.

The independent oracle is the planner suite's bounded enumeration with
native-order simulation (``tests/test_lp_relaxation_scheduler.py``); it never
calls the production validators. Random states are that suite's seeded
ABSTRACT generator.
"""

import dataclasses
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

import exact_integer_plan as eip  # noqa: E402
import lp_relaxation_scheduler as lrs  # noqa: E402
import test_lp_relaxation_scheduler as pt  # noqa: E402


def fractional_admission():
    # RUNTIME shape (pool 16, h = 0): waiting A (charge 16) then B (charge
    # 1), chunk cap 1, coordinated coefficients 1 and 1/4 per token.
    return pt.problem(
        [pt.waiting("A", 0, 256, 16, beta=1.0, service=0),
         pt.waiting("B", 1, 1, 1, beta=0.25, service=1)],
        b=2, c=1, s=2, n=2, free=16, h=0, pid="fractional")


def mixed_ample():
    # RUNTIME shape: resident A (8 prompt tokens left, 1 block), later
    # resident B decode-ready with demand 0; budget 8, two actions.
    return pt.problem(
        [pt.resident_prefill("A", 0, 8, 1, beta=1 / 8, gamma=1.0, service=0,
                             victim=1),
         pt.decoder("B", 1, 0, 1, alpha=0.25, gamma=0.25, service=1,
                    victim=0)],
        b=8, c=8, s=2, n=2, free=6, h=0, pid="mixed")


def _raw(status=0, x=None, fun=None, **extra):
    return SimpleNamespace(status=status, message=f"status {status}", x=x,
                           fun=fun, mip_dual_bound=extra.get("dual"),
                           mip_gap=extra.get("gap"),
                           mip_node_count=extra.get("nodes", 0))


class IntegerModelTest(unittest.TestCase):

    def test_integer_domains_and_shared_constraints(self):
        prob = pt.canon(self, mixed_ample())
        calls = []
        native = eip.milp

        def spy(c, **kwargs):
            calls.append((c, kwargs))
            return native(c, **kwargs)
        with mock.patch.object(lrs, "_build_lp", wraps=lrs._build_lp) as build, \
                mock.patch.object(eip, "milp", spy):
            result = eip.solve_integer_plan(prob)
        self.assertIsInstance(result, eip.IntegerOptimum, result)
        build.assert_called_once()
        c, rows, rhs, bounds, _ = lrs._build_lp(prob)
        (called_c, kwargs), = calls
        np.testing.assert_array_equal(called_c, c)
        np.testing.assert_array_equal(kwargs["integrality"], np.ones(c.size))
        np.testing.assert_array_equal(kwargs["constraints"].A, rows)
        np.testing.assert_array_equal(kwargs["constraints"].ub, rhs)
        self.assertTrue(np.all(kwargs["constraints"].lb == -np.inf))
        np.testing.assert_array_equal(kwargs["bounds"].lb,
                                      [b[0] for b in bounds])
        np.testing.assert_array_equal(
            kwargs["bounds"].ub,
            [np.inf if b[1] is None else b[1] for b in bounds])
        self.assertEqual(kwargs["options"], dict(eip.SOLVER_OPTIONS))
        self.assertEqual(dict(eip.SOLVER_OPTIONS),
                         dict(mip_rel_gap=0.0, presolve=True, time_limit=10.0))
        # Seven A tokens plus B's decode beat A's full chunk: 7/8 + 1/4 > 1.
        plan = result.plan
        self.assertEqual(eip.plan_actions(plan), {"A": "prefill:7",
                                                  "B": "decode"})
        self.assertAlmostEqual(plan.objective, 1.125)
        self.assertIs(lrs.validate_integer_plan(prob, plan), plan)
        for d in plan.decisions:
            self.assertTrue(all(type(v) is int for v in (
                d.prefill_tokens, d.decode, d.preempt, d.prefill_indicator)))
        source = plan.extraction
        self.assertIsInstance(source, lrs.DirectIntegerSolution)
        self.assertEqual((source.raw_status, source.conversion_tol,
                          source.options), (0, 1e-6, eip.SOLVER_OPTIONS))
        self.assertAlmostEqual(source.dual_bound, 1.125)

    def test_known_optima_match_bounded_enumeration(self):
        named = [fractional_admission(), mixed_ample(),
                 pt.abstract_lp_feasible_integer_infeasible(),
                 pt.avoidable_negative_first()]
        rng = random.Random(20261024)
        states = named + [pt.LPRelaxationSchedulerTest.random_state(
            None, rng, k) for k in range(150)]
        counts = dict(optimal=0, infeasible=0)
        for index, raw in enumerate(states):
            prob = pt.canon(self, raw)
            exists, best = pt.oracle_best(prob)
            result = eip.solve_integer_plan(prob)
            if not exists:
                self.assertIsInstance(result, eip.IntegerSolveFailure, index)
                self.assertEqual(result.category, "integer_infeasible", index)
                self.assertIsNone(result.incumbent_objective)
                counts["infeasible"] += 1
                continue
            self.assertIsInstance(result, eip.IntegerOptimum, (index, result))
            counts["optimal"] += 1
            legal, useful, util = pt.oracle_simulate(
                prob, pt.plan_choice(result.plan, prob))
            self.assertTrue(legal and useful, index)
            self.assertAlmostEqual(util, best, places=9, msg=index)
            self.assertAlmostEqual(result.plan.objective, best, places=9)
        self.assertGreater(counts["optimal"], 50)
        self.assertGreater(counts["infeasible"], 0)

    def test_fractional_admission_separates_relaxation_and_extraction(self):
        prob = pt.canon(self, fractional_admission())
        lp = lrs.solve_and_extract(prob)
        integer = eip.solve_integer_plan(prob)
        record = eip.compare_same_state(prob, lp, integer)
        # LP: 15/16 of A and all of B (1.1875); extraction tries B first and
        # cannot add A; the integer optimum admits A alone (1).
        self.assertAlmostEqual(record["lp_relaxed_objective"], 1.1875)
        self.assertEqual(record["lp_actions"], {"B": "prefill:1"})
        self.assertEqual(record["integer_actions"], {"A": "prefill:1"})
        self.assertAlmostEqual(record["relaxation_gap"], 0.1875)
        self.assertAlmostEqual(record["extraction_loss"], 0.75)
        self.assertEqual(record["attribution"], "extraction_loss")
        self.assertTrue(record["bounds_consistent"])

    def test_evaluate_assignment_reports_every_violation(self):
        prob = pt.canon(self, fractional_admission())
        both = eip.evaluate_assignment(prob, {"A": "prefill:2",
                                              "B": "prefill:1"})
        self.assertFalse(both["feasible"])
        self.assertEqual(both["domain_violations"],
                         ["A: prefill 2 outside [1, 1]"])
        self.assertIn("actual_memory", both["violated_rows"])
        self.assertIn("prefill_linkage_upper[A]", both["violated_rows"])
        self.assertEqual(eip.evaluate_assignment(prob, {})["violated_rows"],
                         ["useful_execution"])
        alone = eip.evaluate_assignment(prob, {"A": "prefill:1"})
        self.assertEqual((alone["feasible"], alone["objective"]), (True, 1.0))


class IntegerResultValidationTest(unittest.TestCase):
    """Non-optimal and malformed solver results never become plans."""

    def solve_with(self, raw, prob=None):
        prob = prob or pt.canon(self, mixed_ample())
        with mock.patch.object(eip, "milp", return_value=raw):
            return eip.solve_integer_plan(prob)

    def optimum_vector(self):
        prob = pt.canon(self, mixed_ample())
        return prob, list(eip.solve_integer_plan(prob).raw_vector)

    def test_time_limit_keeps_the_incumbent_as_a_diagnostic(self):
        prob, x = self.optimum_vector()
        result = self.solve_with(_raw(1, x, -1.125, dual=-1.2, gap=0.06,
                                      nodes=7), prob)
        self.assertIsInstance(result, eip.IntegerSolveFailure)
        self.assertEqual((result.stage, result.category),
                         (eip.STAGE_INTEGER_SOLVER, "solver_limit"))
        self.assertEqual(result.incumbent_vector, tuple(x))
        self.assertEqual((result.incumbent_objective, result.dual_bound,
                          result.reported_gap, result.node_count),
                         (1.125, 1.2, 0.06, 7))
        self.assertFalse(hasattr(result, "plan"))

    def test_other_statuses_and_exceptions(self):
        for status, category in ((2, "integer_infeasible"), (3, "unbounded"),
                                 (4, "solver_failure"),
                                 (None, "solver_failure")):
            with self.subTest(status=status):
                result = self.solve_with(_raw(status))
                self.assertIsInstance(result, eip.IntegerSolveFailure)
                self.assertEqual(result.category, category)
        with mock.patch.object(eip, "milp", side_effect=RuntimeError("boom")):
            result = eip.solve_integer_plan(pt.canon(self, mixed_ample()))
        self.assertEqual(result.category, "exception")

    def test_malformed_optimal_results(self):
        prob, x = self.optimum_vector()
        n = len(prob.requests)
        fractional = list(x)
        fractional[0] = 6.5  # A's token count
        bad_indicator = list(x)
        bad_indicator[2 * n] = 0.0  # A's indicator with x_A = 7
        infeasible = list(x)
        infeasible[0] = 8.0  # 8 + 1 tokens exceed b_max = 8
        cases = [
            ("missing vector", _raw(0, None, -1.125), "malformed_output"),
            ("nonfinite", _raw(0, [float("nan")] + x[1:], -1.125),
             "malformed_output"),
            ("wrong length", _raw(0, x[:-1], -1.125), "malformed_output"),
            ("objective disagrees", _raw(0, x, -2.0), "malformed_output"),
            ("fractional", _raw(0, fractional, -(6.5 / 8 + 0.25)),
             "fractional_solution"),
            ("indicator", _raw(0, bad_indicator, -1.125),
             "inconsistent_indicator"),
            ("row violation", _raw(0, infeasible, -1.25),
             "integer_infeasible_point"),
        ]
        for label, raw, category in cases:
            with self.subTest(label):
                result = self.solve_with(raw, prob)
                self.assertIsInstance(result, eip.IntegerSolveFailure)
                self.assertEqual(result.category, category, result.reason)
        # Near-integral values within the conversion tolerance are accepted.
        near = [v + 5e-7 if v else v for v in x]
        c = lrs._build_lp(prob)[0]
        result = self.solve_with(_raw(0, near, float(c @ np.array(near))),
                                 prob)
        self.assertIsInstance(result, eip.IntegerOptimum, result)
        self.assertEqual(eip.plan_actions(result.plan),
                         {"A": "prefill:7", "B": "decode"})

    def test_integer_plans_pass_the_shared_validator_only_when_complete(self):
        prob, _ = self.optimum_vector()
        plan = eip.solve_integer_plan(prob).plan
        tampered = dataclasses.replace(plan, token_use=plan.token_use + 1)
        self.assertEqual(lrs.validate_integer_plan(prob, tampered).category,
                         "total_mismatch")
        no_provenance = dataclasses.replace(plan, extraction=None)
        self.assertEqual(lrs.validate_integer_plan(prob, no_provenance)
                         .category, "malformed_plan")
        # The LP path still carries constructor diagnostics.
        lp = lrs.solve_and_extract(prob)
        self.assertIsInstance(lp.plan.extraction, lrs.ExtractionDiagnostics)


if __name__ == "__main__":
    unittest.main()
