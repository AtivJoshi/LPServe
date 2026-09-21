import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lp_relaxation_scheduler as lrs  # noqa: E402

POLICY_ID = "fixture_supplied_charge_v1"


def req(request_id, key, rem, ub, flags, charges, utilities):
    prefill, decode, preempt = (f == "T" for f in flags.split("/"))
    fixed, decode_charge, recovery = charges
    alpha, beta, gamma = utilities
    return lrs.RequestInput(
        request_id=request_id, order_key=key,
        prompt_tokens_remaining=rem, prefill_upper_bound=ub,
        prefill_eligible=prefill, decode_eligible=decode,
        preemption_eligible=preempt,
        prefill_fixed_charge=fixed, decode_charge=decode_charge,
        preemption_recovery=recovery,
        decode_utility=alpha, prefill_token_utility=beta,
        preemption_penalty=gamma,
    )


def problem(requests, b, c, s, m_free, w, legal=(), problem_id="test"):
    return lrs.LPProblem(
        requests=tuple(requests), legal_preemption_ids=frozenset(legal),
        b_max=b, c_max=c, s_max=s, m_free=m_free, w=w,
        problem_id=problem_id, decode_memory_policy_id=POLICY_ID,
        numerical_policy=lrs.NumericalPolicy(),
    )


def decision(plan, request_id):
    return next(d for d in plan.decisions if d.request_id == request_id)


class LPRelaxationSchedulerTest(unittest.TestCase):
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
        self.assertEqual(decision(result.plan, "smoke-prefill").prefill_tokens, 2)
        self.assertEqual(decision(result.plan, "smoke-decode").decode, 1)

    def test_mixed_case_and_tied_extraction(self):
        victims = dict(rem=0, ub=0, flags="F/T/T", charges=(0, 0, 2),
                       utilities=(0, 0, 0.5))
        prob = problem(
            [
                req("admit", (0, 10), 2, 2, "T/F/F", (2, 0, 0), (0, 4, 0)),
                req("decode", (0, 20), 0, 0, "F/T/F", (0, 0, 0), (3, 0, 0)),
                req("victim-a", (0, 30), **victims),
                req("victim-b", (0, 40), **victims),
            ],
            b=4, c=2, s=3, m_free=0, w=0, legal={"victim-a", "victim-b"},
        )
        result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.SchedulingSuccess)
        plan = result.plan
        self.assertIs(lrs.validate_integer_plan(
            lrs.validate_problem(prob), plan), plan)
        self.assertAlmostEqual(result.relaxed.normalized_objective, 10.5)
        self.assertGreater(decision(plan, "admit").prefill_tokens, 0)
        self.assertEqual(decision(plan, "decode").decode, 1)
        self.assertTrue(
            any(d.preempt for d in plan.decisions if d.request_id.startswith("victim"))
        )
        self.assertEqual(
            [d.request_id for d in plan.decisions],
            ["admit", "decode", "victim-a", "victim-b"],
        )

        # Tied relaxed point: layout is [x.., y.., I.., z..] in order_key order.
        canonical = lrs.validate_problem(prob)
        tied = (2, 0, 0, 0, 0, 1, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0.5, 0.5)
        relaxed = lrs.validate_relaxed_solution(canonical, tied)
        self.assertIsInstance(relaxed, lrs.RelaxedSolution)
        self.assertEqual(relaxed.projection_count, 0)
        extracted = lrs.extract_integer_plan(canonical, relaxed)
        self.assertIsInstance(extracted, lrs.IntegerPlan)
        self.assertEqual(extracted.fractional_request_count, 2)
        self.assertEqual(extracted.dominant_preemption_ids, ())
        self.assertEqual(extracted.safety_preemption_ids, ("victim-a",))
        self.assertEqual(decision(extracted, "victim-a").preempt, 1)
        self.assertEqual(decision(extracted, "victim-b").preempt, 0)
        self.assertEqual(decision(extracted, "victim-b").decode, 1)
        self.assertIs(lrs.validate_integer_plan(canonical, extracted), extracted)

    def test_fractional_prefill(self):
        prob = problem(
            [
                req("decode", (0, 10), 0, 0, "F/T/F", (0, 2, 0), (4, 0, 0)),
                req("prefill", (0, 20), 3, 3, "T/F/F", (0, 0, 0), (0, 1, 0)),
            ],
            b=2, c=3, s=2, m_free=1, w=0,
        )
        result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.SchedulingSuccess)
        relaxed = {d.request_id: d for d in result.relaxed.decisions}
        self.assertAlmostEqual(relaxed["decode"].y, 0.5, places=6)
        self.assertAlmostEqual(relaxed["prefill"].x, 1.5, places=6)
        plan = result.plan
        self.assertTrue(any(d.prefill_tokens > 0 or d.decode for d in plan.decisions))
        self.assertGreaterEqual(plan.residual_token_capacity, 0)
        self.assertGreaterEqual(plan.residual_action_capacity, 0)
        self.assertGreaterEqual(plan.residual_memory_capacity, 0)
        self.assertIs(lrs.validate_integer_plan(
            lrs.validate_problem(prob), plan), plan)

    def test_zero_zero_execution_tie_is_no_action(self):
        # A legal-preemption request with zero recovery is never preempted
        # (c^Z must be positive), so its y=I=0, z=0.5 relaxed point must
        # yield no execution action rather than a decode it is not eligible for.
        prob = problem(
            [
                req("locked", (0, 10), 1, 1, "T/F/F", (2, 0, 0), (0, 1, 0)),
                req("cand", (0, 20), 1, 1, "T/F/T", (1, 0, 0), (0, 0, 0)),
            ],
            b=2, c=1, s=2, m_free=2, w=0, legal={"cand"},
        )
        canonical = lrs.validate_problem(prob)
        self.assertIsInstance(canonical, lrs.LPProblem)
        # Layout is [x.., y.., I.., z..] in order_key order.
        point = (1, 0, 0, 0, 1, 0, 0, 0.5)
        relaxed = lrs.validate_relaxed_solution(canonical, point)
        self.assertIsInstance(relaxed, lrs.RelaxedSolution)
        plan = lrs.extract_integer_plan(canonical, relaxed)
        self.assertIsInstance(plan, lrs.IntegerPlan)
        self.assertEqual(plan.fractional_request_count, 1)
        self.assertEqual(plan.dominant_preemption_ids, ())
        self.assertEqual(plan.safety_preemption_ids, ())
        self.assertEqual(decision(plan, "locked").prefill_tokens, 1)
        other = decision(plan, "cand")
        self.assertEqual(
            (other.prefill_tokens, other.decode, other.preempt,
             other.prefill_indicator),
            (0, 0, 0, 0),
        )
        self.assertIs(lrs.validate_integer_plan(canonical, plan), plan)

    def test_visible_infeasibility(self):
        prob = problem(
            [req("only", (0, 10), 1, 1, "T/F/F", (1, 0, 0), (0, 1, 0))],
            b=1, c=1, s=1, m_free=0, w=1,
        )
        self.assertIsInstance(lrs.validate_problem(prob), lrs.LPProblem)
        result = lrs.solve_and_extract(prob)
        self.assertIsInstance(result, lrs.Failure)
        self.assertNotIsInstance(result, lrs.SchedulingSuccess)
        self.assertEqual(result.stage, "solver")
        self.assertEqual(result.category, "infeasible")
        self.assertEqual(result.solver.diagnostics.raw_status, 2)
        self.assertIsNone(result.solver.raw_vector)
        self.assertFalse(hasattr(result, "plan"))


if __name__ == "__main__":
    unittest.main()
