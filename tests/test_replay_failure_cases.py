"""CPU checks for the failure-case replay (``scripts/replay_failure_cases.py``).

Uses only the LP and integer layers (numpy and scipy) and the retained
``fractional_admission`` diagnostic in the running-order evidence; no engine,
scheduler, or GPU. Expected values are the documented ones in the comparison
handoff (relaxed 0.59375, whole-action plan 0.125, integer optimum 0.5).
"""

import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import replay_failure_cases as rfc  # noqa: E402

DIAGNOSTIC = next(e for e in rfc.EXAMPLES
                  if e[0] == "fractional_admission" and "#" in e[2])


class ReplayTest(unittest.TestCase):

    def setUp(self):
        self.fixture = rfc.diagnostic_fixture(*DIAGNOSTIC[:3])

    def test_output_actions(self):
        outputs = dict(scheduled=[[3, 8], [5, 0]], preempted_seq_ids=[7],
                       ignored_seq_ids=[])
        self.assertEqual(rfc.output_actions(outputs),
                         {"3": "prefill:8", "5": "decode", "7": "preempt"})

    def test_fractional_admission_reproduces(self):
        result = rfc.replay(self.fixture)
        self.assertEqual(result["mismatches"], [])
        self.assertTrue(result["round_trip_equal"])
        self.assertTrue(result["lp_plan_equal"])
        self.assertTrue(result["relaxed_vector_equal"])
        self.assertTrue(result["integer_equal"])
        self.assertEqual(result["relaxed"],
                         {"0": dict(x=0.9375, I=0.9375), "1": dict(x=1.0,
                                                                   I=1.0)})
        self.assertEqual(result["lp_summary"]["actions"], {"1": "prefill:1"})
        self.assertEqual(result["integer"]["actions"], {"0": "prefill:1"})
        comparison = result["comparison"]
        self.assertAlmostEqual(comparison["relaxation_gap"], 0.09375)
        self.assertAlmostEqual(comparison["extraction_loss"], 0.375)
        self.assertEqual(comparison["attribution"], "extraction_loss")

    def test_mismatches_are_reported(self):
        cases = dict(
            plan=lambda f: f["retained_lp"]["result"]["plan"].update(
                objective=0.5),
            integer=lambda f: f["retained_integer"].update(
                actions={"1": "prefill:1"}),
            executed=lambda f: f["executed"].update(scheduled=[[0, 1]]))
        for name, change in cases.items():
            with self.subTest(name):
                fixture = copy.deepcopy(self.fixture)
                change(fixture)
                self.assertNotEqual(rfc.replay(fixture)["mismatches"], [])

    def test_unretained_results_are_labelled(self):
        fixture = copy.deepcopy(self.fixture)
        fixture.update(retained_lp=None, retained_integer=None)
        result = rfc.replay(fixture)
        self.assertEqual(result["mismatches"], [])
        self.assertIn("offline only", result["lp_source"])
        self.assertIn("not retained", result["integer_source"])
        self.assertNotIn("lp_plan_equal", result)
        self.assertNotIn("integer_equal", result)

    def test_main_replays_fixture_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixtures = Path(tmp) / "fixtures"
            fixtures.mkdir()
            (fixtures / "a.json").write_text(rfc.json.dumps(self.fixture))
            with contextlib.redirect_stdout(io.StringIO()):
                status = rfc.main(["--out", tmp, "--fixtures", str(fixtures)])
            self.assertEqual(status, 0)
            report = (Path(tmp) / "report.txt").read_text()
            self.assertIn("RESULT: PASS", report)
            self.assertIn("extraction loss 0.375", report)


if __name__ == "__main__":
    unittest.main()
