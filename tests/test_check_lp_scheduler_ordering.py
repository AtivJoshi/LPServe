"""CPU checks of the GPU helpers' independently derived execution order.

Records come from the real mapper, solver, and executor on CPU (no engine,
model, or GPU) and are serialized with the helpers' own ``to_jsonable`` and
``outputs_record``. The derived order must be victims by victim rank, then
identity; then prefills, zero-demand decodes, and one-block-demand decodes,
each by service rank, then identity. Records in the superseded
identity-only order must fail.
"""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import check_lp_scheduler_contention_gpu as contention  # noqa: E402
import check_lp_scheduler_decode_preemption_reference_gpu as decode_ref  # noqa: E402
import check_lp_scheduler_preemption_gpu as preemption  # noqa: E402
from check_lp_scheduler_gpu import (  # noqa: E402
    CheckFailure, outputs_record, to_jsonable,
)

import lp_relaxation_scheduler as lrs  # noqa: E402
import lpserve_plan_execution as lpe  # noqa: E402
import test_lpserve_plan_execution as fx  # noqa: E402


def _records(holder, snapshot, result):
    """Execute and serialize as the helpers do."""
    outputs = lpe.execute_plan(holder, snapshot, result)
    assert not isinstance(outputs, lrs.Failure), outputs
    return (to_jsonable(snapshot), to_jsonable(result.plan),
            outputs_record(outputs))


def _one_free_block_case():
    """A (seq 0) needs one block, B (seq 1) needs none; one block is free.
    A has the smaller identity and the smaller service rank."""
    holder = fx._SchedulerHolder(num_gpu_blocks=3)
    fx._add_resident(holder, 0, 4, processed=4, num_output_tokens=1)  # gap 1
    fx._add_resident(holder, 1, 4, processed=4)  # gap 0
    utilities = fx._utilities(r0=(1.0, 0.0, 1.0), r1=(1.0, 0.0, 1.0))
    snapshot, result = fx._map_and_solve(holder, utilities)
    return holder, snapshot, result


def _service_rank_case():
    """Prefills whose service ranks (running 5, running 1, waiting 0)
    disagree with identity order (0, 1, 5)."""
    holder = fx._SchedulerHolder()
    fx._add_resident(holder, 5, 6, processed=4)
    fx._add_resident(holder, 1, 6, processed=4)
    fx._add_waiting(holder, 0, 2)
    utilities = fx._utilities(
        r0=(0.0, 1.0, 1.0), r1=(0.0, 1.0, 1.0), r5=(0.0, 1.0, 1.0))
    snapshot, result = fx._map_and_solve(holder, utilities)
    return holder, snapshot, result


class DerivedOrderTest(unittest.TestCase):
    def test_helpers_share_the_derived_order_check(self):
        self.assertIs(contention.check_declared_order,
                      preemption.check_declared_order)
        self.assertIs(decode_ref.preemption.check_declared_order,
                      preemption.check_declared_order)

    def test_zero_demand_decode_runs_before_one_block_decode(self):
        holder, snapshot, result = _one_free_block_case()
        snap, plan, emitted = _records(holder, snapshot, result)
        self.assertEqual(emitted["scheduled"], [[1, 0], [0, 0]])
        victims, actions = preemption.check_declared_order(
            "one-free-block", snap, plan, emitted)
        self.assertEqual(([d["request_id"] for d in actions], victims),
                         (["1", "0"], []))
        obsolete = dict(emitted, scheduled=[[0, 0], [1, 0]])
        with self.assertRaisesRegex(CheckFailure, "emitted"):
            preemption.check_declared_order("obsolete", snap, plan, obsolete)
        bad_plan = dict(plan, execution_order=["0", "1"])
        with self.assertRaisesRegex(CheckFailure, "declared"):
            preemption.check_declared_order("obsolete", snap, bad_plan, emitted)

    def test_prefills_follow_service_rank_not_identity(self):
        holder, snapshot, result = _service_rank_case()
        snap, plan, emitted = _records(holder, snapshot, result)
        ranks = {r["raw_seq_id"]: r["service_rank"] for r in snap["requests"]}
        self.assertEqual(ranks, {5: 0, 1: 1, 0: 2})
        self.assertEqual(emitted["scheduled"], [[5, 2], [1, 2], [0, 2]])
        preemption.check_declared_order("service-rank", snap, plan, emitted)
        obsolete = dict(emitted, scheduled=[[0, 2], [1, 2], [5, 2]])
        with self.assertRaises(CheckFailure):
            preemption.check_declared_order("obsolete", snap, plan, obsolete)

    def test_multiple_victims_follow_victim_rank_not_identity(self):
        holder, snapshot, result = fx._victim_case_with_plan()
        snap, plan, emitted = _records(holder, snapshot, result)
        victim_ranks = {r["raw_seq_id"]: r["victim_rank"]
                        for r in snap["requests"] if r["preemption_eligible"]}
        self.assertEqual(victim_ranks, {1: 2, 2: 1, 3: 0})
        self.assertEqual(emitted["preempted_seq_ids"], [3, 1])
        victims, _ = preemption.check_declared_order(
            "victims", snap, plan, emitted)
        self.assertEqual(victims, ["3", "1"])
        obsolete = dict(emitted, preempted_seq_ids=[1, 3])
        with self.assertRaises(CheckFailure):
            preemption.check_declared_order("obsolete", snap, plan, obsolete)
        bad_plan = dict(plan, victim_order=["1", "3"])
        with self.assertRaises(CheckFailure):
            preemption.check_declared_order("obsolete", snap, bad_plan, emitted)

    def test_full_decision_check_uses_derived_order(self):
        """``check_decision`` of the preemption helper on complete serialized
        events. Its fixed case constants are patched only to this fixture."""
        holder, snapshot, result = _one_free_block_case()
        iteration = holder._iteration_id
        free_before = holder.block_manager.get_num_free_gpu_blocks()
        snap, plan, emitted = _records(holder, snapshot, result)
        events = [
            dict(call="map_scheduler_state", result_type="StateSnapshot",
                 result=snap, num_running_batches=0, iteration_id=iteration),
            dict(call="solve_and_extract", result_type="SchedulingSuccess",
                 result=to_jsonable(result)),
            dict(call="execute_plan", result_type="SchedulerOutputs",
                 result=emitted),
            dict(call="schedule", outputs=emitted),
            dict(call="_run_workers", method="execute_model",
                 sampler_outputs=[[s, 7] for s, _ in emitted["scheduled"]]),
        ]
        before = dict(iteration_id=iteration - 1, free_blocks=free_before,
                      waiting=[], running=[0, 1])
        expected = dict(preempted=[], scheduled=[[1, 0], [0, 0]])
        constants = dict(MAX_NUM_SEQS=fx.RESIDENT_LIMIT, B_MAX=fx.B_MAX,
                         C_MAX=fx.C_MAX, S_MAX=fx.S_MAX, DECODE_UTILITY=1.0,
                         PREFILL_TOKEN_UTILITY=0.0, PREEMPTION_PENALTY=1.0)
        patches = [mock.patch.object(preemption, k, v)
                   for k, v in constants.items()]
        policy_json = to_jsonable(fx.NUMERICAL_POLICY)
        for p in patches:
            p.start()
        try:
            preemption.check_decision("full", expected, before, events,
                                      policy_json)
            stale = copy.deepcopy(events)
            for e in (stale[2]["result"], stale[3]["outputs"]):
                e["scheduled"] = [[0, 0], [1, 0]]
            stale_expected = dict(expected, scheduled=[[0, 0], [1, 0]])
            with self.assertRaises(CheckFailure):
                preemption.check_decision("obsolete", stale_expected, before,
                                          stale, policy_json)
        finally:
            for p in patches:
                p.stop()


if __name__ == "__main__":
    unittest.main()
