import dataclasses
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lp_relaxation_scheduler as lrs  # noqa: E402
import lpserve_plan_execution as lpe  # noqa: E402
import lpserve_state_mapping as lsm  # noqa: E402

from sarathi.core.block_space_manager.vllm_block_space_manager import (  # noqa: E402
    VLLMBlockSpaceManager,
)
from sarathi.core.datatypes.sampling_params import SamplingParams  # noqa: E402
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs  # noqa: E402
from sarathi.core.datatypes.sequence import Sequence  # noqa: E402
from sarathi.core.datatypes.sequence_status import SequenceStatus  # noqa: E402
from sarathi.core.scheduler.base_scheduler import BaseScheduler  # noqa: E402

# Scoped fixture inputs, not production defaults or research decisions.
BLOCK_SIZE = 4
NUM_GPU_BLOCKS = 10
MAX_MODEL_LEN = 32
ITERATION_ID = 7
RESIDENT_LIMIT = 4
SNAPSHOT_TIME = 100.0
B_MAX = 8
C_MAX = 4
S_MAX = 3
MEMORY_RESERVE = 1
DECODE_POLICY_ID = "conservative_one_block_v1"
# Default native watermark (0.01 of 10 blocks -> 0 blocks). The combined-memory
# case uses a visibly different 0.2 (2 blocks) to exercise the native gate.
DEFAULT_WATERMARK = 0.01
TIGHT_WATERMARK = 0.2

NUMERICAL_POLICY = lrs.NumericalPolicy(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)


class _ConfigHolder:
    def __init__(self):
        self.num_pipeline_stages = 1
        self.max_num_seqs = RESIDENT_LIMIT
        self.max_model_len = MAX_MODEL_LEN


class _SchedulerHolder:
    """Minimal scheduler holder reusing the inherited native mutation helpers
    without initializing an engine, model, or GPU."""

    _allocate = BaseScheduler._allocate
    _append_slot = BaseScheduler._append_slot

    def __init__(self, watermark=DEFAULT_WATERMARK):
        self.scheduler_config = _ConfigHolder()
        self.block_manager = VLLMBlockSpaceManager(
            BLOCK_SIZE, NUM_GPU_BLOCKS, MAX_MODEL_LEN, watermark=watermark,
        )
        self._iteration_id = ITERATION_ID
        self.num_running_batches = 0
        self.waiting = []
        self.running = []


def _sequence(raw_seq_id, num_prompt_tokens, arrival_time=1.0):
    return Sequence(
        seq_id=raw_seq_id,
        prompt="",
        prompt_token_ids=list(range(num_prompt_tokens)),
        block_size=BLOCK_SIZE,
        eos_token_id=-2,
        arrival_time=arrival_time,
        sampling_params=SamplingParams(),
    )


def _add_waiting(holder, raw_seq_id, num_prompt_tokens, arrival_time=1.0):
    seq = _sequence(raw_seq_id, num_prompt_tokens, arrival_time)
    holder.waiting.append(seq)
    return seq


def _add_resident(holder, raw_seq_id, num_prompt_tokens, processed,
                  num_output_tokens=0):
    """Allocated PAUSED resident built through legal native transitions."""
    seq = _sequence(raw_seq_id, num_prompt_tokens)
    seq.set_status(SequenceStatus.RUNNING)
    holder.block_manager.allocate(seq)
    if processed:
        seq.update_prompt_tokens_processed(processed)
    seq.set_status(SequenceStatus.PAUSED)
    for token in range(num_output_tokens):
        seq.append_token_id(1000 + token)
    holder.running.append(seq)
    return seq


def _utilities(**by_id):
    return {
        int(k[1:]): lsm.RequestUtility(*v) for k, v in by_id.items()
    }


def _map(holder, utilities):
    snapshot = lsm.map_scheduler_state(
        holder,
        snapshot_time=SNAPSHOT_TIME,
        b_max=B_MAX,
        c_max=C_MAX,
        s_max=S_MAX,
        memory_reserve=MEMORY_RESERVE,
        decode_memory_policy_id=DECODE_POLICY_ID,
        utilities=utilities,
        numerical_policy=NUMERICAL_POLICY,
    )
    assert isinstance(snapshot, lsm.StateSnapshot), snapshot
    return snapshot


def _map_and_solve(holder, utilities):
    snapshot = _map(holder, utilities)
    result = lrs.solve_and_extract(snapshot.lp_problem)
    assert isinstance(result, lrs.SchedulingSuccess), result
    return snapshot, result


def _actions(plan):
    return {
        d.request_id: (d.prefill_tokens, d.decode, d.preempt)
        for d in plan.decisions
        if d.prefill_tokens or d.decode or d.preempt
    }


def _with_plan(problem, result, actions):
    """Replace the plan with one built from ``actions`` (request_id ->
    (prefill_tokens, decode, preempt)); residuals are recomputed."""
    decisions = []
    tokens = width = memory = 0
    objective = 0.0
    for r in problem.requests:
        x, y, z = actions.get(r.request_id, (0, 0, 0))
        ind = 1 if x > 0 else 0
        decisions.append(lrs.IntegerDecision(
            r.request_id, r.order_key, x, y, z, ind,
        ))
        tokens += x + y
        width += ind + y
        memory += (
            r.prefill_fixed_charge * ind + r.decode_charge * y
            - r.preemption_recovery * z
        )
        objective += (
            r.prefill_token_utility * x + r.decode_utility * y
            - r.preemption_penalty * z
        )
    plan = dataclasses.replace(
        result.plan,
        decisions=tuple(decisions),
        residual_token_capacity=problem.b_max - tokens,
        residual_action_capacity=problem.s_max - width,
        residual_memory_capacity=problem.m_free - problem.w - memory,
        fractional_request_count=0,
        dominant_preemption_ids=tuple(
            r.request_id for r in problem.requests
            if actions.get(r.request_id, (0, 0, 0))[2]
        ),
        safety_preemption_ids=(),
        objective=objective,
    )
    return dataclasses.replace(result, plan=plan)


def _seq_fingerprint(seq):
    return (
        seq.seq_id,
        seq.get_status().name,
        tuple(seq.prompt_token_ids),
        tuple(seq.output_token_ids),
        seq.prompt_tokens_processed,
        seq.prompt_processing_finished,
        tuple(
            (b.block_number, b.num_tokens, tuple(b.token_ids))
            for b in seq.logical_token_blocks
        ),
    )


def _fingerprint(holder):
    """Everything a precommit rejection must leave unchanged."""
    block_manager = holder.block_manager
    return (
        tuple(id(s) for s in holder.waiting),
        tuple(id(s) for s in holder.running),
        tuple(_seq_fingerprint(s) for s in holder.waiting),
        tuple(_seq_fingerprint(s) for s in holder.running),
        holder._iteration_id,
        holder.num_running_batches,
        tuple(sorted(
            (seq_id, tuple(b.block_number for b in table))
            for seq_id, table in block_manager.block_tables.items()
        )),
        tuple(b.block_number for b in block_manager.gpu_allocator.free_blocks),
    )


def _emitted(outputs):
    return [
        (m.seq_id, m.prompt_chunk_len)
        for m in outputs.scheduled_seq_metadata_list
    ]


def _table(holder, raw_seq_id):
    return tuple(
        b.block_number for b in holder.block_manager.block_tables[raw_seq_id]
    )


# Admission of waiting 0, continuation of resident partial prefill 1, an
# unrelated unselected resident decode 2, and an unrelated future waiting 3.
# Utilities reuse the mapper fixture except request 2's decode utility, lowered
# from 3.0 to 0.5 so the unique optimum leaves it unselected.
CASE_ONE_UTILITIES = _utilities(
    r0=(0.0, 2.0, 0.0), r1=(0.0, 1.0, 0.25), r2=(0.5, 0.0, 0.5),
)


def _build_case_one():
    holder = _SchedulerHolder()
    _add_waiting(holder, 0, 6)
    _add_waiting(holder, 3, 6, arrival_time=SNAPSHOT_TIME + 1.0)
    _add_resident(holder, 1, 6, processed=2)
    _add_resident(holder, 2, 4, processed=4)
    return holder


# Mixed case: resident decode 0 (block gap one), waiting admission 1, and
# resident partial prefill 2. Global ID order would put the decode first.
MIXED_UTILITIES = _utilities(
    r0=(3.0, 0.0, 0.5), r1=(0.0, 2.0, 0.0), r2=(0.0, 1.0, 0.25),
)


def _build_mixed_case():
    holder = _SchedulerHolder()
    _add_resident(holder, 0, 4, processed=4, num_output_tokens=1)
    _add_waiting(holder, 1, 6)
    _add_resident(holder, 2, 6, processed=2)
    return holder


class PlanExecutionTest(unittest.TestCase):
    def test_admission_and_resident_prefill_through_mapper_and_solver(self):
        holder = _build_case_one()
        snapshot, result = _map_and_solve(holder, CASE_ONE_UTILITIES)
        self.assertEqual(
            _actions(result.plan), {"0": (4, 0, 0), "1": (4, 0, 0)},
        )

        seqs = {s.seq_id: s for s in holder.waiting + holder.running}
        unrelated_before = {
            k: _seq_fingerprint(seqs[k]) for k in (0, 1, 2, 3)
        }
        table_1, table_2 = _table(holder, 1), _table(holder, 2)
        free_before = holder.block_manager.get_num_free_gpu_blocks()
        self.assertEqual(free_before, 7)

        outputs = lpe.execute_plan(holder, snapshot, result)

        self.assertIsInstance(outputs, SchedulerOutputs)
        self.assertEqual(outputs.id, ITERATION_ID)
        self.assertEqual(_emitted(outputs), [(0, 4), (1, 4)])
        self.assertEqual(outputs.ignored_seq_ids, [])
        self.assertEqual(outputs.preempted_seq_ids, [])
        self.assertEqual(outputs.num_batched_prompt_tokens, 8)
        self.assertEqual(outputs.num_batched_output_tokens, 0)
        self.assertEqual(outputs.num_batched_tokens, 8)

        # Queue membership: admission removed once and appended once; the
        # untouched waiting and resident order is preserved.
        self.assertEqual([s.seq_id for s in holder.waiting], [3])
        self.assertEqual([s.seq_id for s in holder.running], [1, 2, 0])
        self.assertIs(holder.waiting[0], seqs[3])
        self.assertEqual(
            [id(s) for s in holder.running],
            [id(seqs[1]), id(seqs[2]), id(seqs[0])],
        )

        # Allocation: full logical context for the 4-token chunk of a
        # 6-token prompt (2 blocks); zero for the resident prefill.
        self.assertEqual(len(_table(holder, 0)), 2)
        self.assertEqual(_table(holder, 1), table_1)
        self.assertEqual(_table(holder, 2), table_2)
        self.assertEqual(set(holder.block_manager.block_tables), {0, 1, 2})
        self.assertEqual(holder.block_manager.get_num_free_gpu_blocks(), 5)

        # Status and prompt progress are left to replay and completion.
        for k in (0, 1, 2, 3):
            self.assertEqual(_seq_fingerprint(seqs[k]), unrelated_before[k])
        self.assertEqual(holder._iteration_id, ITERATION_ID)
        self.assertEqual(holder.num_running_batches, 0)

    def test_decode_with_block_gap_zero_and_one(self):
        holder = _SchedulerHolder()
        gap_zero = _add_resident(holder, 0, 4, processed=4)
        gap_one = _add_resident(holder, 1, 4, processed=4, num_output_tokens=1)
        self.assertEqual(
            (len(gap_zero.logical_token_blocks), len(_table(holder, 0))), (1, 1),
        )
        self.assertEqual(
            (len(gap_one.logical_token_blocks), len(_table(holder, 1))), (2, 1),
        )
        utilities = _utilities(r0=(1.0, 0.0, 1.0), r1=(1.0, 0.0, 1.0))
        snapshot, result = _map_and_solve(holder, utilities)
        self.assertEqual(
            _actions(result.plan), {"0": (0, 1, 0), "1": (0, 1, 0)},
        )
        table_0, table_1 = _table(holder, 0), _table(holder, 1)
        self.assertEqual(holder.block_manager.get_num_free_gpu_blocks(), 8)

        outputs = lpe.execute_plan(holder, snapshot, result)

        self.assertIsInstance(outputs, SchedulerOutputs)
        self.assertEqual(_emitted(outputs), [(0, 0), (1, 0)])
        self.assertEqual(outputs.num_batched_prompt_tokens, 0)
        self.assertEqual(outputs.num_batched_output_tokens, 2)
        # Gap zero appends no block; gap one appends exactly one.
        self.assertEqual(_table(holder, 0), table_0)
        self.assertEqual(_table(holder, 1)[:1], table_1)
        self.assertEqual(len(_table(holder, 1)), 2)
        self.assertEqual(holder.block_manager.get_num_free_gpu_blocks(), 7)
        self.assertEqual([s.seq_id for s in holder.running], [0, 1])
        self.assertEqual(holder.waiting, [])

    def test_append_gate_rejects_gap_zero_decode_without_free_block(self):
        holder = _SchedulerHolder()
        _add_resident(holder, 0, 4, processed=4)
        utilities = _utilities(r0=(1.0, 0.0, 1.0))
        snapshot, result = _map_and_solve(holder, utilities)
        self.assertEqual(_actions(result.plan), {"0": (0, 1, 0)})
        self.assertIs(
            lrs.validate_integer_plan(snapshot.lp_problem, result.plan),
            result.plan,
        )
        # Current state differs from the snapshot: every free block is held
        # elsewhere, so the native append gate fails even at gap zero.
        allocator = holder.block_manager.gpu_allocator
        held = [allocator.allocate() for _ in range(allocator.get_num_free_blocks())]
        self.assertEqual(len(held), 9)
        self.assertFalse(holder.block_manager.can_append_slot())

        before = _fingerprint(holder)
        failure = lpe.execute_plan(holder, snapshot, result)
        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(failure, lrs.Failure)
        self.assertEqual(failure.stage, "precommit_validation")
        self.assertEqual(failure.category, "append_gate")
        self.assertEqual(failure.problem_id, str(ITERATION_ID))

    def test_mixed_output_is_prompt_first_in_native_execution_order(self):
        holder = _build_mixed_case()
        snapshot, result = _map_and_solve(holder, MIXED_UTILITIES)
        self.assertEqual(
            _actions(result.plan),
            {"0": (0, 1, 0), "1": (4, 0, 0), "2": (3, 0, 0)},
        )
        self.assertEqual(holder.block_manager.get_num_free_gpu_blocks(), 7)

        calls = []
        allocate, append_slot = holder._allocate, holder._append_slot

        def record_allocate(seq):
            calls.append(("allocate", seq.seq_id))
            allocate(seq)

        def record_append(seq):
            calls.append(("append_slot", seq.seq_id))
            append_slot(seq)

        with mock.patch.object(holder, "_allocate", side_effect=record_allocate), \
                mock.patch.object(holder, "_append_slot", side_effect=record_append):
            outputs = lpe.execute_plan(holder, snapshot, result)

        self.assertIsInstance(outputs, SchedulerOutputs)
        emitted = _emitted(outputs)
        self.assertEqual(emitted, [(1, 4), (2, 3), (0, 0)])
        self.assertNotEqual(
            [seq_id for seq_id, _ in emitted],
            sorted(seq_id for seq_id, _ in emitted),
        )
        # Central native operations follow the emitted order.
        self.assertEqual(calls, [("allocate", 1), ("append_slot", 0)])
        self.assertEqual(outputs.num_batched_prompt_tokens, 7)
        self.assertEqual(outputs.num_batched_output_tokens, 1)
        self.assertEqual(outputs.num_batched_tokens, 8)
        self.assertEqual([s.seq_id for s in holder.running], [0, 2, 1])
        self.assertEqual(holder.waiting, [])
        self.assertEqual(len(_table(holder, 1)), 2)
        self.assertEqual(len(_table(holder, 0)), 2)
        self.assertEqual(holder.block_manager.get_num_free_gpu_blocks(), 4)

    def test_combined_memory_rejection_with_native_watermark(self):
        holder = _SchedulerHolder(watermark=TIGHT_WATERMARK)
        self.assertEqual(holder.block_manager.watermark_blocks, 2)
        _add_resident(holder, 2, 12, processed=4)  # 3 blocks, unselected
        _add_resident(holder, 3, 8, processed=8)  # 2 blocks, unselected
        first = _add_waiting(holder, 0, 8)  # 2 blocks
        second = _add_waiting(holder, 1, 8)  # 2 blocks
        utilities = _utilities(
            r0=(0.0, 1.0, 0.0), r1=(0.0, 1.0, 0.0),
            r2=(0.0, 0.1, 1.0), r3=(0.1, 0.0, 1.0),
        )
        snapshot, result = _map_and_solve(holder, utilities)
        self.assertEqual(
            _actions(result.plan), {"0": (4, 0, 0), "1": (4, 0, 0)},
        )
        # Planning memory with reserve 1 is exactly feasible: 2 + 2 <= 5 - 1.
        self.assertEqual(snapshot.lp_problem.m_free, 5)
        self.assertEqual(result.plan.residual_memory_capacity, 0)
        # Each admission alone passes the native gate against the original
        # free count, so independent checks would accept the plan.
        self.assertTrue(holder.block_manager.can_allocate(first))
        self.assertTrue(holder.block_manager.can_allocate(second))

        before = _fingerprint(holder)
        failure = lpe.execute_plan(holder, snapshot, result)
        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(failure, lrs.Failure)
        self.assertEqual(failure.stage, "precommit_validation")
        self.assertEqual(failure.category, "admission_gate")
        # The second admission sees 3 free blocks: 3 - 2 < watermark 2.
        self.assertIn("seq_id 1", failure.reason)
        self.assertIn("3 free", failure.reason)

    def test_compact_precommit_rejections(self):
        def set_iteration(h, s, r):
            h._iteration_id = ITERATION_ID + 1
            return r

        def upstream_failure(h, s, r):
            return lrs.Failure(s.snapshot_id, "solver", "infeasible", "fixture")

        def in_flight(h, s, r):
            h.num_running_batches = 1
            return r

        def pipeline(h, s, r):
            h.scheduler_config.num_pipeline_stages = 2
            return r

        def resident_missing(h, s, r):
            h.running = [seq for seq in h.running if seq.seq_id != 1]
            return r

        def waiting_allocated(h, s, r):
            h.block_manager.allocate(h.waiting[0])
            return r

        def future_arrival(h, s, r):
            h.waiting[0].arrival_time = SNAPSHOT_TIME + 50.0
            return r

        def progressed_resident(h, s, r):
            [seq] = [seq for seq in h.running if seq.seq_id == 1]
            seq.update_prompt_tokens_processed(3)
            return r

        def resident_capacity(h, s, r):
            h.scheduler_config.max_num_seqs = len(h.running)
            return r

        def overlength(h, s, r):
            h.scheduler_config.max_model_len = 5
            return r

        def plan_chunk_bound(h, s, r):
            return _with_plan(s.lp_problem, r, {"0": (5, 0, 0)})

        def plan_token_limit(h, s, r):
            return _with_plan(
                s.lp_problem, r,
                {"0": (4, 0, 0), "1": (4, 0, 0), "2": (0, 1, 0)},
            )

        cases = {
            "iteration_mismatch": (set_iteration, "snapshot_mismatch"),
            "upstream_failure": (upstream_failure, "upstream_failure"),
            "in_flight_batch": (in_flight, "unsupported_state"),
            "pipeline_stages": (pipeline, "unsupported_state"),
            "resident_not_owned": (resident_missing, "ownership_mismatch"),
            "waiting_already_allocated": (waiting_allocated, "ineligible_action"),
            "arrival_after_decision": (future_arrival, "ineligible_action"),
            "chunk_above_current_remainder": (progressed_resident, "chunk_bound"),
            "resident_capacity": (resident_capacity, "resident_capacity"),
            "overlength_admission": (overlength, "overlength_admission"),
            "plan_chunk_bound": (plan_chunk_bound, "ineligible_action"),
            "plan_token_limit": (plan_token_limit, "capacity_violation"),
        }
        validator_categories = {"plan_chunk_bound", "plan_token_limit"}

        for name, (perturb, category) in cases.items():
            with self.subTest(name):
                holder = _build_case_one()
                snapshot, result = _map_and_solve(holder, CASE_ONE_UTILITIES)
                result = perturb(holder, snapshot, result)
                before = _fingerprint(holder)
                failure = lpe.execute_plan(holder, snapshot, result)
                self.assertEqual(_fingerprint(holder), before)
                self.assertIsInstance(failure, lrs.Failure)
                self.assertEqual(failure.category, category)
                self.assertEqual(failure.problem_id, str(ITERATION_ID))
                self.assertEqual(
                    failure.stage,
                    "plan_validation" if name in validator_categories
                    else "precommit_validation",
                )

    def test_no_progress_and_selected_preemption_gate(self):
        cases = {
            "all_zero": ({}, "no_progress"),
            "preempt_only": ({"2": (0, 0, 1)}, "no_progress"),
            "execution_with_preemption": (
                {"0": (4, 0, 0), "2": (0, 0, 1)}, "unsupported_preemption",
            ),
        }
        for name, (actions, category) in cases.items():
            with self.subTest(name):
                holder = _build_case_one()
                snapshot, result = _map_and_solve(holder, CASE_ONE_UTILITIES)
                result = _with_plan(snapshot.lp_problem, result, actions)
                # The constructed plan is mathematically legitimate.
                self.assertIs(
                    lrs.validate_integer_plan(snapshot.lp_problem, result.plan),
                    result.plan,
                )
                before = _fingerprint(holder)
                failure = lpe.execute_plan(holder, snapshot, result)
                self.assertEqual(_fingerprint(holder), before)
                self.assertIsInstance(failure, lrs.Failure)
                self.assertEqual(failure.stage, "precommit_validation")
                self.assertEqual(failure.category, category)
                self.assertEqual(failure.problem_id, str(ITERATION_ID))

    def test_malformed_nested_records_return_failure(self):
        cases = {
            "plan_none": (
                lambda s, r: (s, dataclasses.replace(r, plan=None)),
                "result.plan",
            ),
            "lp_problem_none": (
                lambda s, r: (dataclasses.replace(s, lp_problem=None), r),
                "snapshot.lp_problem",
            ),
        }
        for name, (malform, record) in cases.items():
            with self.subTest(name):
                holder = _build_case_one()
                snapshot, result = _map_and_solve(holder, CASE_ONE_UTILITIES)
                snapshot, result = malform(snapshot, result)
                before = _fingerprint(holder)
                with mock.patch.object(
                    holder, "_allocate", wraps=holder._allocate,
                ) as allocate_spy, mock.patch.object(
                    holder, "_append_slot", wraps=holder._append_slot,
                ) as append_spy:
                    failure = lpe.execute_plan(holder, snapshot, result)
                self.assertEqual(_fingerprint(holder), before)
                allocate_spy.assert_not_called()
                append_spy.assert_not_called()
                self.assertIsInstance(failure, lrs.Failure)
                self.assertEqual(failure.stage, "precommit_validation")
                self.assertEqual(failure.category, "malformed_input")
                self.assertEqual(failure.problem_id, str(ITERATION_ID))
                self.assertIn(record, failure.reason)

    def test_post_mutation_exception_propagates_without_recovery(self):
        holder = _build_mixed_case()
        snapshot, result = _map_and_solve(holder, MIXED_UTILITIES)

        # Test double: the native allocation raises after the admission has
        # already been removed from waiting (the first execution mutation).
        with mock.patch.object(
            holder, "_allocate", side_effect=RuntimeError("induced failure"),
        ), mock.patch.object(
            holder, "_append_slot", wraps=holder._append_slot,
        ) as append_spy:
            with self.assertRaisesRegex(RuntimeError, "induced failure"):
                lpe.execute_plan(holder, snapshot, result)

        # No later action ran and nothing was restored.
        append_spy.assert_not_called()
        self.assertEqual(holder.waiting, [])
        self.assertEqual([s.seq_id for s in holder.running], [0, 2])
        self.assertNotIn(1, holder.block_manager.block_tables)
        # The mutated fixture is discarded, not reused.
        del holder


if __name__ == "__main__":
    unittest.main()
