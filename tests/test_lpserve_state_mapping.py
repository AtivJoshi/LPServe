import dataclasses
import functools
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lp_relaxation_scheduler as lrs  # noqa: E402
import lpserve_state_mapping as lsm  # noqa: E402

from sarathi.core.block_space_manager.vllm_block_space_manager import (  # noqa: E402
    VLLMBlockSpaceManager,
)
from sarathi.core.datatypes.sampling_params import SamplingParams  # noqa: E402
from sarathi.core.datatypes.sequence import Sequence  # noqa: E402
from sarathi.core.datatypes.sequence_status import SequenceStatus  # noqa: E402

BLOCK_SIZE = 4
NUM_GPU_BLOCKS = 10
MAX_MODEL_LEN = 32
ITERATION_ID = 7
NUM_PIPELINE_STAGES = 1
NUM_RUNNING_BATCHES = 0
RESIDENT_LIMIT = 4
SNAPSHOT_TIME = 100.0
B_MAX = 8
C_MAX = 4
S_MAX = 3

NUMERICAL_POLICY = lrs.NumericalPolicy(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)


class _ConfigHolder:
    """A tiny faithful stand-in for the scheduler_config fields read."""

    def __init__(self, num_pipeline_stages, max_num_seqs,
                 max_model_len=MAX_MODEL_LEN):
        self.num_pipeline_stages = num_pipeline_stages
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len


class _SchedulerHolder:
    """A tiny faithful stand-in exposing only the fields the mapper reads."""

    def __init__(
        self, scheduler_config, block_manager, iteration_id, num_running_batches,
    ):
        self.scheduler_config = scheduler_config
        self.block_manager = block_manager
        self._iteration_id = iteration_id
        self.num_running_batches = num_running_batches
        self.waiting = []
        self.running = []


def _build_holder():
    """A fresh block manager and scheduler holder for one isolated test."""
    block_manager = VLLMBlockSpaceManager(BLOCK_SIZE, NUM_GPU_BLOCKS, MAX_MODEL_LEN)
    holder = _SchedulerHolder(
        _ConfigHolder(NUM_PIPELINE_STAGES, RESIDENT_LIMIT),
        block_manager, ITERATION_ID, NUM_RUNNING_BATCHES,
    )
    return holder, block_manager


def _make_sequence(raw_seq_id, num_prompt_tokens, arrival_time):
    return Sequence(
        seq_id=raw_seq_id,
        prompt="",
        prompt_token_ids=list(range(num_prompt_tokens)),
        block_size=BLOCK_SIZE,
        eos_token_id=-2,
        arrival_time=arrival_time,
        sampling_params=SamplingParams(),
    )


def _fingerprint(holder):
    """A primitive fingerprint of everything mapping must leave unchanged."""
    block_manager = holder.block_manager

    def seq_fingerprint(seq):
        return (
            seq.seq_id,
            seq.get_status().name,
            tuple(seq.prompt_token_ids),
            tuple(seq.output_token_ids),
            seq.prompt_tokens_processed,
            seq.prompt_processing_finished,
            tuple(
                (
                    block.block_number, block.block_size, block.num_tokens,
                    tuple(block.token_ids),
                )
                for block in seq.logical_token_blocks
            ),
        )

    return (
        tuple(id(s) for s in holder.waiting),
        tuple(id(s) for s in holder.running),
        holder._iteration_id,
        holder.num_running_batches,
        tuple(seq_fingerprint(s) for s in holder.waiting),
        tuple(seq_fingerprint(s) for s in holder.running),
        tuple(sorted(
            (seq_id, tuple(b.block_number for b in table))
            for seq_id, table in block_manager.block_tables.items()
        )),
        tuple(b.block_number for b in block_manager.gpu_allocator.free_blocks),
    )


def _assert_no_mutable_or_framework_objects(value, seen=None):
    """Recursively assert no Sequence/scheduler/block-manager/list/dict escaped."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Sequence):
        raise AssertionError("mapped output retains a live Sequence object")
    if isinstance(value, VLLMBlockSpaceManager):
        raise AssertionError("mapped output retains a live block-manager object")
    if isinstance(value, _SchedulerHolder):
        raise AssertionError("mapped output retains a live scheduler object")
    if isinstance(value, list):
        raise AssertionError(f"mapped output contains a mutable list: {value!r}")
    if isinstance(value, dict):
        raise AssertionError(f"mapped output contains a mutable dict: {value!r}")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _assert_no_mutable_or_framework_objects(
                getattr(value, field.name), seen,
            )
    elif isinstance(value, (tuple, frozenset, set)):
        for item in value:
            _assert_no_mutable_or_framework_objects(item, seen)


def _three_request_utilities():
    """A fresh caller-owned utility mapping for raw_seq_ids 0, 1, and 2."""
    return {
        0: lsm.RequestUtility(
            decode_utility=0.0, prefill_token_utility=2.0,
            preemption_penalty=0.0,
        ),
        1: lsm.RequestUtility(
            decode_utility=0.0, prefill_token_utility=1.0,
            preemption_penalty=0.25,
        ),
        2: lsm.RequestUtility(
            decode_utility=3.0, prefill_token_utility=0.0,
            preemption_penalty=0.5,
        ),
    }


def _build_three_request_holder():
    """Waiting request 0, resident partial prefill 1, resident decode 2."""
    holder, block_manager = _build_holder()
    holder.waiting.append(_make_sequence(0, 6, 1.0))

    partial_prefill_seq = _make_sequence(1, 6, 2.0)
    partial_prefill_seq.set_status(SequenceStatus.RUNNING)
    block_manager.allocate(partial_prefill_seq)
    partial_prefill_seq.update_prompt_tokens_processed(2)
    partial_prefill_seq.set_status(SequenceStatus.PAUSED)
    holder.running.append(partial_prefill_seq)

    decode_seq = _make_sequence(2, 4, 3.0)
    decode_seq.set_status(SequenceStatus.RUNNING)
    block_manager.allocate(decode_seq)
    decode_seq.update_prompt_tokens_processed(4)
    decode_seq.set_status(SequenceStatus.PAUSED)
    holder.running.append(decode_seq)
    return holder


def _map(holder, utilities, **overrides):
    kwargs = dict(
        snapshot_time=SNAPSHOT_TIME,
        b_max=B_MAX,
        c_max=C_MAX,
        s_max=S_MAX,
        coefficient_policy=utilities,
        rank_profile=lsm.vllm_oriented_ranks,
        numerical_policy=NUMERICAL_POLICY,
    )
    kwargs.update(overrides)
    return lsm.map_scheduler_state(holder, **kwargs)


class LPServeStateMappingTest(unittest.TestCase):
    def test_supported_mapping_and_nonmutation(self):
        holder, block_manager = _build_holder()

        # Request 0: waiting, six prompt tokens, none processed, unallocated.
        waiting_seq = _make_sequence(0, 6, 1.0)
        holder.waiting.append(waiting_seq)

        # Request 1: resident paused partial prefill, six prompt tokens, two
        # processed, allocated. Legal transitions: WAITING -> RUNNING -> PAUSED.
        partial_prefill_seq = _make_sequence(1, 6, 2.0)
        partial_prefill_seq.set_status(SequenceStatus.RUNNING)
        block_manager.allocate(partial_prefill_seq)
        partial_prefill_seq.update_prompt_tokens_processed(2)
        partial_prefill_seq.set_status(SequenceStatus.PAUSED)
        holder.running.append(partial_prefill_seq)

        # Request 2: resident paused decode, four prompt tokens, all four
        # processed, allocated. Legal transitions: WAITING -> RUNNING -> PAUSED.
        decode_seq = _make_sequence(2, 4, 3.0)
        decode_seq.set_status(SequenceStatus.RUNNING)
        block_manager.allocate(decode_seq)
        decode_seq.update_prompt_tokens_processed(4)
        decode_seq.set_status(SequenceStatus.PAUSED)
        holder.running.append(decode_seq)

        utilities = _three_request_utilities()
        utilities_before = dict(utilities)

        before = _fingerprint(holder)

        result = _map(holder, utilities)

        # Proof of non-mutation: nothing observable about the scheduler
        # holder, its sequences, or the block manager changed.
        self.assertEqual(_fingerprint(holder), before)

        self.assertIsInstance(result, lsm.StateSnapshot)
        requests = result.requests
        self.assertEqual(
            tuple(r.request_id for r in requests), ("0", "1", "2"),
        )
        self.assertEqual(
            tuple(r.order_key for r in requests), ((0,), (1,), (2,)),
        )

        # The snapshot retains allocation quantities only: the waiting request
        # has a zero physical block count, residents carry their block-table
        # lengths, and no block identities are copied.
        self.assertEqual(
            tuple(r.physical_block_count for r in requests), (0, 2, 1),
        )
        self.assertEqual(
            tuple(r.logical_block_count for r in requests), (2, 2, 1),
        )

        problem = result.lp_problem
        problem_requests = problem.requests
        self.assertEqual(
            tuple(r.request_id for r in problem_requests), ("0", "1", "2"),
        )
        self.assertEqual(
            tuple(r.order_key for r in problem_requests), ((0,), (1,), (2,)),
        )
        self.assertEqual(
            tuple(r.prompt_tokens_remaining for r in problem_requests),
            (6, 4, 0),
        )
        self.assertEqual(
            tuple(r.prefill_upper_bound for r in problem_requests), (4, 4, 0),
        )
        self.assertEqual(
            tuple(
                (r.prefill_eligible, r.decode_eligible, r.preemption_eligible)
                for r in problem_requests
            ),
            ((True, False, False), (True, False, True), (False, True, True)),
        )
        self.assertEqual(
            tuple(r.prefill_fixed_charge for r in problem_requests), (2, 0, 0),
        )
        # The decode resident's four prompt tokens fill one logical and one
        # physical block and no output token has been appended yet: exact
        # decode demand 0. The partial prefill allocates nothing.
        self.assertEqual(
            tuple(r.decode_demand for r in problem_requests), (0, 0, 0),
        )
        self.assertEqual(
            tuple(r.resident for r in problem_requests), (False, True, True),
        )
        # vLLM-oriented ranks: running order (1, 2) then waiting (0);
        # victims in reverse running order.
        self.assertEqual(
            tuple((r.service_rank, r.victim_rank) for r in problem_requests),
            ((2, None), (0, 1), (1, 0)),
        )
        self.assertEqual(
            tuple(r.preemption_recovery for r in problem_requests), (0, 2, 1),
        )

        self.assertEqual(problem.legal_preemption_ids, frozenset({"1", "2"}))
        self.assertEqual(result.free_physical_blocks, 7)
        self.assertEqual(problem.b_max, 8)
        self.assertEqual(problem.c_max, 4)
        self.assertEqual(problem.s_max, 3)
        self.assertEqual(problem.m_free, 7)
        self.assertEqual(problem.admission_watermark, 0)  # int(0.01 * 10)
        self.assertEqual(problem.resident_count, 2)
        self.assertEqual(problem.resident_limit, RESIDENT_LIMIT)
        self.assertEqual(
            (result.total_blocks, result.watermark_blocks, result.max_model_len),
            (NUM_GPU_BLOCKS, 0, MAX_MODEL_LEN),
        )
        self.assertEqual(result.running_order, (1, 2))
        self.assertEqual(result.arrived_waiting_order, (0,))
        self.assertEqual(result.snapshot_id, str(ITERATION_ID))
        self.assertEqual(problem.problem_id, result.snapshot_id)

        revalidated = lrs.validate_problem(problem)
        self.assertIsInstance(revalidated, lrs.LPProblem)

        # Utilities are read by raw_seq_id, copied, and the caller's mapping
        # is left unchanged.
        self.assertEqual(utilities, utilities_before)
        for r in requests:
            self.assertEqual(r.utility, utilities[r.raw_seq_id])
            self.assertIsNot(r.utility, utilities[r.raw_seq_id])
        self.assertEqual(
            tuple(
                (r.decode_utility, r.prefill_token_utility, r.preemption_penalty)
                for r in problem_requests
            ),
            ((0.0, 2.0, 0.0), (0.0, 1.0, 0.25), (3.0, 0.0, 0.5)),
        )

        _assert_no_mutable_or_framework_objects(result)

    def test_malformed_arrival_time_is_rejected(self):
        holder, _ = _build_holder()

        valid_seq = _make_sequence(0, 6, 1.0)
        holder.waiting.append(valid_seq)

        # A waiting request with a non-finite arrival_time must not be
        # silently treated as a future request and omitted.
        nan_arrival_seq = _make_sequence(1, 6, float("nan"))
        holder.waiting.append(nan_arrival_seq)

        utilities = {
            0: lsm.RequestUtility(
                decode_utility=0.0, prefill_token_utility=2.0,
                preemption_penalty=0.0,
            ),
        }

        before = _fingerprint(holder)
        result = _map(holder, utilities)

        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(result, lsm.MappingFailure)
        self.assertEqual(result.stage, "state_mapping")
        self.assertEqual(result.category, "mapping_failure")
        self.assertIn("arrival_time", result.reason)

    def test_orphan_block_table_is_rejected(self):
        holder, block_manager = _build_holder()

        valid_seq = _make_sequence(0, 6, 1.0)
        holder.waiting.append(valid_seq)

        # Allocated through the real block manager, but never owned by
        # waiting or running: an orphan central block table.
        orphan_seq = _make_sequence(99, 4, 1.0)
        block_manager.allocate(orphan_seq)

        utilities = {
            0: lsm.RequestUtility(
                decode_utility=0.0, prefill_token_utility=2.0,
                preemption_penalty=0.0,
            ),
        }

        before = _fingerprint(holder)
        result = _map(holder, utilities)

        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(result, lsm.MappingFailure)
        self.assertEqual(result.stage, "state_mapping")
        self.assertEqual(result.category, "mapping_failure")
        self.assertIn("orphan", result.reason)

    def test_malformed_utilities_return_mapping_failure(self):
        good = lsm.RequestUtility(
            decode_utility=0.0, prefill_token_utility=1.0,
            preemption_penalty=0.0,
        )
        cases = {
            "none": (None, "mapping"),
            "pair_rows": (
                tuple(_three_request_utilities().items()), "mapping",
            ),
            # True hashes equal to 1, so it would otherwise match raw_seq_id 1.
            "bool_key": (
                {
                    True if k == 1 else k: v
                    for k, v in _three_request_utilities().items()
                },
                "integer raw_seq_id",
            ),
            "string_key": (
                {**_three_request_utilities(), "3": good}, "integer raw_seq_id",
            ),
            "non_record_value": (
                {**_three_request_utilities(), 2: (3.0, 0.0, 0.5)},
                "must be a RequestUtility",
            ),
            "missing_key": (
                {k: v for k, v in _three_request_utilities().items() if k != 1},
                "missing coefficients for raw_seq_id(s) [1]",
            ),
            "extra_key": (
                {**_three_request_utilities(), 5: good},
                "extra coefficients for raw_seq_id(s) [5]",
            ),
            "uniform_negative": (
                dataclasses.replace(good, preemption_penalty=-1.0),
                "preemption_penalty for raw_seq_id 0",
            ),
            "callable_wrong_type": (
                lambda captured, request: (1.0, 1.0, 1.0),
                "must be a RequestUtility",
            ),
        }
        for field in (
            "decode_utility", "prefill_token_utility", "preemption_penalty",
        ):
            for bad in (float("nan"), float("inf"), True, "1.0", -0.5,
                        10 ** 400):
                cases[f"{field}={bad!r}"] = (
                    {
                        **_three_request_utilities(),
                        0: dataclasses.replace(good, **{field: bad}),
                    },
                    f"coefficient {field} for raw_seq_id 0",
                )

        for name, (utilities, reason) in cases.items():
            with self.subTest(name):
                holder = _build_three_request_holder()
                before = _fingerprint(holder)
                result = _map(holder, utilities)
                self.assertEqual(_fingerprint(holder), before)
                self.assertIsInstance(result, lsm.MappingFailure)
                self.assertEqual(result.stage, "state_mapping")
                self.assertEqual(result.category, "mapping_failure")
                self.assertIn(reason, result.reason)

    def test_snapshot_is_independent_of_later_utility_changes(self):
        holder = _build_three_request_holder()
        utilities = _three_request_utilities()
        result = _map(holder, utilities)
        self.assertIsInstance(result, lsm.StateSnapshot)
        mapped = tuple(r.utility for r in result.requests)
        problem_utilities = tuple(
            (r.decode_utility, r.prefill_token_utility, r.preemption_penalty)
            for r in result.lp_problem.requests
        )

        utilities[0] = lsm.RequestUtility(
            decode_utility=9.0, prefill_token_utility=9.0,
            preemption_penalty=9.0,
        )
        del utilities[1]
        utilities[7] = utilities[2]

        self.assertEqual(tuple(r.utility for r in result.requests), mapped)
        self.assertEqual(
            tuple(
                (r.decode_utility, r.prefill_token_utility, r.preemption_penalty)
                for r in result.lp_problem.requests
            ),
            problem_utilities,
        )
        self.assertEqual(
            problem_utilities,
            ((0.0, 2.0, 0.0), (0.0, 1.0, 0.25), (3.0, 0.0, 0.5)),
        )

    def test_future_only_universe_is_mapping_failure(self):
        holder, _ = _build_holder()

        # One waiting request that has not yet arrived and no resident owner:
        # the arrived unfinished universe is empty, so direct mapping must not
        # produce an LPProblem.
        future_seq = _make_sequence(0, 6, 101.0)
        holder.waiting.append(future_seq)

        before = _fingerprint(holder)

        # Spies (test-only, not production fault-injection hooks) proving that
        # LPProblem construction and validate_problem are never reached.
        with mock.patch.object(
            lsm.lrs, "LPProblem", wraps=lsm.lrs.LPProblem,
        ) as lp_problem_spy, mock.patch.object(
            lsm.lrs, "validate_problem", wraps=lsm.lrs.validate_problem,
        ) as validate_problem_spy:
            result = _map(holder, {})

        self.assertEqual(_fingerprint(holder), before)

        self.assertIsInstance(result, lsm.MappingFailure)
        self.assertEqual(result.stage, "state_mapping")
        self.assertEqual(result.category, "mapping_failure")
        self.assertIn("arrived, unfinished request universe is empty",
                      result.reason)

        lp_problem_spy.assert_not_called()
        validate_problem_spy.assert_not_called()

        _assert_no_mutable_or_framework_objects(result)

    def test_iteration_snapshot_id_and_plan_identity_check(self):
        def map_at(iteration_id):
            holder, _ = _build_holder()
            holder._iteration_id = iteration_id
            holder.waiting.append(_make_sequence(0, 6, 1.0))
            utilities = {
                0: lsm.RequestUtility(
                    decode_utility=0.0, prefill_token_utility=1.0,
                    preemption_penalty=0.0,
                ),
            }
            return _map(holder, utilities)

        first = map_at(7)
        second = map_at(8)
        self.assertIsInstance(first, lsm.StateSnapshot)
        self.assertIsInstance(second, lsm.StateSnapshot)
        self.assertEqual(first.snapshot_id, "7")
        self.assertEqual(second.snapshot_id, "8")
        self.assertEqual(first.lp_problem.problem_id, "7")
        self.assertEqual(second.lp_problem.problem_id, "8")

        result = lrs.solve_and_extract(first.lp_problem)
        self.assertIsInstance(result, lrs.SchedulingSuccess)
        self.assertEqual(result.problem_id, "7")
        self.assertEqual(result.plan.problem_id, "7")
        self.assertIs(
            lrs.validate_integer_plan(first.lp_problem, result.plan),
            result.plan,
        )

        # A plan from one scheduling decision must not validate against the
        # problem mapped for a different decision.
        mismatch = lrs.validate_integer_plan(second.lp_problem, result.plan)
        self.assertIsInstance(mismatch, lrs.Failure)
        self.assertEqual(mismatch.category, "snapshot_mismatch")
        self.assertEqual(mismatch.problem_id, "8")

    # -- corrected resource and policy inputs ----------------------------------

    def _resident(self, holder, raw_id, prompt, processed, outputs=0,
                  arrival=1.0):
        """Allocate a resident through the real manager and native updates."""
        seq = _make_sequence(raw_id, prompt, arrival)
        seq.set_status(SequenceStatus.RUNNING)
        holder.block_manager.allocate(seq)
        seq.update_prompt_tokens_processed(processed)
        for token in range(outputs):
            seq.append_token_id(1000 + token)
        seq.set_status(SequenceStatus.PAUSED)
        holder.running.append(seq)
        return seq

    def test_full_context_rounding_recovery_and_decode_demand(self):
        holder, block_manager = _build_holder()
        for raw_id, tokens in ((0, 4), (1, 5), (2, 8), (3, 9)):
            holder.waiting.append(_make_sequence(raw_id, tokens, 1.0))
        # Decode gap 0 (four prompt tokens in one block) and gap 1 (one
        # sampled output token opened a second logical block).
        self._resident(holder, 10, 4, 4)
        self._resident(holder, 11, 4, 4, outputs=1)
        self._resident(holder, 12, 7, 3)
        before = _fingerprint(holder)
        result = _map(holder, lsm.RequestUtility(1.0, 1.0, 1.0))
        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(result, lsm.StateSnapshot, result)
        by_id = {r.raw_seq_id: r for r in result.requests}
        # Full-context admission charge with native block rounding.
        self.assertEqual([by_id[i].prefill_fixed_charge for i in range(4)],
                         [1, 2, 2, 3])
        self.assertEqual([by_id[i].context_len for i in range(4)],
                         [4, 5, 8, 9])
        self.assertEqual(
            (by_id[10].decode_demand, by_id[10].logical_block_count,
             by_id[10].physical_block_count, by_id[10].output_len),
            (0, 1, 1, 0))
        self.assertEqual(
            (by_id[11].decode_demand, by_id[11].logical_block_count,
             by_id[11].physical_block_count, by_id[11].output_len),
            (1, 2, 1, 1))
        # Resident partial prefill: zero charge, recovery = its table.
        self.assertEqual(
            (by_id[12].prefill_fixed_charge, by_id[12].preemption_recovery,
             by_id[12].prefill_eligible, by_id[12].decode_eligible),
            (0, 2, True, False))
        self.assertEqual(by_id[11].preemption_recovery, 1)
        self.assertEqual(result.free_physical_blocks, NUM_GPU_BLOCKS - 4)
        self.assertEqual(result.resident_count, 3)

    def test_unsupported_gap_and_malformed_allocation(self):
        def holder_with(mutate):
            holder, block_manager = _build_holder()
            seq = self._resident(holder, 1, 4, 4)
            holder.waiting.append(_make_sequence(0, 4, 1.0))
            mutate(holder, block_manager, seq)
            return holder

        def gap_two(holder, manager, seq):
            seq.set_status(SequenceStatus.RUNNING)
            for token in range(5):  # 9 tokens: three logical, one physical
                seq.append_token_id(token)
            seq.set_status(SequenceStatus.PAUSED)

        def repeated_block(holder, manager, seq):
            table = manager.block_tables[1]
            table.append(table[0])

        def shared_block(holder, manager, seq):
            self._resident(holder, 2, 4, 4)
            manager.block_tables[2][0] = manager.block_tables[1][0]

        def pool_mismatch(holder, manager, seq):
            manager.gpu_allocator.free_blocks.pop()

        def prefill_gap(holder, manager, seq):
            self._resident(holder, 3, 8, 2)
            manager.gpu_allocator.free(manager.block_tables[3].pop())

        cases = [
            ("decode gap 2", gap_two, "outside {0, 1}"),
            ("repeated block", repeated_block, "repeats a block"),
            ("shared block", shared_block, "shared by two resident"),
            ("pool mismatch", pool_mismatch, "differ from the total pool"),
            ("partial-prefill gap", prefill_gap, "gap 1 must be 0"),
        ]
        for name, mutate, reason in cases:
            with self.subTest(name):
                holder = holder_with(mutate)
                before = _fingerprint(holder)
                result = _map(holder, lsm.RequestUtility(1.0, 1.0, 1.0))
                self.assertEqual(_fingerprint(holder), before)
                self.assertIsInstance(result, lsm.MappingFailure)
                self.assertIn(reason, result.reason)

    def test_resident_ledger_and_inadmissible_registered_context(self):
        # A finished request left in running is not a valid resident ledger.
        holder, _ = _build_holder()
        seq = self._resident(holder, 1, 4, 4)
        seq.set_status(SequenceStatus.FINISHED_STOPPED)
        result = _map(holder, lsm.RequestUtility(1.0, 1.0, 1.0))
        self.assertIsInstance(result, lsm.MappingFailure)
        self.assertIn("running holds a finished request", result.reason)

        # A recomputation-expanded waiting context beyond max_model_len.
        holder, _ = _build_holder()
        expanded = self._resident(holder, 1, 30, 30, outputs=3)
        holder.running.remove(expanded)
        holder.block_manager.free(expanded)
        expanded.reset_for_recompute()
        holder.waiting.append(expanded)
        result = _map(holder, lsm.RequestUtility(1.0, 1.0, 1.0))
        self.assertIsInstance(result, lsm.MappingFailure)
        self.assertIn("registered context length 33 exceeds max_model_len 32",
                      result.reason)

        # Context length 32 == max_model_len and 8 blocks + watermark 2 ==
        # pool 10 are accepted (equality); watermark 3 is rejected. The
        # mapped problem carries that watermark.
        for watermark, ok in ((2, True), (3, False)):
            holder, manager = _build_holder()
            manager.watermark_blocks = watermark
            holder.waiting.append(_make_sequence(0, 32, 1.0))
            result = _map(holder, lsm.RequestUtility(1.0, 1.0, 1.0))
            if ok:
                self.assertIsInstance(result, lsm.StateSnapshot, result)
                self.assertEqual(result.lp_problem.admission_watermark, 2)
            else:
                self.assertIsInstance(result, lsm.MappingFailure)
                self.assertIn("plus watermark 3 exceeds the total pool of 10",
                              result.reason)

    def test_future_head_does_not_hide_arrived_work(self):
        holder, _ = _build_holder()
        holder.waiting.append(_make_sequence(0, 4, SNAPSHOT_TIME + 1))
        holder.waiting.append(_make_sequence(1, 4, 1.0))
        result = _map(holder, lsm.RequestUtility(1.0, 1.0, 1.0))
        self.assertIsInstance(result, lsm.StateSnapshot, result)
        self.assertEqual([r.raw_seq_id for r in result.requests], [1])
        self.assertEqual(result.arrived_waiting_order, (1,))
        self.assertEqual(result.requests[0].queue_position, 1)
        self.assertEqual(result.requests[0].service_rank, 0)

    def test_one_capture_and_state_dependent_policy_inputs(self):
        """A compact state-dependent coefficient fixture (not a production
        utility policy): decode utility grows with generated output and the
        penalty with recovered blocks. Ranks and coefficients come from the
        one capture, are fixed in the snapshot, and change in the next
        decision after the queues change."""
        seen = []

        def policy(captured, request):
            seen.append((type(captured), type(request)))
            return lsm.RequestUtility(
                decode_utility=1.0 + request.output_len,
                prefill_token_utility=0.5,
                preemption_penalty=float(request.preemption_recovery),
            )

        holder, block_manager = _build_holder()
        self._resident(holder, 1, 4, 4)
        second = self._resident(holder, 2, 4, 4)
        holder.waiting.append(_make_sequence(0, 4, 1.0))
        with mock.patch.object(
            lsm, "_capture", wraps=lsm._capture,
        ) as capture_spy:
            snapshot = _map(holder, policy)
        self.assertEqual(capture_spy.call_count, 1)
        self.assertIsInstance(snapshot, lsm.StateSnapshot, snapshot)
        self.assertEqual(set(seen), {(lsm.CapturedState, lsm.CapturedRequest)})
        coeffs = {r.raw_seq_id: r.utility for r in snapshot.requests}
        ranks = {r.raw_seq_id: (r.service_rank, r.victim_rank)
                 for r in snapshot.requests}
        self.assertEqual(coeffs[1], lsm.RequestUtility(1.0, 0.5, 1.0))
        self.assertEqual(ranks, {1: (0, 1), 2: (1, 0), 0: (2, None)})
        frozen = snapshot.lp_problem

        # Next decision: second generated a token and moved to the front.
        second.set_status(SequenceStatus.RUNNING)
        block_manager.append_slot(second)
        second.append_token_id(5)
        second.set_status(SequenceStatus.PAUSED)
        holder.running.reverse()
        holder._iteration_id += 1
        later = _map(holder, policy)
        self.assertIsInstance(later, lsm.StateSnapshot, later)
        later_coeffs = {r.raw_seq_id: r.utility for r in later.requests}
        later_ranks = {r.raw_seq_id: (r.service_rank, r.victim_rank)
                       for r in later.requests}
        self.assertEqual(later_coeffs[2], lsm.RequestUtility(2.0, 0.5, 1.0))
        self.assertEqual(later_coeffs[1], coeffs[1])
        self.assertEqual(later_ranks, {2: (0, 1), 1: (1, 0), 0: (2, None)})
        # The earlier decision's inputs are unchanged.
        self.assertEqual(snapshot.lp_problem, frozen)
        self.assertEqual({r.raw_seq_id: r.utility for r in snapshot.requests},
                         coeffs)

        # Uniform and per-request mappings remain accepted policies.
        uniform = _map(holder, lsm.RequestUtility(3.0, 2.0, 1.0))
        self.assertEqual({r.utility for r in uniform.requests},
                         {lsm.RequestUtility(3.0, 2.0, 1.0)})
        self.assertEqual(
            lsm.construct_coefficients(
                lsm.capture_scheduler_state(holder,
                                            snapshot_time=SNAPSHOT_TIME),
                {0: lsm.RequestUtility(0.0, 1.0, 0.0),
                 1: lsm.RequestUtility(1.0, 0.0, 2.0),
                 2: lsm.RequestUtility(2.0, 0.0, 3.0)}),
            (lsm.RequestUtility(0.0, 1.0, 0.0),
             lsm.RequestUtility(1.0, 0.0, 2.0),
             lsm.RequestUtility(2.0, 0.0, 3.0)))

    # -- rank-based utility policy ------------------------------------------------

    @staticmethod
    def _rank_policy(decode_scale, prefill_scale, penalty_scale):
        return functools.partial(
            lsm.rank_based_utility, decode_scale=decode_scale,
            prefill_scale=prefill_scale, penalty_scale=penalty_scale,
        )

    @staticmethod
    def _triples(snapshot):
        return {r.raw_seq_id: (r.utility.decode_utility,
                               r.utility.prefill_token_utility,
                               r.utility.preemption_penalty)
                for r in snapshot.requests}

    def test_rank_based_utility_values_and_compact_positions(self):
        # Raw IDs deliberately disagree with queue positions. Running: 5
        # (partial prefill), 3 (decode). Waiting: future head 0, then 9 and
        # 2. Distinct scales D=2, P=3, G=5 expose any mix-up.
        holder, _ = _build_holder()
        self._resident(holder, 5, 6, 2)
        self._resident(holder, 3, 4, 4)
        holder.waiting.append(_make_sequence(0, 4, SNAPSHOT_TIME + 1))
        holder.waiting.append(_make_sequence(9, 4, 1.0))
        holder.waiting.append(_make_sequence(2, 4, 1.0))
        bound = self._rank_policy(2.0, 3.0, 5.0)
        seen = []

        def policy(captured, request):
            seen.append((type(captured), type(request)))
            return bound(captured, request)

        before = _fingerprint(holder)
        with mock.patch.object(lsm, "_capture", wraps=lsm._capture) as spy:
            snapshot = _map(holder, policy)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(snapshot, lsm.StateSnapshot, snapshot)
        self.assertEqual(set(seen), {(lsm.CapturedState, lsm.CapturedRequest)})

        # Compact service positions r: 5->0, 3->1, 9->2, 2->3 (the future
        # head is excluded); running positions s: 5->0, 3->1. Waiting
        # requests have no victim penalty.
        triples = self._triples(snapshot)
        self.assertEqual(triples, {
            5: (2.0, 3.0, 5.0),
            3: (1.0, 1.5, 2.5),
            9: (2.0 / 3, 1.0, 0.0),
            2: (0.5, 0.75, 0.0),
        })
        by_id = {r.raw_seq_id: r for r in snapshot.requests}
        # Raw waiting indices include the future head; policy positions do not.
        self.assertEqual((by_id[9].queue_position, by_id[2].queue_position), (1, 2))
        for r in snapshot.requests:
            self.assertEqual(r.utility.decode_utility, 2.0 / (1 + r.service_rank))
            self.assertEqual(r.utility.prefill_token_utility,
                             3.0 / (1 + r.service_rank))
        # Gamma decreases toward the running tail and uses the forward
        # running position, not the reverse victim rank.
        self.assertEqual((by_id[5].victim_rank, by_id[3].victim_rank), (1, 0))
        self.assertGreater(triples[5][2], triples[3][2])
        self.assertEqual(
            {r.request_id: (r.decode_utility, r.prefill_token_utility,
                            r.preemption_penalty)
             for r in snapshot.lp_problem.requests},
            {str(k): v for k, v in triples.items()},
        )

        # Construction reads only the capture: with the live holder emptied
        # and its block manager removed, the same capture yields the same
        # coefficients.
        captured = lsm.capture_scheduler_state(holder, snapshot_time=SNAPSHOT_TIME)
        holder.waiting.clear()
        holder.running.clear()
        holder.block_manager = None
        coefficients = lsm.construct_coefficients(captured, bound)
        self.assertEqual(
            {r.raw_seq_id: c for r, c in zip(captured.requests, coefficients)},
            {r.raw_seq_id: r.utility for r in snapshot.requests},
        )

    def test_rank_based_utility_follows_state_between_decisions(self):
        holder, block_manager = _build_holder()
        resident = self._resident(holder, 1, 4, 4)
        waiting = {k: _make_sequence(k, 4, 1.0) for k in (0, 2)}
        holder.waiting.extend([waiting[0], waiting[2]])
        policy = self._rank_policy(1.0, 1.0, 1.0)

        def decide():
            holder._iteration_id += 1
            snapshot = _map(holder, policy)
            self.assertIsInstance(snapshot, lsm.StateSnapshot, snapshot)
            return snapshot

        first = decide()
        self.assertEqual(self._triples(first), {
            1: (1.0, 1.0, 1.0), 0: (0.5, 0.5, 0.0), 2: (1 / 3, 1 / 3, 0.0)})
        recorded = [(first, self._triples(first), first.lp_problem.requests)]

        # Admission: 0 joins the running tail (native allocation and status
        # transitions) and gains a penalty; 2's position is unchanged.
        seq = waiting[0]
        holder.waiting.remove(seq)
        seq.set_status(SequenceStatus.RUNNING)
        block_manager.allocate(seq)
        seq.update_prompt_tokens_processed(4)
        seq.set_status(SequenceStatus.PAUSED)
        holder.running.append(seq)
        second = decide()
        self.assertEqual(self._triples(second), {
            1: (1.0, 1.0, 1.0), 0: (0.5, 0.5, 0.5), 2: (1 / 3, 1 / 3, 0.0)})
        recorded.append((second, self._triples(second), second.lp_problem.requests))

        # Resident removal: 1 leaves; 0 moves to the head, 2 moves up.
        holder.running.remove(resident)
        block_manager.free(resident)
        resident.set_status(SequenceStatus.FINISHED_STOPPED)
        third = decide()
        self.assertEqual(self._triples(third), {
            0: (1.0, 1.0, 1.0), 2: (0.5, 0.5, 0.0)})
        recorded.append((third, self._triples(third), third.lp_problem.requests))

        # Queue-position change by victim reinsertion at the waiting front:
        # 0 keeps service position 0 but, now waiting, has no penalty.
        holder.running.remove(seq)
        block_manager.free(seq)
        seq.reset_for_recompute()
        holder.waiting.insert(0, seq)
        fourth = decide()
        self.assertEqual(self._triples(fourth), {
            0: (1.0, 1.0, 0.0), 2: (0.5, 0.5, 0.0)})

        # Earlier decisions keep their own coefficients and problems.
        for snapshot, triples, problem_requests in recorded:
            self.assertEqual(self._triples(snapshot), triples)
            self.assertEqual(snapshot.lp_problem.requests, problem_requests)
        self.assertEqual(
            [s.snapshot_id for s, _, _ in recorded] + [fourth.snapshot_id],
            [str(ITERATION_ID + k) for k in (1, 2, 3, 4)],
        )

    def test_rank_based_utility_rejects_invalid_scales(self):
        for field in ("decode_scale", "prefill_scale", "penalty_scale"):
            for bad in (0, 0.0, -1.0, float("nan"), float("inf"),
                        float("-inf"), True, False, "1.0", None, 10 ** 400):
                with self.subTest(field=field, bad=bad):
                    scales = dict(decode_scale=1.0, prefill_scale=1.0,
                                  penalty_scale=1.0)
                    scales[field] = bad
                    holder = _build_three_request_holder()
                    before = _fingerprint(holder)
                    with mock.patch.object(
                        lsm.lrs, "LPProblem", wraps=lsm.lrs.LPProblem,
                    ) as problem_spy:
                        result = _map(holder, functools.partial(
                            lsm.rank_based_utility, **scales))
                    self.assertEqual(_fingerprint(holder), before)
                    self.assertIsInstance(result, lsm.MappingFailure)
                    self.assertEqual(result.stage, "state_mapping")
                    self.assertEqual(result.category, "mapping_failure")
                    self.assertIn(f"rank-based utility {field}", result.reason)
                    problem_spy.assert_not_called()

    # -- coordinated rank utility policy ---------------------------------------

    @staticmethod
    def _coordinated(c_max=C_MAX, b_max=B_MAX, rank_base=0.25):
        return functools.partial(lsm.coordinated_rank_utility,
                                 rank_base=rank_base, c_max=c_max, b_max=b_max)

    def test_coordinated_utility_values_ranks_and_chunk_normalization(self):
        # Running: 5 (partial prefill, 2 of 6 left), 3 (decode). Waiting:
        # future head 0, then 9 (10 left) and 2 (4 left). Positions r:
        # 5->0, 3->1, 9->2, 2->3, so w = 1, 1/4, 1/16, 1/64.
        holder, _ = _build_holder()
        self._resident(holder, 5, 6, 4)
        self._resident(holder, 3, 4, 4)
        holder.waiting.append(_make_sequence(0, 4, SNAPSHOT_TIME + 1))
        holder.waiting.append(_make_sequence(9, 10, 1.0))
        holder.waiting.append(_make_sequence(2, 4, 1.0))
        before = _fingerprint(holder)
        with mock.patch.object(lsm, "_capture", wraps=lsm._capture) as spy:
            snapshot = _map(holder, self._coordinated())
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(snapshot, lsm.StateSnapshot, snapshot)
        # q = min(remaining, C_max = 4, B_max = 8): 2 (remainder binds) for
        # 5, 4 (chunk cap binds) for 9, 4 for 2. Decode and penalty are w;
        # ineligible terms are exactly 0.
        self.assertEqual(self._triples(snapshot), {
            5: (0.0, 1.0 / 2, 1.0),
            3: (0.25, 0.0, 0.25),
            9: (0.0, (1 / 16) / 4, 0.0),
            2: (0.0, (1 / 64) / 4, 0.0),
        })
        by_id = {r.raw_seq_id: r for r in snapshot.requests}
        # A complete available chunk earns the same value as one decode.
        for raw_id, q in ((5, 2), (9, 4), (2, 4)):
            r = by_id[raw_id]
            self.assertEqual(r.utility.prefill_token_utility * q,
                             0.25 ** r.service_rank)
        # Ranks come from the unchanged profile; the victim rank is separate.
        self.assertEqual({k: (r.service_rank, r.victim_rank)
                          for k, r in by_id.items()},
                         {5: (0, 1), 3: (1, 0), 9: (2, None), 2: (3, None)})
        self.assertEqual(
            {r.request_id: (r.decode_utility, r.prefill_token_utility,
                            r.preemption_penalty)
             for r in snapshot.lp_problem.requests},
            {str(k): v for k, v in self._triples(snapshot).items()})
        # The token budget also binds q when it is the smallest limit.
        captured = lsm.capture_scheduler_state(holder,
                                               snapshot_time=SNAPSHOT_TIME)
        tight = lsm.construct_coefficients(captured,
                                           self._coordinated(b_max=3))
        self.assertEqual(
            {r.raw_seq_id: c.prefill_token_utility
             for r, c in zip(captured.requests, tight)},
            {5: 1.0 / 2, 3: 0.0, 9: (1 / 16) / 3, 2: (1 / 64) / 3})

    def test_coordinated_utility_is_fixed_by_the_capture(self):
        holder, block_manager = _build_holder()
        self._resident(holder, 1, 4, 4)
        holder.waiting.append(_make_sequence(0, 6, 1.0))
        policy = self._coordinated()
        snapshot = _map(holder, policy)
        self.assertIsInstance(snapshot, lsm.StateSnapshot, snapshot)
        triples = self._triples(snapshot)
        problem = snapshot.lp_problem
        self.assertEqual(triples, {1: (1.0, 0.0, 1.0), 0: (0.0, 0.25 / 4, 0.0)})
        # Construction reads only the capture: with the live holder emptied
        # and its block manager removed, the same capture yields the same
        # coefficients, and the earlier snapshot and problem are unchanged.
        captured = lsm.capture_scheduler_state(holder,
                                               snapshot_time=SNAPSHOT_TIME)
        holder.waiting.clear()
        holder.running.clear()
        holder.block_manager = None
        self.assertEqual(
            {r.raw_seq_id: c for r, c in zip(
                captured.requests,
                lsm.construct_coefficients(captured, policy))},
            {r.raw_seq_id: r.utility for r in snapshot.requests})
        self.assertEqual(self._triples(snapshot), triples)
        self.assertIs(snapshot.lp_problem, problem)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.requests[0].utility.decode_utility = 2.0

    def test_coordinated_utility_rejects_invalid_parameters(self):
        cases = [("rank_base", bad) for bad in (
            0, 0.0, -0.25, 1.5, float("nan"), float("inf"), True, "0.25",
            None)]
        cases += [(name, bad) for name in ("c_max", "b_max")
                  for bad in (0, -1, 1.5, True, None, "4")]
        for field, bad in cases:
            with self.subTest(field=field, bad=bad):
                params = dict(rank_base=0.25, c_max=C_MAX, b_max=B_MAX)
                params[field] = bad
                holder = _build_three_request_holder()
                before = _fingerprint(holder)
                result = _map(holder, functools.partial(
                    lsm.coordinated_rank_utility, **params))
                self.assertEqual(_fingerprint(holder), before)
                self.assertIsInstance(result, lsm.MappingFailure)
                self.assertIn(f"coordinated utility {field}", result.reason)

    # -- reciprocal-square rank utility policy ---------------------------------

    @staticmethod
    def _reciprocal_square(c_max=C_MAX, b_max=B_MAX):
        return functools.partial(lsm.reciprocal_square_rank_utility,
                                 c_max=c_max, b_max=b_max)

    def test_reciprocal_square_values_terms_and_normalization(self):
        # The coordinated test's state: positions 5->0, 3->1, 9->2, 2->3,
        # so w = 1, 1/4, 1/9, 1/16 (geometric 1/4 would give 1/16, 1/64).
        holder, _ = _build_holder()
        self._resident(holder, 5, 6, 4)
        self._resident(holder, 3, 4, 4)
        holder.waiting.append(_make_sequence(0, 4, SNAPSHOT_TIME + 1))
        holder.waiting.append(_make_sequence(9, 10, 1.0))
        holder.waiting.append(_make_sequence(2, 4, 1.0))
        before = _fingerprint(holder)
        with mock.patch.object(lsm, "_capture", wraps=lsm._capture) as spy:
            snapshot = _map(holder, self._reciprocal_square())
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(_fingerprint(holder), before)
        self.assertIsInstance(snapshot, lsm.StateSnapshot, snapshot)
        # q = min(remaining, C_max = 4, B_max = 8); ineligible terms are 0.
        self.assertEqual(self._triples(snapshot), {
            5: (0.0, 1.0 / 2, 1.0),
            3: (0.25, 0.0, 0.25),
            9: (0.0, (1 / 9) / 4, 0.0),
            2: (0.0, (1 / 16) / 4, 0.0),
        })
        by_id = {r.raw_seq_id: r for r in snapshot.requests}
        for raw_id, q in ((5, 2), (9, 4), (2, 4)):
            r = by_id[raw_id]
            self.assertAlmostEqual(r.utility.prefill_token_utility * q,
                                   1 / (1 + r.service_rank) ** 2, places=15)
        self.assertEqual({k: (r.service_rank, r.victim_rank)
                          for k, r in by_id.items()},
                         {5: (0, 1), 3: (1, 0), 9: (2, None), 2: (3, None)})
        self.assertEqual(
            {r.request_id: (r.decode_utility, r.prefill_token_utility,
                            r.preemption_penalty)
             for r in snapshot.lp_problem.requests},
            {str(k): v for k, v in self._triples(snapshot).items()})
        captured = lsm.capture_scheduler_state(holder,
                                               snapshot_time=SNAPSHOT_TIME)
        tight = lsm.construct_coefficients(captured,
                                           self._reciprocal_square(b_max=3))
        self.assertEqual(
            {r.raw_seq_id: c.prefill_token_utility
             for r, c in zip(captured.requests, tight)},
            {5: 1.0 / 2, 3: 0.0, 9: (1 / 9) / 3, 2: (1 / 16) / 3})

    def test_reciprocal_square_later_ranks_and_fixed_capture(self):
        # One decode-ready resident, then 13 arrived waiting requests whose
        # raw IDs run backwards: positions 0..13.
        holder, _ = _build_holder()
        self._resident(holder, 20, 4, 4)
        for raw_id in range(13, 0, -1):
            holder.waiting.append(_make_sequence(raw_id, 8, 1.0))
        policy = self._reciprocal_square()
        snapshot = _map(holder, policy)
        self.assertIsInstance(snapshot, lsm.StateSnapshot, snapshot)
        by_id = {r.raw_seq_id: r for r in snapshot.requests}
        self.assertEqual(by_id[20].utility, lsm.RequestUtility(1.0, 0.0, 1.0))
        for raw_id in range(1, 14):
            r = by_id[raw_id]
            position = 14 - raw_id
            self.assertEqual(r.service_rank, position)
            self.assertEqual(r.utility, lsm.RequestUtility(
                0.0, (1.0 / (1 + position) ** 2) / C_MAX, 0.0))
        # Rank 13: w = 1/196, against the geometric 4^-13 (about 1.5e-8).
        last = by_id[1].utility.prefill_token_utility * C_MAX
        self.assertEqual(last, 1 / 196)
        self.assertGreater(last / 0.25 ** 13, 1e5)
        # Construction reads only the capture.
        captured = lsm.capture_scheduler_state(holder,
                                               snapshot_time=SNAPSHOT_TIME)
        holder.waiting.clear()
        holder.running.clear()
        holder.block_manager = None
        self.assertEqual(
            {r.raw_seq_id: c for r, c in zip(
                captured.requests,
                lsm.construct_coefficients(captured, policy))},
            {r.raw_seq_id: r.utility for r in snapshot.requests})

    def test_reciprocal_square_rejects_invalid_parameters(self):
        for name in ("c_max", "b_max"):
            for bad in (0, -1, 1.5, True, None, "4"):
                with self.subTest(field=name, bad=bad):
                    params = dict(c_max=C_MAX, b_max=B_MAX)
                    params[name] = bad
                    holder = _build_three_request_holder()
                    before = _fingerprint(holder)
                    result = _map(holder, functools.partial(
                        lsm.reciprocal_square_rank_utility, **params))
                    self.assertEqual(_fingerprint(holder), before)
                    self.assertIsInstance(result, lsm.MappingFailure)
                    self.assertIn(f"reciprocal-square utility {name}",
                                  result.reason)
        with self.assertRaises(TypeError):
            lsm.reciprocal_square_rank_utility(None, None, rank_base=0.5,
                                               c_max=C_MAX, b_max=B_MAX)



if __name__ == "__main__":
    unittest.main()
