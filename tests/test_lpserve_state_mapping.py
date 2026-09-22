import dataclasses
import sys
import unittest
from pathlib import Path

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
MEMORY_RESERVE = 1
DECODE_POLICY_ID = "conservative_one_block_v1"

NUMERICAL_POLICY = lrs.NumericalPolicy(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)


class _ConfigHolder:
    """A tiny faithful stand-in for the two scheduler_config fields read."""

    def __init__(self, num_pipeline_stages, max_num_seqs):
        self.num_pipeline_stages = num_pipeline_stages
        self.max_num_seqs = max_num_seqs


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


class LPServeStateMappingTest(unittest.TestCase):
    def test_supported_mapping_and_nonmutation(self):
        block_manager = VLLMBlockSpaceManager(
            BLOCK_SIZE, NUM_GPU_BLOCKS, MAX_MODEL_LEN,
        )
        holder = _SchedulerHolder(
            _ConfigHolder(NUM_PIPELINE_STAGES, RESIDENT_LIMIT),
            block_manager, ITERATION_ID, NUM_RUNNING_BATCHES,
        )

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

        utilities = (
            (0, lsm.RequestUtility(
                decode_utility=0.0, prefill_token_utility=2.0,
                preemption_penalty=0.0,
            )),
            (1, lsm.RequestUtility(
                decode_utility=0.0, prefill_token_utility=1.0,
                preemption_penalty=0.25,
            )),
            (2, lsm.RequestUtility(
                decode_utility=3.0, prefill_token_utility=0.0,
                preemption_penalty=0.5,
            )),
        )

        before = _fingerprint(holder)

        result = lsm.map_scheduler_state(
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
        self.assertEqual(
            tuple(r.decode_charge for r in problem_requests), (0, 0, 1),
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
        self.assertEqual(problem.w, 1)
        self.assertEqual(problem.problem_id, result.snapshot_id)

        revalidated = lrs.validate_problem(problem)
        self.assertIsInstance(revalidated, lrs.LPProblem)

        _assert_no_mutable_or_framework_objects(result)


if __name__ == "__main__":
    unittest.main()
