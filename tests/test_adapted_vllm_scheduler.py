"""Focused CPU checks for the adapted upstream vLLM v0.31.0 reference.

Groups: procedural policy selection and configuration; real central/worker
replay through the native sequence managers; CPU input packing, attention
metadata, hidden-state pruning, and greedy sampling with hand-built tensors.
No model is initialized and no GPU is used.
"""

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import test_lp_scheduler as lpt  # noqa: E402

from sarathi.config import (  # noqa: E402
    AdaptedVLLMSchedulerConfig,
    SchedulerType,
    VLLMSchedulerConfig,
)
from sarathi.core.block_space_manager.block_space_manager_registry import (  # noqa: E402
    BlockSpaceManagerRegistry,
)
from sarathi.core.block_space_manager.vllm_block_space_manager import (  # noqa: E402
    VLLMBlockSpaceManager,
)
from sarathi.core.datatypes.sampling_params import SamplingParams  # noqa: E402
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs  # noqa: E402
from sarathi.core.datatypes.sequence import (  # noqa: E402
    Sequence,
    SequenceScheduleMetadata,
)
from sarathi.core.datatypes.sequence_status import SequenceStatus  # noqa: E402
from sarathi.core.scheduler import adapted_vllm_scheduler as avs  # noqa: E402
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry  # noqa: E402
from sarathi.core.sequence_manager.worker_sequence_manager import (  # noqa: E402
    WorkerSequenceManager,
)
from sarathi.engine import arg_utils, llm_engine  # noqa: E402
from sarathi.engine.arg_utils import EngineArgs  # noqa: E402
from sarathi.model_executor.attention.flash_attention_wrapper import (  # noqa: E402
    FlashAttentionWrapper,
)
from sarathi.model_executor.layers import sampler as sampler_module  # noqa: E402
from sarathi.model_executor.model_runner import ModelRunner  # noqa: E402

BLOCK_SIZE = lpt.BLOCK_SIZE  # 4
MAX_MODEL_LEN = 32
SAMPLED_TOKEN = lpt.SAMPLED_TOKEN

setUpModule = lpt.setUpModule
tearDownModule = lpt.tearDownModule


class _Harness(lpt._Harness):
    """The LP fixture's replay harness with the registered reference: real
    scheduler and central block manager, engine sequence manager sharing its
    sequences, and a worker sequence manager holding separate copies."""

    def __init__(self, num_gpu_blocks, max_num_seqs, max_num_batched_tokens,
                 max_model_len=MAX_MODEL_LEN):
        self.scheduler_config = AdaptedVLLMSchedulerConfig(
            max_num_seqs, max_model_len, 1, max_num_batched_tokens)
        cache_config = lpt._cache_config(num_gpu_blocks)
        self.scheduler = SchedulerRegistry.get(
            SchedulerType.ADAPTED_VLLM, self.scheduler_config, cache_config)
        self.engine_seqs = lpt._EngineSequenceManager()
        self.worker_seqs = WorkerSequenceManager(
            cache_config, self.scheduler_config)
        self.clock = mock.Mock()
        self.calls = []
        # Record native central mutation calls in order.
        sched = self.scheduler
        for name in ("_preempt", "_allocate", "_append_slot"):
            real = getattr(sched, name)

            def wrapper(seq, _real=real, _name=name):
                self.calls.append((_name, seq.seq_id))
                return _real(seq)
            setattr(sched, name, wrapper)

    def add(self, seq_id, num_prompt_tokens, arrival_time, max_tokens,
            token_offset=0):
        seq = Sequence(
            seq_id=seq_id,
            prompt=None,
            prompt_token_ids=list(range(token_offset,
                                        token_offset + num_prompt_tokens)),
            block_size=BLOCK_SIZE,
            eos_token_id=-2,
            arrival_time=arrival_time,
            sampling_params=SamplingParams(ignore_eos=True, max_tokens=max_tokens,
                                           temperature=0.0),
        )
        self.engine_seqs.add_seq(seq)
        self.worker_seqs.add_seq(copy.deepcopy(seq))
        self.scheduler.add_seq(seq)
        return seq

    def schedule(self, now):
        self.calls.clear()
        self.proposals = []
        real_propose = self.scheduler._propose

        def propose(t):
            proposal = real_propose(t)
            self.proposals.append(proposal)
            return proposal
        self.clock.monotonic.return_value = now
        with mock.patch.object(avs, "time", self.clock), \
                mock.patch.object(self.scheduler, "_propose", propose):
            return self.scheduler.schedule()

    def seed(self, entries):
        """Hand-construct a state through native operations: admit or
        continue ``(seq, chunk)`` entries in the given order and replay them.
        Used only for states this policy cannot reach from a fresh start."""
        sched = self.scheduler
        for seq, _ in entries:
            if seq in sched.waiting:
                sched.waiting.remove(seq)
                sched.block_manager.allocate(seq)
                sched.running.append(seq)
        outputs = SchedulerOutputs(
            -1, [], [], [SequenceScheduleMetadata(s.seq_id, c) for s, c in entries])
        sched.num_running_batches = 1
        self.replay(outputs)

    def ids(self, seqs):
        return [s.seq_id for s in seqs]

    def gaps(self):
        manager = self.scheduler.block_manager
        return {s.seq_id: len(s.logical_token_blocks)
                - len(manager.block_tables[s.seq_id])
                for s in self.scheduler.running}


def _assert_conserved(test, harness):
    """Per-manager block conservation, equal per-request table lengths and
    free counts between central and worker managers. Physical IDs are not
    required to agree."""
    pool = harness.scheduler.block_manager.num_total_gpu_blocks
    central, worker = harness.central_tables(), harness.worker_tables()
    test.assertEqual({k: len(v) for k, v in central.items()},
                     {k: len(v) for k, v in worker.items()})
    for manager in (harness.scheduler.block_manager,
                    harness.worker_seqs.block_manager):
        allocated = [b.block_number for t in manager.block_tables.values()
                     for b in t]
        free = [b.block_number for b in manager.gpu_allocator.free_blocks]
        test.assertEqual(sorted(allocated + free), list(range(pool)))
    test.assertEqual(
        harness.scheduler.block_manager.get_num_free_gpu_blocks(),
        harness.worker_seqs.block_manager.get_num_free_gpu_blocks())
    for seq in harness.scheduler.running:
        test.assertEqual(lpt._seq_fingerprint(seq), lpt._seq_fingerprint(
            harness.worker_seqs.get_seq(seq.seq_id)))


# ---------------------------------------------------------------------------
# Configuration and registries
# ---------------------------------------------------------------------------


class ConfigurationTest(unittest.TestCase):

    def test_direct_configuration_and_registries(self):
        config = AdaptedVLLMSchedulerConfig(2, 16, 1, 8)
        self.assertEqual(config.type, SchedulerType.ADAPTED_VLLM)
        self.assertEqual(SchedulerType.from_str("adapted_vllm"),
                         SchedulerType.ADAPTED_VLLM)
        self.assertEqual((config.max_num_seqs, config.max_model_len,
                          config.num_pipeline_stages,
                          config.max_num_batched_tokens), (2, 16, 1, 8))
        # Chunked prefill: B below max_model_len is accepted.
        self.assertLess(config.max_num_batched_tokens, config.max_model_len)
        for attr in ("chunk_size", "c_max", "s_max", "watermark"):
            self.assertFalse(hasattr(config, attr))
        scheduler = SchedulerRegistry.get(
            SchedulerType.ADAPTED_VLLM, config, lpt._cache_config(8))
        self.assertIsInstance(scheduler, avs.AdaptedVLLMScheduler)
        self.assertIs(type(scheduler.block_manager), VLLMBlockSpaceManager)
        self.assertIs(type(BlockSpaceManagerRegistry.get(
            SchedulerType.ADAPTED_VLLM, BLOCK_SIZE, 8, 16)),
            VLLMBlockSpaceManager)
        # The inherited vLLM configuration is unchanged.
        self.assertEqual(SchedulerType.VLLM.value, 1)
        self.assertEqual(SchedulerType.LP.value, 8)
        with self.assertRaises(AssertionError):
            VLLMSchedulerConfig(2, 16, 1, 8)

    def test_invalid_values_are_rejected(self):
        cases = [
            dict(num_pipeline_stages=2), dict(max_num_batched_tokens=1),
            dict(max_num_seqs=0), dict(max_model_len=0),
            dict(max_num_batched_tokens=None), dict(max_num_batched_tokens=True),
            dict(max_num_seqs=2.0),
        ]
        for override in cases:
            values = dict(max_num_seqs=2, max_model_len=16,
                          num_pipeline_stages=1, max_num_batched_tokens=8)
            values.update(override)
            with self.subTest(**override):
                with self.assertRaises(ValueError):
                    AdaptedVLLMSchedulerConfig(**values)


class EngineArgsTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output_dir = os.path.join(tmp.name, "run")
        patcher = mock.patch.object(
            arg_utils, "ModelConfig",
            side_effect=lambda **_: lpt_engine_model_config())
        self.model_config_cls = patcher.start()
        self.addCleanup(patcher.stop)

    def kwargs(self, **overrides):
        values = dict(model="unused-model-name", scheduler_type="adapted_vllm",
                      max_num_seqs=2, max_model_len=16,
                      max_num_batched_tokens=8, output_dir=self.output_dir)
        values.update(overrides)
        return values

    def test_selection_through_engine_args_without_model_initialization(self):
        config = EngineArgs(**self.kwargs()).create_engine_configs()[3]
        self.assertIs(type(config), AdaptedVLLMSchedulerConfig)
        self.assertEqual((config.max_num_seqs, config.max_model_len,
                          config.max_num_batched_tokens), (2, 16, 8))
        with mock.patch.object(llm_engine, "BaseLLMEngine") as base_engine, \
                mock.patch.object(llm_engine, "PipelineParallelLLMEngine") as pp:
            engine = llm_engine.LLMEngine.from_engine_args(**self.kwargs())
        pp.assert_not_called()
        self.assertIs(engine, base_engine.return_value)
        self.assertIs(type(base_engine.call_args.args[3]),
                      AdaptedVLLMSchedulerConfig)

    def test_missing_budget_and_multi_stage_fail_before_model(self):
        for override in (dict(max_num_batched_tokens=None),
                         dict(pipeline_parallel_size=2)):
            with self.subTest(**override):
                with self.assertRaises(ValueError):
                    EngineArgs(**self.kwargs(**override))
                self.model_config_cls.assert_not_called()
                self.assertFalse(os.path.exists(
                    os.path.join(self.output_dir, "config.yml")))


def lpt_engine_model_config():
    model_config = mock.Mock()
    model_config.get_max_model_len.return_value = 16
    model_config.get_total_num_layers.return_value = 2
    return model_config


# ---------------------------------------------------------------------------
# Procedural policy selection (real scheduler, native replay)
# ---------------------------------------------------------------------------


class PolicySelectionTest(unittest.TestCase):

    def test_running_order_scan_with_partial_prefill_ahead_of_decode(self):
        # Hand-constructed: P (partial, 4 of 16 processed) ahead of decode D.
        # This order is unreachable from a fresh start under this policy.
        h = _Harness(num_gpu_blocks=8, max_num_seqs=3, max_num_batched_tokens=8)
        p = h.add(0, 16, 1.0, 4)
        d = h.add(1, 4, 1.0, 4)
        h.add(2, 4, 1.0, 4)
        h.seed([(p, 4), (d, 4)])
        self.assertEqual(h.ids(h.scheduler.running), [0, 1])
        self.assertTrue(d.prompt_processing_finished)
        outputs = h.step(10.0)
        # P takes the whole budget B=8 (remaining 12); D is omitted and W is
        # not reached. A decodes-first policy would emit D:0 and P:7.
        prop = h.proposals[0]
        self.assertEqual(lpt._emitted(outputs), [(0, 8)])
        self.assertEqual(prop.running_stop_reason, avs.RUNNING_BUDGET_EXHAUSTED)
        self.assertEqual(prop.waiting_stop_reason, avs.WAITING_BUDGET_EXHAUSTED)
        self.assertEqual(h.ids(h.scheduler.running), [0, 1])
        self.assertEqual(h.ids(h.scheduler.waiting), [2])
        # P completes its prompt and keeps its position after the phase
        # change; W is admitted to the tail with the remaining budget.
        outputs = h.step(11.0)
        self.assertEqual(lpt._emitted(outputs), [(0, 4), (2, 3), (1, 0)])
        self.assertEqual(h.ids(h.scheduler.running), [0, 1, 2])
        self.assertEqual(h.calls, [("_allocate", 2), ("_append_slot", 1)])
        outputs = h.step(12.0)
        self.assertEqual(lpt._emitted(outputs), [(2, 1), (0, 0), (1, 0)])
        self.assertEqual(h.ids(h.scheduler.running), [0, 1, 2])
        _assert_conserved(self, h)

    def test_budget_chunking_without_chunk_or_action_cap(self):
        h = _Harness(num_gpu_blocks=16, max_num_seqs=6, max_num_batched_tokens=16)
        for i in range(4):
            h.add(i, 4, 1.0, 3)
        h.add(4, 12, 1.0, 3)
        h.add(5, 10, 1.0, 3)
        outputs = h.step(10.0)
        # Four 4-token admissions use the full budget 16.
        self.assertEqual(lpt._emitted(outputs), [(i, 4) for i in range(4)])
        self.assertEqual(h.proposals[0].waiting_stop_reason,
                         avs.WAITING_BUDGET_EXHAUSTED)
        outputs = h.step(11.0)
        # Four decodes (1 token each) and a 12-token admission chunk use the
        # budget before request 5. Five actions in one step; no per-request
        # chunk cap.
        self.assertEqual(lpt._emitted(outputs),
                         [(4, 12), (0, 0), (1, 0), (2, 0), (3, 0)])
        outputs = h.step(12.0)
        self.assertEqual(lpt._emitted(outputs),
                         [(5, 10), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        self.assertEqual(outputs.num_batched_tokens, 15)
        # Full-context allocation of the 10-token admission (3 blocks).
        self.assertEqual(len(h.central_tables()[5]), 3)
        _assert_conserved(self, h)

    def test_tail_eviction_multiple_victims_and_no_admission_after(self):
        h = _Harness(num_gpu_blocks=4, max_num_seqs=4, max_num_batched_tokens=16)
        seqs = [h.add(i, 4, 1.0, 3) for i in range(4)]
        h.step(10.0)  # four admissions, pool exhausted
        h.step(11.0)  # four zero-demand decodes; every gap becomes 1
        self.assertEqual(h.gaps(), {0: 1, 1: 1, 2: 1, 3: 1})
        late = h.add(9, 1, 1.0, 1)
        outputs = h.step(12.0)
        prop = h.proposals[0]
        # 0 needs a block: evict tail 3. 1 needs a block: evict new tail 2.
        self.assertEqual(prop.victims, (3, 2))
        self.assertEqual(outputs.preempted_seq_ids, [3, 2])
        self.assertEqual(lpt._emitted(outputs), [(0, 0), (1, 0)])
        self.assertEqual(prop.running_stop_reason, avs.RUNNING_EXHAUSTED)
        self.assertEqual(prop.waiting_stop_reason,
                         avs.WAITING_SKIPPED_AFTER_PREEMPTION)
        self.assertEqual(h.calls, [("_preempt", 3), ("_preempt", 2),
                                   ("_append_slot", 0), ("_append_slot", 1)])
        # Native front insertion: reverse eviction order ahead of the queue.
        self.assertEqual(h.ids(h.scheduler.waiting), [2, 3, 9])
        self.assertEqual(h.ids(h.scheduler.running), [0, 1])
        # Recomputation reset after replay: outputs folded into the prompt.
        for s in (seqs[2], seqs[3]):
            self.assertEqual((s.get_prompt_len(), s.get_output_len(),
                              s.get_num_prompt_tokens_processed(),
                              s.get_status()),
                             (5, 0, 0, SequenceStatus.WAITING))
        _assert_conserved(self, h)

    def test_self_eviction_preserves_accepted_work(self):
        h = _Harness(num_gpu_blocks=3, max_num_seqs=2, max_num_batched_tokens=16)
        x = h.add(0, 5, 1.0, 4)
        y = h.add(1, 4, 1.0, 4)
        h.step(10.0)
        h.step(11.0)
        self.assertEqual(h.gaps(), {0: 0, 1: 1})
        outputs = h.step(12.0)
        prop = h.proposals[0]
        # X (zero demand) is accepted; Y needs a block, is the tail, and
        # evicts itself; X is never a victim and still executes.
        self.assertEqual(prop.victims, (1,))
        self.assertEqual(prop.running_stop_reason, avs.RUNNING_SELF_EVICTED)
        self.assertEqual(lpt._emitted(outputs), [(0, 0)])
        self.assertEqual(outputs.preempted_seq_ids, [1])
        self.assertEqual(h.ids(h.scheduler.running), [0])
        self.assertEqual(h.ids(h.scheduler.waiting), [1])
        self.assertEqual(x.get_output_len(), 2)
        self.assertEqual(y.get_prompt_len(), 5)
        _assert_conserved(self, h)

    def test_resident_limit_retained_order_and_tail_admission(self):
        h = _Harness(num_gpu_blocks=8, max_num_seqs=2, max_num_batched_tokens=8)
        h.add(0, 3, 1.0, 3)
        h.add(1, 3, 1.0, 3)
        h.add(2, 3, 1.0, 3)
        outputs = h.step(10.0)
        self.assertEqual(lpt._emitted(outputs), [(0, 3), (1, 3)])
        self.assertEqual(h.proposals[0].waiting_stop_reason,
                         avs.WAITING_RESIDENT_LIMIT)
        outputs = h.step(11.0)
        self.assertEqual(lpt._emitted(outputs), [(0, 0), (1, 0)])
        self.assertEqual(h.proposals[0].waiting_stop_reason,
                         avs.WAITING_RESIDENT_LIMIT)
        h.step(12.0)
        outputs = h.step(13.0)  # both finish (3 tokens each)
        self.assertEqual(h.ids(h.scheduler.running), [])
        outputs = h.step(14.0)
        self.assertEqual(lpt._emitted(outputs), [(2, 3)])
        self.assertEqual(h.ids(h.scheduler.running), [2])

    def test_arrived_work_behind_future_dated_entry(self):
        h = _Harness(num_gpu_blocks=8, max_num_seqs=3, max_num_batched_tokens=8)
        h.add(0, 4, 50.0, 1)   # registered before arrival
        h.add(1, 4, 1.0, 1)
        h.add(2, 4, 2.0, 1)
        outputs = h.step(10.0)
        prop = h.proposals[0]
        self.assertEqual(prop.arrived_waiting_order, (1, 2))
        self.assertEqual(lpt._emitted(outputs), [(1, 4), (2, 4)])
        self.assertEqual(h.ids(h.scheduler.waiting), [0])
        self.assertEqual(h.ids(h.scheduler.running), [1, 2])


# ---------------------------------------------------------------------------
# Real central/worker replay of resource rules
# ---------------------------------------------------------------------------


class NativeReplayTest(unittest.TestCase):

    def test_zero_demand_decode_with_admission_of_last_free_block(self):
        # Research-context section 21 counterexample: one free block, zero
        # watermark, resident zero-demand decode, admission charging one.
        h = _Harness(num_gpu_blocks=2, max_num_seqs=2, max_num_batched_tokens=8)
        d = h.add(0, 4, 1.0, 3)
        w = h.add(1, 4, 5.0, 3)
        h.step(1.0)
        manager = h.scheduler.block_manager
        self.assertEqual((manager.get_num_free_gpu_blocks(),
                          manager.watermark_blocks, h.gaps()), (1, 0, {0: 0}))
        self.assertTrue(manager.can_append_slot())
        outputs = h.schedule(10.0)
        prop = h.proposals[0]
        self.assertEqual([(a.seq_id, a.kind, a.block_demand)
                          for a in prop.actions],
                         [(0, avs.ACTION_DECODE, 0),
                          (1, avs.ACTION_ADMISSION, 1)])
        # Commit and metadata: admission (prefill) first, then the decode.
        self.assertEqual(h.calls, [("_allocate", 1), ("_append_slot", 0)])
        self.assertEqual(lpt._emitted(outputs), [(1, 4), (0, 0)])
        self.assertEqual(manager.get_num_free_gpu_blocks(), 0)
        h.replay(outputs)
        self.assertEqual(h.worker_seqs.block_manager.get_num_free_gpu_blocks(), 0)
        self.assertEqual((d.get_output_len(), w.prompt_processing_finished),
                         (1, True))
        self.assertEqual(h.ids(h.scheduler.running), [0, 1])
        _assert_conserved(self, h)

    def test_standalone_zero_demand_decode_then_control_only_report(self):
        h = _Harness(num_gpu_blocks=1, max_num_seqs=1, max_num_batched_tokens=8)
        d = h.add(0, 4, 1.0, 3)
        h.step(1.0)
        manager = h.scheduler.block_manager
        self.assertEqual(manager.get_num_free_gpu_blocks(), 0)
        self.assertFalse(manager.can_append_slot())
        outputs = h.step(2.0)  # zero-demand decode with zero free blocks
        self.assertEqual(lpt._emitted(outputs), [(0, 0)])
        self.assertEqual(d.get_output_len(), 1)
        self.assertEqual(h.gaps(), {0: 1})
        # Now the lone resident needs a block it can never get: the policy
        # evicts it and selects nothing. Reported before mutation.
        before = lpt._fingerprint(h.scheduler)
        with self.assertRaises(avs.AdaptedVLLMControlOnlyProposal) as ctx:
            h.schedule(3.0)
        prop = ctx.exception.proposal
        self.assertEqual(ctx.exception.category, avs.CATEGORY_CONTROL_ONLY)
        self.assertEqual((prop.victims, prop.actions), ((0,), ()))
        self.assertEqual(prop.running_stop_reason, avs.RUNNING_SELF_EVICTED)
        self.assertEqual(lpt._fingerprint(h.scheduler), before)
        self.assertEqual(h.calls, [])
        self.assertEqual(h.scheduler.num_running_batches, 0)

    def test_mixed_decode_demands_in_service_order(self):
        h = _Harness(num_gpu_blocks=4, max_num_seqs=2, max_num_batched_tokens=16)
        h.add(0, 4, 1.0, 4)
        h.add(1, 5, 1.0, 4)
        h.step(10.0)
        h.step(11.0)
        manager = h.scheduler.block_manager
        self.assertEqual((h.gaps(), manager.get_num_free_gpu_blocks()),
                         ({0: 1, 1: 0}, 1))
        outputs = h.schedule(12.0)
        # The one-demand decode takes the last free block before the
        # zero-demand decode; service order, not regrouped by demand.
        self.assertEqual(lpt._emitted(outputs), [(0, 0), (1, 0)])
        self.assertEqual(h.calls, [("_append_slot", 0), ("_append_slot", 1)])
        self.assertEqual(manager.get_num_free_gpu_blocks(), 0)
        self.assertEqual(outputs.preempted_seq_ids, [])
        h.replay(outputs)
        _assert_conserved(self, h)

    def test_full_context_charge_resident_prefill_and_positive_watermark(self):
        # 100-block pool: native watermark int(0.01 * 100) = 1 block.
        h = _Harness(num_gpu_blocks=100, max_num_seqs=4,
                     max_num_batched_tokens=200, max_model_len=400)
        manager = h.scheduler.block_manager
        self.assertEqual(manager.watermark_blocks, 1)
        a = h.add(0, 388, 1.0, 2)   # 97 blocks
        h.add(1, 12, 1.0, 2)        # 3 blocks
        h.add(2, 4, 1.0, 2)         # 1 block, would fit
        outputs = h.step(10.0)
        # Full-context admission: 200-token chunk, 97 blocks charged.
        self.assertEqual(lpt._emitted(outputs), [(0, 200)])
        self.assertEqual(len(h.central_tables()[0]), 97)
        self.assertEqual(manager.get_num_free_gpu_blocks(), 3)
        tables = h.central_tables()
        outputs = h.step(11.0)
        prop = h.proposals[0]
        # Resident prefill: zero charge, tables unchanged. Head 1 needs 3
        # blocks: 3 - 3 = 0 < watermark 1, so admission stops at the head;
        # the later 1-block request is not searched for.
        self.assertEqual(prop.actions[0].block_demand, 0)
        self.assertEqual(lpt._emitted(outputs), [(0, 188)])
        self.assertEqual(h.central_tables(), tables)
        self.assertEqual(prop.waiting_stop_reason, avs.WAITING_ADMISSION_MEMORY)
        # Temporary contention: the long request decodes (one block at
        # 389 tokens), finishes, frees its blocks; then the queue is admitted.
        steps = 0
        while not a.is_finished():
            outputs = h.step(12.0 + steps)
            self.assertEqual(h.proposals[0].waiting_stop_reason,
                             avs.WAITING_ADMISSION_MEMORY)
            steps += 1
        self.assertEqual(manager.get_num_free_gpu_blocks(), 100)
        outputs = h.step(20.0)
        self.assertEqual(lpt._emitted(outputs), [(1, 12), (2, 4)])
        self.assertEqual(manager.get_num_free_gpu_blocks(), 96)
        _assert_conserved(self, h)

    def test_zero_watermark_admission_to_exact_exhaustion(self):
        h = _Harness(num_gpu_blocks=3, max_num_seqs=2, max_num_batched_tokens=16)
        h.add(0, 12, 1.0, 1)
        outputs = h.step(10.0)
        self.assertEqual(lpt._emitted(outputs), [(0, 12)])
        self.assertEqual(h.scheduler.block_manager.get_num_free_gpu_blocks(), 0)

    def test_preemption_recomputation_and_readmission(self):
        h = _Harness(num_gpu_blocks=4, max_num_seqs=4, max_num_batched_tokens=16)
        seqs = [h.add(i, 4, 1.0, 3) for i in range(4)]
        generated = {i: 0 for i in range(4)}

        def step(now):
            outputs = h.step(now)
            for m in outputs.scheduled_seq_metadata_list:
                if m.prompt_chunk_len == 0:
                    generated[m.seq_id] += 1
            _assert_conserved(self, h)
            return outputs

        step(10.0)
        step(11.0)
        outputs = step(12.0)
        self.assertEqual(outputs.preempted_seq_ids, [3, 2])
        step(13.0)  # 0 and 1 finish
        self.assertEqual(h.ids(h.scheduler.running), [])
        outputs = step(14.0)
        # Readmission of the expanded 5-token contexts, front order 2, 3.
        self.assertEqual(lpt._emitted(outputs), [(2, 5), (3, 5)])
        self.assertEqual(h.calls, [("_allocate", 2), ("_allocate", 3)])
        self.assertEqual({k: len(v) for k, v in h.central_tables().items()},
                         {2: 2, 3: 2})
        for now in (15.0, 16.0, 17.0):
            step(now)
        # Inherited recomputation counter: the output counter restarted, so
        # each restarted request produced 1 + 3 generated-token events
        # against max_tokens 3.
        for i in (2, 3):
            self.assertTrue(seqs[i].is_finished())
            self.assertEqual((generated[i], seqs[i].get_output_len()), (4, 3))
        self.assertEqual((generated[0], generated[1]), (3, 3))


class PreMutationFailureTest(unittest.TestCase):

    def _assert_rejected(self, h, now, category, error=avs.AdaptedVLLMSchedulingError):
        before = lpt._fingerprint(h.scheduler)
        with self.assertRaises(error) as ctx:
            h.schedule(now)
        self.assertEqual(ctx.exception.category, category)
        self.assertEqual(lpt._fingerprint(h.scheduler), before)
        self.assertEqual(h.calls, [])
        return ctx.exception

    def test_unsupported_and_malformed_state(self):
        h = _Harness(num_gpu_blocks=4, max_num_seqs=2, max_num_batched_tokens=8)
        h.add(0, 8, float("nan"), 1)
        self._assert_rejected(h, 10.0, avs.CATEGORY_MALFORMED_STATE)

        h = _Harness(num_gpu_blocks=4, max_num_seqs=2, max_num_batched_tokens=8)
        h.add(0, 8, 1.0, 1)
        h.scheduler.num_running_batches = 1
        iteration = h.scheduler._iteration_id
        self._assert_rejected(h, 10.0, avs.CATEGORY_UNSUPPORTED_STATE)
        self.assertEqual(h.scheduler._iteration_id, iteration)

        h = _Harness(num_gpu_blocks=4, max_num_seqs=2, max_num_batched_tokens=8,
                     max_model_len=8)
        h.add(0, 9, 1.0, 1)
        self._assert_rejected(h, 10.0, avs.CATEGORY_UNSUPPORTED_STATE)

        # Partial prefill whose table lost a block (moved to the free list).
        h = _Harness(num_gpu_blocks=4, max_num_seqs=2, max_num_batched_tokens=4)
        h.add(0, 8, 1.0, 1)
        h.step(10.0)
        manager = h.scheduler.block_manager
        manager.gpu_allocator.free(manager.block_tables[0].pop())
        self._assert_rejected(h, 11.0, avs.CATEGORY_UNSUPPORTED_STATE)

        # A block in both a table and the free list.
        h = _Harness(num_gpu_blocks=4, max_num_seqs=2, max_num_batched_tokens=4)
        h.add(0, 8, 1.0, 1)
        h.step(10.0)
        manager = h.scheduler.block_manager
        manager.gpu_allocator.free(manager.block_tables[0][0])
        self._assert_rejected(h, 11.0, avs.CATEGORY_MALFORMED_STATE)

        # A decode at max_model_len.
        h = _Harness(num_gpu_blocks=4, max_num_seqs=1, max_num_batched_tokens=8,
                     max_model_len=5)
        h.add(0, 4, 1.0, 5)
        h.step(10.0)
        h.step(11.0)
        self._assert_rejected(h, 12.0, avs.CATEGORY_UNSUPPORTED_STATE)

    def test_inadmissible_head_and_idle(self):
        h = _Harness(num_gpu_blocks=2, max_num_seqs=2, max_num_batched_tokens=16)
        h.add(0, 12, 1.0, 1)  # 3 blocks in a 2-block pool
        err = self._assert_rejected(h, 10.0, avs.CATEGORY_INADMISSIBLE_HEAD)
        self.assertEqual(err.proposal.waiting_stop_reason,
                         avs.WAITING_ADMISSION_MEMORY)

        h = _Harness(num_gpu_blocks=2, max_num_seqs=2, max_num_batched_tokens=16)
        h.add(0, 4, 50.0, 1)  # future only: ordinary idle
        before = lpt._fingerprint(h.scheduler)
        outputs = h.schedule(10.0)
        self.assertTrue(outputs.has_no_output())
        self.assertEqual(lpt._fingerprint(h.scheduler), before)


# ---------------------------------------------------------------------------
# CPU packing, attention metadata, pruning, and greedy sampling
# ---------------------------------------------------------------------------

VOCAB = 128
SHIFT = 50


class PackingAndSamplingTest(unittest.TestCase):
    """Synthetic forward: hidden row = one-hot(input token); the embedding
    maps token t to logit row (t + SHIFT) % VOCAB, so greedy sampling returns
    ``(last input token of the request + SHIFT) % VOCAB``. The tensor-parallel
    gather is stubbed with its single-rank identity."""

    def _forward(self, h, outputs):
        h.engine_seqs.on_schedule(outputs)
        _, metadata = h.worker_seqs.on_schedule(outputs)
        runner = ModelRunner.__new__(ModelRunner)
        runner.device = torch.device("cpu")
        tokens, positions = runner._prepare_inputs(metadata)
        attention = FlashAttentionWrapper.__new__(FlashAttentionWrapper)
        attention.device = torch.device("cpu")
        attention.block_size = BLOCK_SIZE
        attention.begin_forward(metadata)
        hidden = torch.nn.functional.one_hot(tokens, VOCAB).float()
        embedding = torch.zeros(VOCAB, VOCAB)
        for t in range(VOCAB):
            embedding[(t + SHIFT) % VOCAB, t] = 1.0
        with mock.patch.object(sampler_module,
                               "gather_from_tensor_model_parallel_region",
                               lambda x: x):
            samples = sampler_module.Sampler(embedding, VOCAB)(hidden, metadata)
        h.worker_seqs.on_step_completed(outputs, samples)
        h.engine_seqs.on_step_completed(outputs, samples)
        h.scheduler.on_step_completed()
        return metadata, tokens.tolist(), positions.tolist(), attention, samples

    def test_mixed_batch_rows_and_samples_follow_requests(self):
        h = _Harness(num_gpu_blocks=8, max_num_seqs=3, max_num_batched_tokens=8)
        a = h.add(0, 3, 1.0, 3, token_offset=10)
        b = h.add(1, 4, 1.0, 3, token_offset=20)
        outputs = h.schedule(10.0)
        self.assertEqual(lpt._emitted(outputs), [(0, 3), (1, 4)])
        self._forward(h, outputs)
        c = h.add(2, 6, 1.0, 3, token_offset=40)
        outputs = h.schedule(11.0)
        # Selection order is A, B (decodes) then C; layout puts C first.
        self.assertEqual([a.seq_id for a in h.proposals[0].actions], [0, 1, 2])
        self.assertEqual(lpt._emitted(outputs), [(2, 6), (0, 0), (1, 0)])
        metadata, tokens, positions, attn, samples = self._forward(h, outputs)
        self.assertEqual([m.seq.seq_id for m in metadata], [2, 0, 1])
        # Prompt-first packing: C's 6 prompt tokens, then A's and B's last
        # prompt tokens (final-prefill samples were discarded), then padding.
        self.assertEqual(tokens, [40, 41, 42, 43, 44, 45, 12, 23])
        self.assertEqual(positions, [0, 1, 2, 3, 4, 5, 2, 3])
        worker = {i: h.worker_seqs.get_seq(i) for i in (0, 1, 2)}
        self.assertEqual(attn.prefill_query_lens, [6])
        self.assertEqual([t.tolist() for t in attn.prefill_cache_lens], [[0]])
        self.assertEqual(attn.prefill_block_tables[0].tolist(),
                         [h.worker_seqs.block_manager.get_block_table(worker[2])])
        self.assertEqual(attn.decode_cache_len.tolist(), [2, 3])
        self.assertEqual(
            attn.decode_block_table.tolist(),
            [h.worker_seqs.block_manager.get_block_table(worker[0]),
             h.worker_seqs.block_manager.get_block_table(worker[1])])
        # Samples keep metadata order and each request's own last row.
        self.assertEqual([(s.seq_id, s.output_token) for s in samples],
                         [(2, 45 + SHIFT), (0, 12 + SHIFT), (1, 23 + SHIFT)])
        # Native completion: C's final-prefill sample is discarded (inherited);
        # A and B append their own tokens.
        self.assertEqual(a.get_output_token_ids(), [12 + SHIFT])
        self.assertEqual(b.get_output_token_ids(), [23 + SHIFT])
        self.assertEqual(h.engine_seqs.get_seq(2).get_output_len(), 0)
        self.assertTrue(h.engine_seqs.get_seq(2).prompt_processing_finished)
        for i in (0, 1, 2):
            self.assertEqual(lpt._seq_fingerprint(worker[i]),
                             lpt._seq_fingerprint(h.engine_seqs.get_seq(i)))
        # Next decode step feeds each request its own sampled token.
        outputs = h.schedule(12.0)
        self.assertEqual(lpt._emitted(outputs), [(0, 0), (1, 0), (2, 0)])
        _, tokens, positions, _, samples = self._forward(h, outputs)
        self.assertEqual(tokens[:3], [12 + SHIFT, 23 + SHIFT, 45])
        self.assertEqual(positions[:3], [3, 4, 5])
        self.assertEqual([(s.seq_id, s.output_token) for s in samples],
                         [(0, (12 + 2 * SHIFT) % VOCAB),
                          (1, (23 + 2 * SHIFT) % VOCAB), (2, 45 + SHIFT)])
        _assert_conserved(self, h)


if __name__ == "__main__":
    unittest.main()
