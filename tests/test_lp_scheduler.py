import copy
import dataclasses
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lp_relaxation_scheduler as lrs  # noqa: E402
import lpserve_plan_execution as lpe  # noqa: E402
import lpserve_state_mapping as lsm  # noqa: E402

from sarathi.config import (  # noqa: E402
    CacheConfig,
    LPSchedulerConfig,
    SchedulerType,
)
from sarathi.core.block_space_manager.block_space_manager_registry import (  # noqa: E402
    BlockSpaceManagerRegistry,
)
from sarathi.core.block_space_manager.vllm_block_space_manager import (  # noqa: E402
    VLLMBlockSpaceManager,
)
from sarathi.core.datatypes.sampling_params import SamplingParams  # noqa: E402
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs  # noqa: E402
from sarathi.core.datatypes.sequence import SamplerOutput, Sequence  # noqa: E402
from sarathi.core.datatypes.sequence_status import SequenceStatus  # noqa: E402
from sarathi.core.scheduler import lp_scheduler  # noqa: E402
from sarathi.core.scheduler.lp_scheduler import (  # noqa: E402
    LPScheduler,
    LPSchedulingError,
)
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry  # noqa: E402
from sarathi.core.sequence_manager.engine_sequence_manager import (  # noqa: E402
    EngineSequenceManager,
)
from sarathi.core.sequence_manager.worker_sequence_manager import (  # noqa: E402
    WorkerSequenceManager,
)
from sarathi.metrics.metrics_store import MetricsStore  # noqa: E402
from sarathi.utils.singleton import Singleton  # noqa: E402

# Approved provisional plumbing inputs, not production defaults or research
# decisions. Individual tests override one value only where documented.
BLOCK_SIZE = 4
NUM_GPU_BLOCKS = 10
MAX_MODEL_LEN = 32
RESIDENT_LIMIT = 4
B_MAX = 8
C_MAX = 4
S_MAX = 3
MEMORY_RESERVE = 1
DECODE_POLICY_ID = "conservative_one_block_v1"
DECODE_UTILITY = 1.0
PREFILL_TOKEN_UTILITY = 1.0
PREEMPTION_PENALTY = 1.0
NUMERICAL_POLICY = lrs.NumericalPolicy(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)
SAMPLED_TOKEN = 7

_created_metrics_store = False


def setUpModule():
    # BaseScheduler reads the MetricsStore singleton; initialize it through
    # its existing disabled mode for this module only.
    global _created_metrics_store
    if MetricsStore not in Singleton._instances:
        MetricsStore(None)
        _created_metrics_store = True


def tearDownModule():
    if _created_metrics_store:
        Singleton._instances.pop(MetricsStore, None)


def _config(**overrides):
    values = dict(
        max_num_seqs=RESIDENT_LIMIT,
        max_model_len=MAX_MODEL_LEN,
        num_pipeline_stages=1,
        b_max=B_MAX,
        c_max=C_MAX,
        s_max=S_MAX,
        memory_reserve=MEMORY_RESERVE,
        decode_memory_policy_id=DECODE_POLICY_ID,
        decode_utility=DECODE_UTILITY,
        prefill_token_utility=PREFILL_TOKEN_UTILITY,
        preemption_penalty=PREEMPTION_PENALTY,
        numerical_policy=NUMERICAL_POLICY,
    )
    values.update(overrides)
    return LPSchedulerConfig(**values)


def _cache_config(num_gpu_blocks=NUM_GPU_BLOCKS):
    cache_config = CacheConfig(block_size=BLOCK_SIZE, gpu_memory_utilization=0.9)
    cache_config.num_gpu_blocks = num_gpu_blocks
    return cache_config


class _EngineSequenceManager(EngineSequenceManager):
    """Engine sequence manager without a tokenizer: text detokenization is
    suppressed; status, progress, and token bookkeeping stay native. This
    does not test text generation."""

    def __init__(self):
        super().__init__(tokenizer=None)

    def _decode_seq(self, seq):
        pass


class _Harness:
    """Real registered scheduler, central sequence manager sharing the
    scheduler's sequence objects, and a worker sequence manager holding
    separate copies, replayed in the engine's single-stage order."""

    def __init__(self, num_gpu_blocks=NUM_GPU_BLOCKS, **config_overrides):
        # One cache configuration builds both the central and worker pools.
        self.scheduler_config = _config(**config_overrides)
        cache_config = _cache_config(num_gpu_blocks)
        self.scheduler = SchedulerRegistry.get(
            SchedulerType.LP, self.scheduler_config, cache_config,
        )
        self.engine_seqs = _EngineSequenceManager()
        self.worker_seqs = WorkerSequenceManager(
            cache_config, self.scheduler_config,
        )
        self.clock = mock.Mock()

    def add(self, seq_id, num_prompt_tokens, arrival_time, max_tokens):
        seq = Sequence(
            seq_id=seq_id,
            prompt=None,
            prompt_token_ids=list(range(num_prompt_tokens)),
            block_size=BLOCK_SIZE,
            eos_token_id=-2,
            arrival_time=arrival_time,
            sampling_params=SamplingParams(ignore_eos=True, max_tokens=max_tokens),
        )
        self.engine_seqs.add_seq(seq)
        self.worker_seqs.add_seq(copy.deepcopy(seq))
        self.scheduler.add_seq(seq)
        return seq

    def schedule(self, now):
        self.clock.monotonic.return_value = now
        with mock.patch.object(lp_scheduler, "time", self.clock):
            return self.scheduler.schedule()

    def replay(self, outputs, sample=None):
        """Engine replay, worker replay/completion with synthetic sampler
        output (one entry per scheduled entry, in order, including prefill
        entries), then central and scheduler completion. ``sample`` maps a
        scheduled metadata entry to its token; by default every entry gets
        ``SAMPLED_TOKEN``."""
        self.engine_seqs.on_schedule(outputs)
        self.worker_seqs.on_schedule(outputs)
        sampler_outputs = [
            SamplerOutput(m.seq_id, SAMPLED_TOKEN if sample is None else sample(m))
            for m in outputs.scheduled_seq_metadata_list
        ]
        self.worker_seqs.on_step_completed(outputs, sampler_outputs)
        self.engine_seqs.on_step_completed(outputs, sampler_outputs)
        self.scheduler.on_step_completed()

    def step(self, now, sample=None):
        outputs = self.schedule(now)
        self.replay(outputs, sample)
        return outputs

    def central_tables(self):
        return _tables(self.scheduler.block_manager)

    def worker_tables(self):
        return _tables(self.worker_seqs.block_manager)


def _tables(block_manager):
    return {
        seq_id: [b.block_number for b in table]
        for seq_id, table in block_manager.block_tables.items()
    }


def _emitted(outputs):
    return [
        (m.seq_id, m.prompt_chunk_len)
        for m in outputs.scheduled_seq_metadata_list
    ]


def _seq_fingerprint(seq):
    return (
        seq.seq_id,
        seq.get_status().name,
        tuple(seq.prompt_token_ids),
        tuple(seq.output_token_ids),
        seq.prompt_tokens_processed,
        seq.prompt_processing_finished,
        len(seq.logical_token_blocks),
    )


def _fingerprint(scheduler):
    """Scheduler state other than ``_iteration_id`` that a non-executing
    call must leave unchanged."""
    block_manager = scheduler.block_manager
    return (
        tuple(id(s) for s in scheduler.waiting),
        tuple(id(s) for s in scheduler.running),
        tuple(_seq_fingerprint(s) for s in scheduler.waiting + scheduler.running),
        scheduler.num_running_batches,
        tuple(sorted(
            (seq_id, tuple(b.block_number for b in table))
            for seq_id, table in block_manager.block_tables.items()
        )),
        tuple(b.block_number for b in block_manager.gpu_allocator.free_blocks),
    )


class _Spies:
    """Test-only wrappers around the real pipeline entry points that record
    call order and the values crossing each boundary."""

    def __init__(self):
        self.calls = []
        self._patches = []

    def __enter__(self):
        real_utilities = LPScheduler._build_utilities
        real_map = lsm.map_scheduler_state
        real_solve = lrs.solve_and_extract
        real_execute = lpe.execute_plan
        calls = self.calls

        def utilities(scheduler, now):
            calls.append(("utilities", now))
            return real_utilities(scheduler, now)

        def map_state(scheduler, **kwargs):
            result = real_map(scheduler, **kwargs)
            calls.append((
                "map", kwargs["snapshot_time"], scheduler.num_running_batches,
                scheduler._iteration_id, result,
            ))
            return result

        def solve(problem):
            result = real_solve(problem)
            calls.append(("solve", problem.problem_id, result))
            return result

        def execute(scheduler, snapshot, result):
            # Recorded on entry so a call that raises is still visible.
            call = ["execute", snapshot, scheduler.num_running_batches, None]
            calls.append(call)
            call[3] = real_execute(scheduler, snapshot, result)
            return call[3]

        self._patches = [
            mock.patch.object(LPScheduler, "_build_utilities", utilities),
            mock.patch.object(lsm, "map_scheduler_state", map_state),
            mock.patch.object(lrs, "solve_and_extract", solve),
            mock.patch.object(lpe, "execute_plan", execute),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        return False

    def names(self):
        return [call[0] for call in self.calls]


class RegistrationAndConfigTest(unittest.TestCase):
    def test_registration_and_explicit_configuration(self):
        self.assertEqual(int(SchedulerType.LP), 8)
        self.assertEqual(int(SchedulerType.SLAI_SCHEDULER), 7)
        self.assertIs(SchedulerRegistry.get_class(SchedulerType.LP), LPScheduler)
        self.assertIs(
            BlockSpaceManagerRegistry.get_class(SchedulerType.LP),
            VLLMBlockSpaceManager,
        )

        harness = _Harness()
        scheduler = harness.scheduler
        config = scheduler.scheduler_config
        self.assertIsInstance(scheduler, LPScheduler)
        self.assertIs(type(scheduler.block_manager), VLLMBlockSpaceManager)
        self.assertIs(type(harness.worker_seqs.block_manager), VLLMBlockSpaceManager)
        self.assertEqual(scheduler.block_manager.num_total_gpu_blocks, NUM_GPU_BLOCKS)
        self.assertEqual(config.type, SchedulerType.LP)
        self.assertEqual(config.max_num_batched_tokens, B_MAX)
        self.assertEqual(
            (config.c_max, config.s_max, config.max_num_seqs), (C_MAX, S_MAX, 4),
        )
        self.assertIs(config.numerical_policy, NUMERICAL_POLICY)
        self.assertEqual(scheduler._iteration_id, -1)
        self.assertEqual(scheduler.num_running_batches, 0)

    def test_multi_stage_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one pipeline stage"):
            _config(num_pipeline_stages=2)


class OrdinaryIdleTest(unittest.TestCase):
    def _assert_idle(self, harness, now):
        scheduler = harness.scheduler
        before = _fingerprint(scheduler)
        iteration = scheduler._iteration_id
        with _Spies() as spies:
            outputs = harness.schedule(now)
        self.assertIsInstance(outputs, SchedulerOutputs)
        self.assertTrue(outputs.has_no_output())
        self.assertEqual(scheduler._iteration_id, iteration + 1)
        self.assertEqual(outputs.id, scheduler._iteration_id)
        self.assertEqual(spies.calls, [])
        self.assertEqual(_fingerprint(scheduler), before)
        self.assertEqual(harness.clock.monotonic.call_count, 1)

    def test_empty_state_is_ordinary_idle(self):
        harness = _Harness()
        self._assert_idle(harness, now=5.0)

    def test_future_only_state_is_ordinary_idle(self):
        harness = _Harness()
        harness.add(0, 4, arrival_time=10.0, max_tokens=2)
        self._assert_idle(harness, now=5.0)
        self.assertEqual(harness.scheduler.waiting[0].get_status(), SequenceStatus.WAITING)


class LivePipelineTest(unittest.TestCase):
    def _assert_synchronous_pipeline(self, harness, spies, now, outputs):
        scheduler = harness.scheduler
        self.assertEqual(spies.names(), ["utilities", "map", "solve", "execute"])
        self.assertEqual(harness.clock.monotonic.call_count, 1)
        _, utilities_now = spies.calls[0]
        _, map_now, map_batches, map_iteration, snapshot = spies.calls[1]
        _, problem_id, result = spies.calls[2]
        _, executed_snapshot, execute_batches, executed_outputs = spies.calls[3]
        decision_id = str(scheduler._iteration_id)
        self.assertEqual((utilities_now, map_now, snapshot.snapshot_time), (now,) * 3)
        self.assertEqual(map_iteration, scheduler._iteration_id)
        self.assertEqual(
            (snapshot.snapshot_id, problem_id, result.problem_id,
             result.plan.problem_id),
            (decision_id,) * 4,
        )
        self.assertIs(executed_snapshot, snapshot)
        self.assertEqual((map_batches, execute_batches), (0, 0))
        self.assertIs(executed_outputs, outputs)
        self.assertEqual(outputs.id, scheduler._iteration_id)
        self.assertEqual(scheduler.num_running_batches, 1)

    def test_request_lifecycle_through_real_pipeline_and_replay(self):
        harness = _Harness()
        scheduler = harness.scheduler
        seq = harness.add(0, 8, arrival_time=1.0, max_tokens=2)
        worker_seq = harness.worker_seqs.get_seq(0)
        self.assertIs(harness.engine_seqs.get_seq(0), seq)
        self.assertIsNot(worker_seq, seq)

        # Each entry: emitted output, then the expected prompt progress,
        # output length, and central physical block count after replay.
        expected = [
            ([(0, 4)], 4, 0, 2),  # admission: full 8-token context, 2 blocks
            ([(0, 4)], 8, 0, 2),  # resident prefill completes the prompt
            ([(0, 0)], 8, 1, 2),  # decode with logical/physical gap 0
            ([(0, 0)], 8, 2, 3),  # decode with gap 1 allocates; then finishes
        ]
        for index, (emitted, processed, output_len, blocks) in enumerate(expected):
            now = 2.0 + index
            harness.clock.reset_mock()
            with _Spies() as spies:
                outputs = harness.schedule(now)
            self._assert_synchronous_pipeline(harness, spies, now, outputs)
            self.assertEqual(_emitted(outputs), emitted)
            self.assertEqual(outputs.ignored_seq_ids, [])
            self.assertEqual(outputs.preempted_seq_ids, [])
            self.assertEqual(scheduler.waiting, [])
            self.assertEqual(scheduler.running, [seq])
            self.assertEqual(len(harness.central_tables()[0]), blocks)

            harness.replay(outputs)

            self.assertEqual(scheduler.num_running_batches, 0)
            for copy_ in (seq, worker_seq):
                self.assertEqual(copy_.get_num_prompt_tokens_processed(), processed)
                self.assertEqual(copy_.prompt_processing_finished, processed == 8)
                self.assertEqual(copy_.get_output_len(), output_len)
            if index < len(expected) - 1:
                self.assertEqual(seq.get_status(), SequenceStatus.PAUSED)
                self.assertEqual(worker_seq.get_status(), SequenceStatus.PAUSED)
                self.assertEqual(scheduler.running, [seq])
                self.assertEqual(harness.central_tables(), harness.worker_tables())
                self.assertEqual(len(harness.central_tables()[0]), blocks)

        # Finished: removed everywhere and blocks freed centrally and on the
        # worker.
        self.assertEqual(seq.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
        self.assertEqual(worker_seq.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
        self.assertEqual(seq.get_output_token_ids(), [SAMPLED_TOKEN] * 2)
        self.assertEqual((scheduler.waiting, scheduler.running), ([], []))
        self.assertIsNone(harness.engine_seqs.get_seq(0))
        self.assertIsNone(harness.worker_seqs.get_seq(0))
        self.assertEqual((harness.central_tables(), harness.worker_tables()), ({}, {}))
        for block_manager in (scheduler.block_manager, harness.worker_seqs.block_manager):
            self.assertEqual(block_manager.get_num_free_gpu_blocks(), NUM_GPU_BLOCKS)
        self.assertFalse(scheduler.has_unfinished_seqs())

        # The drained scheduler returns to ordinary idle.
        with _Spies() as spies:
            outputs = harness.schedule(10.0)
        self.assertTrue(outputs.has_no_output())
        self.assertEqual(spies.calls, [])
        self.assertEqual(outputs.id, 4)

    def test_mixed_prompt_first_output_and_unselected_resident(self):
        harness = _Harness()
        scheduler = harness.scheduler
        harness.add(0, 4, arrival_time=1.0, max_tokens=4)
        harness.add(1, 4, arrival_time=2.0, max_tokens=4)
        harness.add(2, 4, arrival_time=3.0, max_tokens=4)
        harness.add(3, 8, arrival_time=4.0, max_tokens=4)

        # Staggered arrivals give unique optima: every arrived request fits
        # the token, action, and memory limits, so each is selected.
        self.assertEqual(_emitted(harness.step(1.0)), [(0, 4)])
        # Request 1 is admitted ahead of request 0's decode (prompt first),
        # although request 0 has the smaller order_key.
        self.assertEqual(_emitted(harness.step(2.0)), [(1, 4), (0, 0)])
        self.assertEqual(_emitted(harness.step(3.0)), [(2, 4), (0, 0), (1, 0)])
        self.assertEqual(harness.central_tables(), harness.worker_tables())
        self.assertEqual(scheduler.num_running_batches, 0)

        # Three decode-ready residents and one arriving 8-token prompt with
        # S_MAX=3: request 3's 4-token admission is in every optimum, and two
        # of the three decodes fill the remaining action slots. Which decode
        # is left out is a solver tie, so only the invariant is asserted.
        seqs = {s.seq_id: s for s in scheduler.waiting + scheduler.running}
        before = {k: _seq_fingerprint(seqs[k]) for k in seqs}
        tables_before = harness.central_tables()
        harness.clock.reset_mock()
        with _Spies() as spies:
            outputs = harness.schedule(4.0)
        self._assert_synchronous_pipeline(harness, spies, 4.0, outputs)
        emitted = _emitted(outputs)
        self.assertEqual(emitted[0], (3, 4))
        decoded = [seq_id for seq_id, chunk in emitted[1:]]
        self.assertEqual([chunk for _, chunk in emitted[1:]], [0, 0])
        self.assertEqual(decoded, sorted(decoded))
        self.assertEqual(len(decoded), 2)
        self.assertTrue(set(decoded) < {0, 1, 2})
        (unselected,) = {0, 1, 2} - set(decoded)
        self.assertEqual(outputs.num_batched_tokens, 6)
        self.assertEqual(len(scheduler.running), RESIDENT_LIMIT)

        harness.replay(outputs)

        # The unselected resident stays owned and allocated, unchanged.
        self.assertIn(seqs[unselected], scheduler.running)
        self.assertEqual(_seq_fingerprint(seqs[unselected]), before[unselected])
        self.assertEqual(
            harness.central_tables()[unselected], tables_before[unselected],
        )
        for seq_id in decoded:
            self.assertEqual(
                seqs[seq_id].get_output_len(), len(before[seq_id][3]) + 1,
            )
        self.assertEqual(seqs[3].get_num_prompt_tokens_processed(), 4)
        self.assertEqual(scheduler.waiting, [])
        self.assertEqual(scheduler.num_running_batches, 0)
        self.assertEqual(harness.central_tables(), harness.worker_tables())

    def test_three_requests_contend_for_two_action_slots_until_completion(self):
        # Case-specific: three 4-token requests, each limited to 4 generated
        # tokens, with two scheduled actions per step and room for all three
        # residents. Which tied request the solver selects is not asserted;
        # only the per-step invariants and the contention witnesses are.
        prompt_len, max_tokens, s_max, c_max, max_num_seqs = 4, 4, 2, 2, 3
        harness = _Harness(
            max_num_seqs=max_num_seqs, b_max=B_MAX, c_max=c_max, s_max=s_max,
        )
        scheduler = harness.scheduler
        block_manager = scheduler.block_manager
        seqs = {
            seq_id: harness.add(seq_id, prompt_len, arrival_time=1.0 + seq_id,
                                max_tokens=max_tokens)
            for seq_id in range(3)
        }
        utility = lsm.RequestUtility(
            DECODE_UTILITY, PREFILL_TOKEN_UTILITY, PREEMPTION_PENALTY,
        )

        worker_manager = harness.worker_seqs.block_manager

        def assert_pool_integrity(manager):
            allocated = [b.block_number for t in manager.block_tables.values()
                         for b in t]
            free = [b.block_number
                    for b in manager.gpu_allocator.free_blocks]
            self.assertEqual(len(allocated), len(set(allocated)))
            self.assertEqual(len(free), len(set(free)))
            self.assertFalse(set(allocated) & set(free))
            self.assertEqual(
                sorted(allocated + free), list(range(NUM_GPU_BLOCKS)),
            )

        def assert_managers_agree():
            # Physical block IDs are not compared: native frees iterate
            # set(block_table), whose order follows object identity, so the
            # two independent allocators can hand out different IDs.
            central, worker = harness.central_tables(), harness.worker_tables()
            self.assertEqual(
                {k: len(v) for k, v in central.items()},
                {k: len(v) for k, v in worker.items()},
            )
            self.assertEqual(
                block_manager.get_num_free_gpu_blocks(),
                worker_manager.get_num_free_gpu_blocks(),
            )
            assert_pool_integrity(block_manager)
            assert_pool_integrity(worker_manager)

        def remaining_work():
            return sum(
                prompt_len - s.get_num_prompt_tokens_processed()
                + max_tokens - s.get_output_len()
                for s in seqs.values()
            )

        work_limit = remaining_work()
        self.assertEqual(work_limit, 24)
        waiting_omitted, all_resident_omitted, later_selected = [], [], []
        steps = 0
        while scheduler.has_unfinished_seqs():
            self.assertLess(steps, work_limit, "termination guard reached")
            now = 10.0 + steps
            steps += 1

            # Quiescent decision boundary with exact ownership.
            unfinished = {k for k, s in seqs.items() if not s.is_finished()}
            waiting_ids = [s.seq_id for s in scheduler.waiting]
            running_ids = [s.seq_id for s in scheduler.running]
            self.assertEqual(scheduler.scheduler_config.num_pipeline_stages, 1)
            self.assertEqual(scheduler.num_running_batches, 0)
            self.assertEqual(
                sorted(waiting_ids + running_ids), sorted(unfinished),
            )
            for seq_id in waiting_ids:
                self.assertEqual(seqs[seq_id].get_status(), SequenceStatus.WAITING)
                self.assertNotIn(seq_id, block_manager.block_tables)
            for seq_id in running_ids:
                self.assertEqual(seqs[seq_id].get_status(), SequenceStatus.PAUSED)
                self.assertIn(seq_id, block_manager.block_tables)
            self.assertEqual(set(harness.engine_seqs.seq_map), unfinished)
            self.assertEqual(set(harness.worker_seqs.seq_map), unfinished)
            assert_managers_agree()
            before = {k: _seq_fingerprint(seqs[k]) for k in unfinished}
            tables_before = harness.central_tables()
            worker_tables_before = harness.worker_tables()
            free_before = block_manager.get_num_free_gpu_blocks()
            work_before = remaining_work()

            harness.clock.reset_mock()
            with _Spies() as spies:
                outputs = harness.schedule(now)
            self._assert_synchronous_pipeline(harness, spies, now, outputs)
            snapshot = spies.calls[1][4]
            result = spies.calls[2][2]
            self.assertIsInstance(snapshot, lsm.StateSnapshot)
            self.assertIsInstance(result, lrs.SchedulingSuccess)

            # Mapped inputs match the supplied configuration and state.
            problem = snapshot.lp_problem
            self.assertEqual(
                (problem.b_max, problem.c_max, problem.s_max),
                (B_MAX, c_max, s_max),
            )
            self.assertEqual(
                (snapshot.memory_reserve, snapshot.resident_limit,
                 snapshot.decode_memory_policy_id, snapshot.free_physical_blocks,
                 problem.m_free),
                (MEMORY_RESERVE, max_num_seqs, DECODE_POLICY_ID, free_before,
                 free_before),
            )
            self.assertIs(snapshot.numerical_policy, NUMERICAL_POLICY)
            self.assertEqual(
                sorted(r.raw_seq_id for r in snapshot.requests), sorted(unfinished),
            )
            for r in snapshot.requests:
                self.assertEqual(r.utility, utility)
            raw = {r.request_id: r.raw_seq_id for r in snapshot.requests}

            # The complete integer plan matches the emitted actions.
            plan = result.plan
            self.assertEqual(
                (plan.dominant_preemption_ids, plan.safety_preemption_ids), ((), ()),
            )
            selected = []
            for d in plan.decisions:
                self.assertEqual(d.preempt, 0)
                self.assertFalse(d.prefill_tokens > 0 and d.decode)
                if d.prefill_tokens > 0 or d.decode:
                    selected.append(d)
            selected.sort(key=lambda d: (0 if d.prefill_tokens > 0 else 1, d.order_key))
            emitted = _emitted(outputs)
            self.assertEqual(
                emitted, [(raw[d.request_id], d.prefill_tokens) for d in selected],
            )
            self.assertEqual((outputs.ignored_seq_ids, outputs.preempted_seq_ids), ([], []))
            ids = [seq_id for seq_id, _ in emitted]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(1 <= len(ids) <= s_max)
            chunks = [c for _, c in emitted]
            prefill_flags = [c > 0 for c in chunks]
            self.assertEqual(prefill_flags, sorted(prefill_flags, reverse=True))
            prefill_ids = {k for k, c in emitted if c > 0}
            decode_ids = {k for k, c in emitted if c == 0}
            for seq_id, chunk in emitted:
                remainder = prompt_len - before[seq_id][4]
                if chunk > 0:
                    self.assertLessEqual(chunk, min(remainder, c_max))
                else:
                    self.assertEqual(remainder, 0)
                    self.assertIn(seq_id, running_ids)
            self.assertEqual(outputs.num_batched_prompt_tokens, sum(chunks))
            self.assertEqual(outputs.num_batched_output_tokens, len(decode_ids))
            self.assertEqual(
                outputs.num_batched_tokens, sum(chunks) + len(decode_ids),
            )
            self.assertLessEqual(outputs.num_batched_tokens, B_MAX)
            self.assertLessEqual(len(scheduler.running), max_num_seqs)

            # Scheduling mutation: native admissions/appends only, no progress.
            tables_sched = harness.central_tables()
            expected_free = free_before
            for seq_id in unfinished:
                seq = seqs[seq_id]
                self.assertEqual(
                    (seq.get_num_prompt_tokens_processed(), seq.get_output_len()),
                    (before[seq_id][4], len(before[seq_id][3])),
                )
                if seq_id in prefill_ids and seq_id in waiting_ids:
                    self.assertIn(seq, scheduler.running)
                    self.assertEqual(
                        len(tables_sched[seq_id]), len(seq.logical_token_blocks),
                    )
                    expected_free -= len(tables_sched[seq_id])
                elif seq_id in decode_ids:
                    gap = before[seq_id][6] - len(tables_before[seq_id])
                    self.assertIn(gap, (0, 1))
                    self.assertEqual(
                        tables_sched[seq_id][:len(tables_before[seq_id])],
                        tables_before[seq_id],
                    )
                    self.assertEqual(
                        len(tables_sched[seq_id]), len(tables_before[seq_id]) + gap,
                    )
                    expected_free -= gap
                else:
                    self.assertEqual(
                        tables_sched.get(seq_id), tables_before.get(seq_id),
                    )
            self.assertEqual(block_manager.get_num_free_gpu_blocks(), expected_free)

            # Contention witnesses observed at this boundary.
            omitted = sorted(unfinished - set(ids))
            if any(k in waiting_ids for k in omitted):
                waiting_omitted.append(steps)
            if any(set(ids) & set(o) for _, o in all_resident_omitted):
                later_selected.append(steps)
            if len(running_ids) == 3 and all(before[k][5] for k in running_ids):
                self.assertTrue(omitted)
                all_resident_omitted.append((steps, omitted))

            harness.replay(outputs)

            # Completion advances only the selected requests.
            self.assertEqual(scheduler.num_running_batches, 0)
            freed = 0
            for seq_id in unfinished:
                seq = seqs[seq_id]
                processed, outputs_before = before[seq_id][4], before[seq_id][3]
                if seq_id in prefill_ids:
                    chunk = dict(emitted)[seq_id]
                    self.assertEqual(
                        seq.get_num_prompt_tokens_processed(), processed + chunk,
                    )
                    self.assertEqual(
                        seq.prompt_processing_finished, processed + chunk == prompt_len,
                    )
                    self.assertEqual(tuple(seq.output_token_ids), outputs_before)
                elif seq_id in decode_ids:
                    self.assertEqual(
                        tuple(seq.output_token_ids), outputs_before + (SAMPLED_TOKEN,),
                    )
                    self.assertEqual(seq.get_num_prompt_tokens_processed(), processed)
                else:
                    self.assertEqual(_seq_fingerprint(seq), before[seq_id])
                    self.assertIn(
                        seq, scheduler.waiting if seq_id in waiting_ids
                        else scheduler.running,
                    )
                    self.assertEqual(
                        harness.central_tables().get(seq_id), tables_before.get(seq_id),
                    )
                    self.assertEqual(
                        harness.worker_tables().get(seq_id),
                        worker_tables_before.get(seq_id),
                    )
                    self.assertEqual(
                        _seq_fingerprint(harness.worker_seqs.get_seq(seq_id)),
                        before[seq_id],
                    )
                if seq.is_finished():
                    self.assertIn(seq_id, decode_ids)
                    self.assertEqual(seq.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
                    self.assertEqual(seq.get_output_len(), max_tokens)
                    self.assertNotIn(seq, scheduler.waiting + scheduler.running)
                    self.assertNotIn(seq_id, harness.central_tables())
                    self.assertIsNone(harness.engine_seqs.get_seq(seq_id))
                    self.assertIsNone(harness.worker_seqs.get_seq(seq_id))
                    freed += len(tables_sched[seq_id])
                else:
                    self.assertEqual(seq.get_status(), SequenceStatus.PAUSED
                                     if seq_id in ids or seq_id in running_ids
                                     else SequenceStatus.WAITING)
            self.assertEqual(
                block_manager.get_num_free_gpu_blocks(), expected_free + freed,
            )
            assert_managers_agree()
            self.assertEqual(
                work_before - remaining_work(), sum(chunks) + len(decode_ids),
            )
            self.assertGreaterEqual(work_before - remaining_work(), 1)

        # Every required contention witness occurred.
        self.assertTrue(waiting_omitted, "no eligible waiting request was omitted")
        self.assertTrue(
            all_resident_omitted,
            "no boundary had three prompt-complete residents",
        )
        self.assertTrue(later_selected, "no omitted resident was later selected")

        # All three finished; central and worker state drained.
        for seq in seqs.values():
            self.assertEqual(seq.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
            self.assertEqual(seq.get_output_token_ids(), [SAMPLED_TOKEN] * max_tokens)
        self.assertEqual((scheduler.waiting, scheduler.running), ([], []))
        self.assertEqual(
            (harness.engine_seqs.seq_map, harness.worker_seqs.seq_map), ({}, {}),
        )
        self.assertEqual((harness.central_tables(), harness.worker_tables()), ({}, {}))
        for bm in (block_manager, worker_manager):
            self.assertEqual(bm.get_num_free_gpu_blocks(), NUM_GPU_BLOCKS)
            assert_pool_integrity(bm)

        # One ordinary idle call.
        before = _fingerprint(scheduler)
        iteration = scheduler._iteration_id
        with _Spies() as spies:
            outputs = harness.schedule(100.0)
        self.assertTrue(outputs.has_no_output())
        self.assertEqual(spies.calls, [])
        self.assertEqual(scheduler._iteration_id, iteration + 1)
        self.assertEqual(_fingerprint(scheduler), before)
        self.assertEqual(harness.worker_tables(), {})


class LivePreemptionTest(unittest.TestCase):
    # Case-specific provisional inputs: a 4-block pool in both managers,
    # one resident, one scheduled action, no planning reserve, decode
    # utility 10, prefill-token utility 1, and preemption penalty 1. The
    # harness gives request A (ID 0) prompt tokens 0..4 and B (ID 1) 0..11.
    POOL = 4
    CONFIG = dict(
        max_num_seqs=1, b_max=4, c_max=4, s_max=1, memory_reserve=0,
        decode_utility=10.0, prefill_token_utility=1.0, preemption_penalty=1.0,
    )
    PROMPT_A = list(range(5))
    PROMPT_B = list(range(12))

    _assert_synchronous_pipeline = LivePipelineTest._assert_synchronous_pipeline

    def _add(self, harness, seq_id, prompt, arrival_time):
        seq = harness.add(seq_id, len(prompt), arrival_time, max_tokens=1)
        self.assertEqual(seq.prompt_token_ids, prompt)
        return seq

    def _assert_managers_agree(self, harness):
        # Allocated sets, per-request counts, and free counts agree; physical
        # IDs can differ after inherited set-order frees and are not compared.
        central, worker = harness.central_tables(), harness.worker_tables()
        self.assertEqual(
            {k: len(v) for k, v in central.items()},
            {k: len(v) for k, v in worker.items()},
        )
        for manager in (harness.scheduler.block_manager,
                        harness.worker_seqs.block_manager):
            allocated = [b.block_number for t in manager.block_tables.values()
                         for b in t]
            free = [b.block_number for b in manager.gpu_allocator.free_blocks]
            self.assertEqual(sorted(allocated + free), list(range(self.POOL)))
            self.assertEqual(
                manager.get_num_free_gpu_blocks(),
                self.POOL - sum(len(v) for v in central.values()),
            )

    def test_solver_selected_preemption_is_recomputed_to_completion(self):
        harness = _Harness(num_gpu_blocks=self.POOL, **self.CONFIG)
        scheduler = harness.scheduler
        worker_manager = harness.worker_seqs.block_manager
        a = self._add(harness, 0, self.PROMPT_A, arrival_time=1.0)
        worker_a = harness.worker_seqs.get_seq(0)

        # A alone: a 4-token admission allocating its full 2-block context.
        self.assertEqual(_emitted(harness.step(1.0)), [(0, 4)])
        for copy_ in (a, worker_a):
            self.assertEqual(copy_.get_status(), SequenceStatus.PAUSED)
            self.assertEqual(copy_.get_num_prompt_tokens_processed(), 4)
            self.assertEqual(copy_.get_output_len(), 0)
        self.assertEqual(len(harness.central_tables()[0]), 2)
        self._assert_managers_agree(harness)
        self.assertEqual(scheduler.block_manager.get_num_free_gpu_blocks(), 2)
        self.assertEqual(worker_manager.get_num_free_gpu_blocks(), 2)

        b = self._add(harness, 1, self.PROMPT_B, arrival_time=2.0)
        worker_b = harness.worker_seqs.get_seq(1)

        # Boundary: the real mapper, solver, and extraction select A's
        # preemption and B's 4-token admission.
        harness.clock.reset_mock()
        with _Spies() as spies:
            outputs = harness.schedule(2.0)
        self._assert_synchronous_pipeline(harness, spies, 2.0, outputs)
        snapshot, result = spies.calls[1][4], spies.calls[2][2]
        problem = snapshot.lp_problem
        self.assertEqual(problem.legal_preemption_ids, frozenset({"0"}))
        self.assertEqual((problem.m_free, problem.w), (2, 0))
        requests = {r.request_id: r for r in problem.requests}
        self.assertEqual(requests["0"].preemption_recovery, 2)
        self.assertEqual(requests["1"].prefill_fixed_charge, 3)
        relaxed = {d.request_id: d for d in result.relaxed.decisions}
        tol = NUMERICAL_POLICY.feasibility_tol
        for rid, values in (("0", (0, 0, 0, 0.5)), ("1", (4, 0, 1, 0))):
            d = relaxed[rid]
            for got, want in zip((d.x, d.y, d.prefill_indicator, d.z), values):
                self.assertAlmostEqual(got, want, delta=tol)
        self.assertAlmostEqual(
            result.relaxed.normalized_objective, 3.5, delta=tol,
        )
        self.assertEqual(
            [(d.request_id, d.prefill_tokens, d.decode, d.preempt)
             for d in result.plan.decisions],
            [("0", 0, 0, 1), ("1", 4, 0, 0)],
        )
        self.assertEqual(result.plan.dominant_preemption_ids, ("0",))
        self.assertEqual(result.plan.safety_preemption_ids, ())
        self.assertEqual(outputs.preempted_seq_ids, [0])
        self.assertEqual(_emitted(outputs), [(1, 4)])

        # Central execution: A freed and returned to waiting with its state
        # untouched until replay; B admitted with its full 3-block context.
        self.assertEqual(scheduler.waiting, [a])
        self.assertEqual(scheduler.running, [b])
        self.assertEqual(a.get_status(), SequenceStatus.PAUSED)
        self.assertEqual(a.get_num_prompt_tokens_processed(), 4)
        self.assertEqual(set(harness.central_tables()), {1})
        self.assertEqual(len(harness.central_tables()[1]), 3)
        self.assertEqual(scheduler.block_manager.get_num_free_gpu_blocks(), 1)
        self.assertEqual(set(harness.worker_tables()), {0})

        # Worker replay frees A's local blocks before allocating B's.
        worker_calls = []
        real_free, real_allocate = worker_manager.free, worker_manager.allocate

        def free(seq):
            worker_calls.append(("free", seq.seq_id))
            real_free(seq)

        def allocate(seq):
            worker_calls.append(("allocate", seq.seq_id))
            real_allocate(seq)

        with mock.patch.object(worker_manager, "free", free), \
                mock.patch.object(worker_manager, "allocate", allocate):
            harness.replay(outputs)
        self.assertEqual(worker_calls, [("free", 0), ("allocate", 1)])

        # Central and worker replay reset A for recomputation from prompt
        # token zero; no generated tokens existed, so the prompt is intact.
        for copy_ in (a, worker_a):
            self.assertEqual(copy_.get_status(), SequenceStatus.WAITING)
            self.assertEqual(copy_.get_num_prompt_tokens_processed(), 0)
            self.assertFalse(copy_.prompt_processing_finished)
            self.assertEqual(copy_.prompt_token_ids, self.PROMPT_A)
            self.assertEqual(copy_.get_output_len(), 0)
        for copy_ in (b, worker_b):
            self.assertEqual(copy_.get_num_prompt_tokens_processed(), 4)
        self.assertEqual(scheduler.waiting, [a])
        self.assertEqual(scheduler.running, [b])
        self._assert_managers_agree(harness)

        # Remaining decisions until completion, bounded by remaining work.
        a_snapshots, a_admission, finished_order = [], None, []
        seqs = {0: a, 1: b}
        step_limit = 19  # B: 8 prompt + 1 decode; A: 5 prompt + 1 decode
        for step in range(step_limit + 1):
            self.assertLess(step, step_limit, "termination guard reached")
            if not scheduler.has_unfinished_seqs():
                break
            now = 3.0 + step
            waiting_before = list(scheduler.waiting)
            processed_before = {
                k: s.get_num_prompt_tokens_processed() for k, s in seqs.items()
            }
            harness.clock.reset_mock()
            with _Spies() as spies:
                outputs = harness.schedule(now)
            self._assert_synchronous_pipeline(harness, spies, now, outputs)
            for r in spies.calls[1][4].requests:
                if r.raw_seq_id == 0:
                    a_snapshots.append(r)
            self.assertEqual(outputs.preempted_seq_ids, [])
            (emitted,) = _emitted(outputs)
            if emitted[0] == 0 and a in waiting_before:
                a_admission = (emitted[1], processed_before[0],
                               len(harness.central_tables()[0]))
            harness.replay(outputs)
            self._assert_managers_agree(harness)
            for seq_id, seq in seqs.items():
                if seq.is_finished() and seq_id not in finished_order:
                    finished_order.append(seq_id)

        # Later mapping saw A waiting and unallocated, with its full-context
        # admission charge.
        self.assertEqual(a_snapshots[0].ownership, lsm.OWNERSHIP_WAITING)
        self.assertEqual(a_snapshots[0].physical_block_count, 0)
        self.assertEqual(a_snapshots[0].prompt_tokens_processed, 0)
        self.assertEqual(a_snapshots[0].prefill_fixed_charge, 2)
        # Readmission from prompt token zero with the full 2-block context.
        self.assertEqual(a_admission, (4, 0, 2))
        self.assertEqual(finished_order, [1, 0])
        for seq, prompt in ((a, self.PROMPT_A), (b, self.PROMPT_B)):
            self.assertEqual(seq.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
            self.assertEqual(seq.get_output_token_ids(), [SAMPLED_TOKEN])
            self.assertEqual(seq.prompt_token_ids, prompt)
        self.assertEqual((scheduler.waiting, scheduler.running), ([], []))
        self.assertEqual(
            (harness.engine_seqs.seq_map, harness.worker_seqs.seq_map), ({}, {}),
        )
        self.assertEqual((harness.central_tables(), harness.worker_tables()), ({}, {}))
        for manager in (scheduler.block_manager, worker_manager):
            self.assertEqual(manager.get_num_free_gpu_blocks(), self.POOL)

        # One ordinary idle call.
        before = _fingerprint(scheduler)
        with _Spies() as spies:
            outputs = harness.schedule(100.0)
        self.assertTrue(outputs.has_no_output())
        self.assertEqual(spies.calls, [])
        self.assertEqual(_fingerprint(scheduler), before)


class LiveDecodePreemptionTest(unittest.TestCase):
    # Case-specific provisional inputs: a 4-block pool in both managers,
    # max_model_len 16, one resident, one scheduled action, and no planning
    # reserve. Utilities are fixed per request for this case only (D-03
    # remains OPEN): A decode 40, prefill-token 1, penalty 1; B decode 40,
    # prefill-token 20, penalty 1. The configured uniform triple is A's; a
    # validation-only override of _build_utilities supplies B's.
    POOL = 4
    CONFIG = dict(
        max_num_seqs=1, max_model_len=16, b_max=4, c_max=4, s_max=1,
        memory_reserve=0, decode_utility=40.0, prefill_token_utility=1.0,
        preemption_penalty=1.0,
    )
    A, B = 0, 1  # raw seq IDs for requests A and B
    UTILITIES = {
        A: lsm.RequestUtility(40.0, 1.0, 1.0),
        B: lsm.RequestUtility(40.0, 20.0, 1.0),
    }
    PROMPT_A = list(range(5))
    PROMPT_B = list(range(12))
    MAX_TOKENS = {A: 2, B: 1}
    # Distinct synthetic decode tokens in completion order. Every prefill
    # entry samples PREFILL_SAMPLE, which native completion must discard.
    DECODE_TOKENS = {A: [101, 102, 103], B: [201]}
    PREFILL_SAMPLE = 900
    # Each nonempty step: emitted preempted IDs, emitted (seq_id, chunk), and
    # (prompt tokens processed, output token IDs) per unfinished request
    # after replay.
    EXPECTED = [
        ([], [(A, 4)], {A: (4, [])}),
        ([], [(A, 1)], {A: (5, [])}),
        ([], [(A, 0)], {A: (5, [101])}),
        ([A], [(B, 4)], {A: (0, []), B: (4, [])}),
        ([], [(B, 4)], {A: (0, []), B: (8, [])}),
        ([], [(B, 4)], {A: (0, []), B: (12, [])}),
        ([], [(B, 0)], {A: (0, [])}),
        ([], [(A, 4)], {A: (4, [])}),
        ([], [(A, 2)], {A: (6, [])}),
        ([], [(A, 0)], {A: (6, [102])}),
        ([], [(A, 0)], {}),
    ]
    BOUNDARY_STEP = 3
    READMISSION_STEP = 7

    _assert_synchronous_pipeline = LivePipelineTest._assert_synchronous_pipeline
    _assert_managers_agree = LivePreemptionTest._assert_managers_agree

    def _fixed_utilities(self):
        """Validation-only: give exactly the arrived, unfinished raw IDs that
        the real method selects their fixed A/B triple. The mapper still
        validates the exact key set and copies the values immutably."""
        real = LPScheduler._build_utilities
        fixed = self.UTILITIES

        def build(scheduler, now):
            return {seq_id: fixed[seq_id] for seq_id in real(scheduler, now)}

        return mock.patch.object(LPScheduler, "_build_utilities", build)

    def _state(self, harness):
        """Primitive central/worker state at a step boundary, for the trace."""
        def seqs(manager):
            return {k: _seq_fingerprint(s) for k, s in manager.seq_map.items()}
        scheduler = harness.scheduler
        return dict(
            iteration_id=scheduler._iteration_id,
            waiting=[s.seq_id for s in scheduler.waiting],
            running=[s.seq_id for s in scheduler.running],
            central_seqs=seqs(harness.engine_seqs),
            worker_seqs=seqs(harness.worker_seqs),
            central_tables=harness.central_tables(),
            worker_tables=harness.worker_tables(),
            central_free=scheduler.block_manager.get_num_free_gpu_blocks(),
            worker_free=harness.worker_seqs.block_manager.get_num_free_gpu_blocks(),
        )

    def test_decode_preemption_recomputes_expanded_context_to_completion(self):
        real_build_utilities = LPScheduler._build_utilities
        with self._fixed_utilities():
            self._run_case()
        self.assertIs(LPScheduler._build_utilities, real_build_utilities)

    def _run_case(self):
        A, B = self.A, self.B
        harness = _Harness(num_gpu_blocks=self.POOL, **self.CONFIG)
        scheduler = harness.scheduler
        central_manager = scheduler.block_manager
        worker_manager = harness.worker_seqs.block_manager
        a = harness.add(A, len(self.PROMPT_A), 1.0, self.MAX_TOKENS[A])
        self.assertEqual(a.prompt_token_ids, self.PROMPT_A)
        worker_a = harness.worker_seqs.get_seq(A)
        b = worker_b = None
        seqs = {A: a}

        pending = {k: list(v) for k, v in self.DECODE_TOKENS.items()}
        decode_events = []  # (step, seq_id, token) as sampled
        self.trace = trace = []

        for step, (preempted, emitted, after) in enumerate(self.EXPECTED):
            now = 1.0 + step

            def sample(m):
                if m.prompt_chunk_len > 0:
                    return self.PREFILL_SAMPLE
                token = pending[m.seq_id].pop(0)
                decode_events.append((step, m.seq_id, token))
                return token

            if step == self.BOUNDARY_STEP:
                # A has generated exactly one token and is an unfinished,
                # paused resident holding two blocks, with two free in each
                # manager. B is added only now, with no batch in flight.
                for copy_ in (a, worker_a):
                    self.assertEqual(copy_.get_status(), SequenceStatus.PAUSED)
                    self.assertEqual(copy_.output_token_ids, [101])
                    self.assertEqual(copy_.prompt_token_ids, self.PROMPT_A)
                    self.assertTrue(copy_.prompt_processing_finished)
                self.assertEqual(
                    (len(harness.central_tables()[A]), len(harness.worker_tables()[A])),
                    (2, 2),
                )
                self.assertEqual(
                    (central_manager.get_num_free_gpu_blocks(),
                     worker_manager.get_num_free_gpu_blocks()), (2, 2),
                )
                self.assertEqual(scheduler.num_running_batches, 0)
                b = harness.add(B, len(self.PROMPT_B), now, self.MAX_TOKENS[B])
                self.assertEqual(b.prompt_token_ids, self.PROMPT_B)
                worker_b = harness.worker_seqs.get_seq(B)
                seqs[B] = b

            unfinished = sorted(k for k, s in seqs.items() if not s.is_finished())
            before = self._state(harness)
            record = dict(step=step, now=now, before=before)
            trace.append(record)

            # Central block operations during scheduling: (op, seq_id, blocks).
            central_ops = []
            real_free, real_allocate = central_manager.free, central_manager.allocate
            tables = central_manager.block_tables

            def central_free(seq):
                central_ops.append(("free", seq.seq_id, len(tables[seq.seq_id])))
                real_free(seq)

            def central_allocate(seq):
                real_allocate(seq)
                central_ops.append(("allocate", seq.seq_id, len(tables[seq.seq_id])))

            harness.clock.reset_mock()
            with _Spies() as spies, \
                    mock.patch.object(central_manager, "free", central_free), \
                    mock.patch.object(central_manager, "allocate", central_allocate):
                outputs = harness.schedule(now)
            self._assert_synchronous_pipeline(harness, spies, now, outputs)
            snapshot, result = spies.calls[1][4], spies.calls[2][2]
            problem = snapshot.lp_problem
            requests = {r.raw_seq_id: r for r in snapshot.requests}
            relaxed = {d.request_id: d for d in result.relaxed.decisions}
            record.update(
                snapshot=snapshot, result=result, central_ops=list(central_ops),
                outputs=dict(id=outputs.id,
                             preempted=list(outputs.preempted_seq_ids),
                             scheduled=_emitted(outputs)),
                after_execution=self._state(harness),
            )

            # The fixed per-request utilities reached the mapped problem for
            # exactly the arrived, unfinished requests.
            self.assertEqual(sorted(requests), unfinished)
            for raw_id, r in requests.items():
                self.assertEqual(r.utility, self.UTILITIES[raw_id])
            self.assertEqual((problem.m_free, problem.w), (before["central_free"], 0))

            self.assertEqual(outputs.id, step)
            self.assertEqual(outputs.preempted_seq_ids, preempted)
            self.assertEqual(outputs.ignored_seq_ids, [])
            self.assertEqual(_emitted(outputs), emitted)
            if step != self.BOUNDARY_STEP:
                self.assertTrue(all(d.preempt == 0 for d in result.plan.decisions))

            replay_log = []
            if step == self.BOUNDARY_STEP:
                self._assert_boundary(
                    harness, problem, requests, relaxed, result, central_ops,
                    a, b, before,
                )
                # Replay order: central reset, worker reset and local free of
                # A, then B's worker allocation.
                real_reset = Sequence.reset_for_recompute
                real_wfree, real_wallocate = worker_manager.free, worker_manager.allocate

                def reset(seq):
                    copy_name = ("central" if seq is a else
                                 "worker" if seq is worker_a else seq.seq_id)
                    replay_log.append(("reset", copy_name, list(seq.prompt_token_ids),
                                       list(seq.output_token_ids)))
                    real_reset(seq)

                def worker_free(seq):
                    replay_log.append(("free", seq.seq_id))
                    real_wfree(seq)

                def worker_allocate(seq):
                    replay_log.append(("allocate", seq.seq_id))
                    real_wallocate(seq)

                with mock.patch.object(Sequence, "reset_for_recompute", reset), \
                        mock.patch.object(worker_manager, "free", worker_free), \
                        mock.patch.object(worker_manager, "allocate", worker_allocate):
                    harness.replay(outputs, sample)
                self.assertEqual(replay_log, [
                    ("reset", "central", self.PROMPT_A, [101]),
                    ("reset", "worker", self.PROMPT_A, [101]),
                    ("free", A),
                    ("allocate", B),
                ])
            elif step == self.READMISSION_STEP:
                # The waiting, unallocated A is admitted with its full
                # expanded 6-token context (2 blocks) and recomputed from
                # prompt token zero.
                r = requests[A]
                self.assertEqual(
                    (r.ownership, r.status, r.physical_block_count,
                     r.prompt_len, r.prompt_tokens_processed,
                     r.prefill_fixed_charge),
                    (lsm.OWNERSHIP_WAITING, "WAITING", 0, 6, 0, 2),
                )
                self.assertEqual(central_ops, [("allocate", A, 2)])
                harness.replay(outputs, sample)
            else:
                harness.replay(outputs, sample)
            record.update(replay_log=replay_log, after=self._state(harness))

            # Completed-step boundary: progress, outputs, ownership, and
            # allocation counts agree between the central and worker copies.
            self.assertEqual(scheduler.num_running_batches, 0)
            self._assert_managers_agree(harness)
            self.assertEqual(sorted(harness.engine_seqs.seq_map), sorted(after))
            self.assertEqual(sorted(harness.worker_seqs.seq_map), sorted(after))
            for seq_id, (processed, output_ids) in after.items():
                for copy_ in (harness.engine_seqs.get_seq(seq_id),
                              harness.worker_seqs.get_seq(seq_id)):
                    self.assertEqual(copy_.get_num_prompt_tokens_processed(), processed)
                    self.assertEqual(copy_.output_token_ids, output_ids)
                    self.assertNotIn(self.PREFILL_SAMPLE, copy_.get_token_ids())
            if step == 2:
                first_epoch_outputs = list(a.output_token_ids)
            if step == self.BOUNDARY_STEP:
                # A was reset centrally and on the worker: its expanded
                # prompt holds the first generated token exactly once.
                for copy_ in (a, worker_a):
                    self.assertEqual(copy_.get_status(), SequenceStatus.WAITING)
                    self.assertEqual(copy_.prompt_token_ids, self.PROMPT_A + [101])
                    self.assertFalse(copy_.prompt_processing_finished)
                    self.assertEqual(len(copy_.logical_token_blocks), 2)
                self.assertEqual(scheduler.waiting, [a])
                self.assertEqual(scheduler.running, [b])
                self.assertNotIn(A, harness.worker_tables())
                self.assertEqual(len(harness.worker_tables()[B]), 3)

        # Native stopping completed B first and A later.
        self.assertEqual(b.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
        self.assertEqual(a.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
        self.assertEqual(worker_b.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
        self.assertEqual(worker_a.get_status(), SequenceStatus.FINISHED_LENGTH_CAPPED)
        self.assertEqual(
            [(s, k) for s, k, _ in decode_events],
            [(2, A), (6, B), (9, A), (10, A)],
        )
        self.assertEqual(pending, {A: [], B: []})
        self.assertEqual(b.prompt_token_ids, self.PROMPT_B)
        self.assertEqual(b.output_token_ids, [201])

        # A's cumulative history comes from its completed decode events:
        # the first token before the reset, then two after recomputation.
        # The native final state keeps the first token in the expanded prompt
        # and only the post-reset tokens as output IDs. Three tokens for a
        # 2-token cap is the inherited generation-limit behavior (16.1).
        history = [t for _, k, t in decode_events if k == A]
        self.assertEqual(history, [101, 102, 103])
        self.assertEqual(first_epoch_outputs, [101])
        self.assertEqual(a.prompt_token_ids, self.PROMPT_A + [101])
        self.assertEqual(a.output_token_ids, [102, 103])
        self.assertEqual(first_epoch_outputs + a.output_token_ids, history)
        self.assertEqual(len(history), self.MAX_TOKENS[A] + 1)

        # Final release in both managers and empty ownership.
        self.assertEqual((scheduler.waiting, scheduler.running), ([], []))
        self.assertEqual(
            (harness.engine_seqs.seq_map, harness.worker_seqs.seq_map), ({}, {}),
        )
        self.assertEqual((harness.central_tables(), harness.worker_tables()), ({}, {}))
        for manager in (central_manager, worker_manager):
            self.assertEqual(manager.get_num_free_gpu_blocks(), self.POOL)
        self._assert_managers_agree(harness)

        # One ordinary idle call; the inherited iteration counter advances.
        self.assertEqual(scheduler._iteration_id, len(self.EXPECTED) - 1)
        before = _fingerprint(scheduler)
        harness.clock.reset_mock()
        with _Spies() as spies:
            outputs = harness.schedule(100.0)
        self.assertTrue(outputs.has_no_output())
        self.assertEqual(spies.calls, [])
        self.assertEqual(outputs.id, len(self.EXPECTED))
        self.assertEqual(scheduler._iteration_id, len(self.EXPECTED))
        self.assertEqual(scheduler.num_running_batches, 0)
        self.assertEqual(_fingerprint(scheduler), before)
        trace.append(dict(idle=dict(output_id=outputs.id,
                                    after=self._state(harness))))
        self.decode_events = decode_events

    def _assert_boundary(self, harness, problem, requests, relaxed, result,
                         central_ops, a, b, before):
        A, B = self.A, self.B
        tol = NUMERICAL_POLICY.feasibility_tol
        scheduler = harness.scheduler
        ra, rb = requests[A], requests[B]

        # Mapped state: A is the only legal victim, a prompt-complete
        # resident with one generated token, two blocks, and recovery two;
        # B waits unallocated with fixed admission charge three.
        self.assertEqual(problem.legal_preemption_ids, frozenset({str(A)}))
        self.assertEqual((problem.m_free, problem.w), (2, 0))
        self.assertEqual(before["central_seqs"][A][3], (101,))
        self.assertEqual(
            (ra.ownership, ra.status, ra.prompt_len, ra.prompt_tokens_processed,
             ra.prompt_processing_finished, ra.physical_block_count,
             ra.preemption_eligible, ra.preemption_recovery,
             ra.decode_eligible, ra.decode_charge, ra.prefill_eligible),
            (lsm.OWNERSHIP_RUNNING, "PAUSED", 5, 5, True, 2,
             True, 2, True, 1, False),
        )
        self.assertEqual(
            (rb.ownership, rb.status, rb.prompt_len, rb.physical_block_count,
             rb.preemption_eligible, rb.prefill_fixed_charge),
            (lsm.OWNERSHIP_WAITING, "WAITING", 12, 0, False, 3),
        )

        # Relaxed solution and integer extraction.
        for rid, values in ((str(A), (0, 0, 0, 0.5)), (str(B), (4, 0, 1, 0))):
            d = relaxed[rid]
            for got, want in zip((d.x, d.y, d.prefill_indicator, d.z), values):
                self.assertAlmostEqual(got, want, delta=tol)
        self.assertAlmostEqual(result.relaxed.normalized_objective, 79.5, delta=tol)
        self.assertEqual(
            [(d.request_id, d.prefill_tokens, d.decode, d.preempt)
             for d in result.plan.decisions],
            [(str(A), 0, 0, 1), (str(B), 4, 0, 0)],
        )
        self.assertEqual(result.plan.dominant_preemption_ids, (str(A),))
        self.assertEqual(result.plan.safety_preemption_ids, ())
        self.assertAlmostEqual(result.plan.objective, 79.0, delta=tol)

        # Central execution: A's two blocks are freed through native
        # preemption before B's three-block admission; A returns to waiting
        # with its prompt and generated token untouched until replay.
        self.assertEqual(central_ops, [("free", A, 2), ("allocate", B, 3)])
        self.assertEqual(scheduler.waiting, [a])
        self.assertEqual(scheduler.running, [b])
        self.assertEqual(a.get_status(), SequenceStatus.PAUSED)
        self.assertEqual(a.prompt_token_ids, self.PROMPT_A)
        self.assertEqual(a.output_token_ids, [101])
        self.assertEqual(a.get_num_prompt_tokens_processed(), 5)
        self.assertEqual(set(harness.central_tables()), {B})
        self.assertEqual(scheduler.block_manager.get_num_free_gpu_blocks(), 1)
        self.assertEqual(set(harness.worker_tables()), {A})


class PreMutationFailureTest(unittest.TestCase):
    def _assert_failure(self, harness, now, expected_calls):
        scheduler = harness.scheduler
        before = _fingerprint(scheduler)
        iteration = scheduler._iteration_id
        with _Spies() as spies:
            with self.assertRaises(LPSchedulingError) as caught:
                harness.schedule(now)
        self.assertEqual(spies.names(), expected_calls)
        self.assertEqual(scheduler._iteration_id, iteration + 1)
        self.assertEqual(_fingerprint(scheduler), before)
        error = caught.exception
        self.assertTrue(dataclasses.is_dataclass(error.failure))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            error.failure.reason = "changed"
        self.assertEqual(
            (error.stage, error.category, error.reason),
            (error.failure.stage, error.failure.category, error.failure.reason),
        )
        return error, spies

    def test_mapping_failure_stops_before_solver(self):
        # Case-specific: a decode policy identifier the mapper does not accept.
        harness = _Harness(decode_memory_policy_id="exact_gap_v1")
        harness.add(0, 4, arrival_time=1.0, max_tokens=2)
        error, spies = self._assert_failure(harness, 2.0, ["utilities", "map"])
        self.assertIsInstance(error.failure, lsm.MappingFailure)
        self.assertEqual(error.stage, lsm.STAGE_STATE_MAPPING)
        self.assertIn("decode_memory_policy_id", error.reason)
        self.assertIsNone(error.solver_diagnostics)

    def test_mathematical_failure_stops_before_executor(self):
        # Case-specific: a reserve above the 10 free blocks makes the LP
        # infeasible even for the all-zero plan.
        harness = _Harness(memory_reserve=11)
        harness.add(0, 4, arrival_time=1.0, max_tokens=2)
        error, spies = self._assert_failure(harness, 2.0, ["utilities", "map", "solve"])
        self.assertIsInstance(error.failure, lrs.Failure)
        self.assertEqual((error.stage, error.category), ("solver", "infeasible"))
        self.assertEqual(error.snapshot_id, "0")
        self.assertIsInstance(error.solver_diagnostics, lrs.SolverDiagnostics)
        self.assertEqual(error.solver_diagnostics.raw_status, 2)

    def test_physically_infeasible_plan_raises_executor_failure(self):
        # Case-specific resident limit 1: the LP has no resident constraint,
        # so it selects request 0's decode and request 1's admission, which
        # native precommit validation rejects.
        harness = _Harness(max_num_seqs=1)
        harness.add(0, 4, arrival_time=1.0, max_tokens=4)
        harness.add(1, 4, arrival_time=2.0, max_tokens=4)
        self.assertEqual(_emitted(harness.step(1.0)), [(0, 4)])
        error, spies = self._assert_failure(
            harness, 2.0, ["utilities", "map", "solve", "execute"],
        )
        self.assertIsInstance(error.failure, lrs.Failure)
        self.assertEqual(
            (error.stage, error.category),
            (lpe.STAGE_PRECOMMIT_VALIDATION, lpe.CATEGORY_RESIDENT_CAPACITY),
        )
        self.assertEqual(error.snapshot_id, "1")
        self.assertEqual(harness.scheduler.num_running_batches, 0)

    def test_no_progress_plan_is_not_ordinary_idle(self):
        # Case-specific reserve 9: one planning block remains, below the
        # 2-block admission charge, so the validated plan is all-zero.
        harness = _Harness(memory_reserve=9)
        harness.add(0, 8, arrival_time=1.0, max_tokens=2)
        error, spies = self._assert_failure(
            harness, 2.0, ["utilities", "map", "solve", "execute"],
        )
        self.assertEqual(error.category, lpe.CATEGORY_NO_PROGRESS)
        self.assertEqual(error.snapshot_id, "0")
        _, _, result = spies.calls[2]
        self.assertIsInstance(result, lrs.SchedulingSuccess)
        self.assertEqual(
            [(d.prefill_tokens, d.decode, d.preempt) for d in result.plan.decisions],
            [(0, 0, 0)],
        )


class UnsupportedEntryAndDestructiveFailureTest(unittest.TestCase):
    def test_changed_stage_count_fails_before_base_scheduling(self):
        harness = _Harness()
        harness.add(0, 4, arrival_time=1.0, max_tokens=2)
        harness.scheduler.scheduler_config.num_pipeline_stages = 2
        before = _fingerprint(harness.scheduler)
        with _Spies() as spies:
            with self.assertRaises(LPSchedulingError) as caught:
                harness.schedule(2.0)
        self.assertEqual(caught.exception.stage, lp_scheduler.STAGE_SCHEDULER_ENTRY)
        self.assertIsNone(caught.exception.snapshot_id)
        self.assertEqual(harness.scheduler._iteration_id, -1)
        self.assertEqual(spies.calls, [])
        self.assertEqual(harness.clock.monotonic.call_count, 0)
        self.assertEqual(_fingerprint(harness.scheduler), before)

    def test_nonzero_running_batches_fails_before_base_scheduling(self):
        harness = _Harness()
        scheduler = harness.scheduler
        harness.add(0, 8, arrival_time=1.0, max_tokens=2)
        first = harness.schedule(1.0)
        self.assertEqual(_emitted(first), [(0, 4)])
        self.assertEqual(scheduler.num_running_batches, 1)

        # Scheduling again before completion: the inherited method would
        # return an ordinary empty output here.
        before = _fingerprint(scheduler)
        harness.clock.reset_mock()
        with _Spies() as spies:
            with self.assertRaises(LPSchedulingError) as caught:
                harness.schedule(2.0)
        error = caught.exception
        self.assertEqual(
            (error.stage, error.category),
            (lp_scheduler.STAGE_SCHEDULER_ENTRY, lp_scheduler.CATEGORY_UNSUPPORTED_STATE),
        )
        self.assertIn("num_running_batches=1", error.reason)
        self.assertEqual(scheduler._iteration_id, 0)
        self.assertEqual(spies.calls, [])
        self.assertEqual(harness.clock.monotonic.call_count, 0)
        self.assertEqual(_fingerprint(scheduler), before)

    def test_post_mutation_failure_propagates_without_recovery(self):
        harness = _Harness()
        scheduler = harness.scheduler
        resident = harness.add(0, 4, arrival_time=1.0, max_tokens=4)
        first = harness.add(1, 3, arrival_time=2.0, max_tokens=4)
        second = harness.add(2, 3, arrival_time=2.0, max_tokens=4)
        self.assertEqual(_emitted(harness.step(1.0)), [(0, 4)])
        resident_table = harness.central_tables()[0]

        # Planned order: admit 1, admit 2, decode 0. The second native
        # allocation fails after the first admission has mutated state.
        real_allocate = scheduler._allocate
        allocations = []

        def allocate(seq):
            allocations.append(seq.seq_id)
            if len(allocations) == 2:
                raise RuntimeError("induced allocation failure")
            real_allocate(seq)

        with mock.patch.object(scheduler, "_allocate", allocate), \
                mock.patch.object(scheduler, "_append_slot") as append_slot, \
                _Spies() as spies:
            with self.assertRaises(RuntimeError) as caught:
                harness.schedule(2.0)

        self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual(str(caught.exception), "induced allocation failure")
        self.assertEqual(spies.names(), ["utilities", "map", "solve", "execute"])
        self.assertIsNone(spies.calls[3][3])
        self.assertEqual(allocations, [1, 2])
        append_slot.assert_not_called()
        # The first admission mutated state; the failed admission was already
        # removed from waiting. No output was returned and the batch count
        # was not raised. Nothing was rolled back.
        self.assertIn(first, scheduler.running)
        self.assertIn(1, scheduler.block_manager.block_tables)
        self.assertNotIn(second, scheduler.waiting + scheduler.running)
        self.assertEqual(harness.central_tables()[0], resident_table)
        self.assertEqual(resident.get_output_len(), 0)
        self.assertEqual(scheduler.num_running_batches, 0)
        self.assertEqual(scheduler._iteration_id, 1)
        # The mutated fixture is discarded, not recovered or reused.
        del harness, scheduler


if __name__ == "__main__":
    unittest.main()
