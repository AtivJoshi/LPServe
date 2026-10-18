"""CPU checks for the scheduler comparison GPU driver.

No engine, Ray worker, model, or GPU is used. Each of the six runs goes
through the driver's ``run_comparison`` with the registered scheduler built
from the driver's native ``EngineArgs`` (model metadata mocked), the central
block manager, the engine sequence manager, and a worker sequence manager
holding separate copies, replayed in the engine's single-stage order at the
run's block size 16, 8-block pool, budgets, and output limits. Sampled
tokens are synthetic; no token ID or tied LP choice is prescribed by the
driver. The expected summaries below are the earlier bounded CPU comparison
(block size 4) rows, which this block-size-16 fixture is checked against.
"""

import copy
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_scheduler_gpu as driver  # noqa: E402
import test_lp_scheduler as lpt  # noqa: E402

import lp_relaxation_scheduler as lrs  # noqa: E402
import lpserve_plan_execution as lpe  # noqa: E402
import lpserve_state_mapping as lsm  # noqa: E402
from sarathi.config import (  # noqa: E402
    AdaptedVLLMSchedulerConfig,
    CacheConfig,
    LPSchedulerConfig,
)
from sarathi.core.datatypes.sequence import SamplerOutput, Sequence  # noqa: E402
from sarathi.core.scheduler.lp_scheduler import LPSchedulingError  # noqa: E402
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry  # noqa: E402
from sarathi.core.sequence_manager.worker_sequence_manager import (  # noqa: E402
    WorkerSequenceManager,
)
from sarathi.engine import arg_utils  # noqa: E402

setUpModule = lpt.setUpModule
tearDownModule = lpt.tearDownModule

SAMPLED_TOKEN = 7
RUNS = [(p, n) for n in driver.RESIDENT_LIMITS for p in driver.POLICIES]

# Earlier CPU comparison rows (validation_output/adapted_vllm_reference/
# 20261018T051258Z/comparison.txt): decisions, first prefill, first decode,
# completion (A, B), preemptions, outcome.
EARLIER_CPU = {
    ("adapted_vllm", 2): (4, (0, 0), (1, 2), (2, 3), 0, "drained"),
    ("lp_uniform", 2): (6, (1, 0), (4, 2), (5, 3), 0, "drained"),
    ("lp_rank_based", 2): (6, (0, 1), (2, 4), (3, 5), 0, "drained"),
    ("adapted_vllm", 1): (6, (0, 3), (1, 4), (2, 5), 0, "drained"),
    ("lp_uniform", 1): (8, (1, 0), (None, None), (None, None), 7,
                        "bounded_unfinished"),
    ("lp_rank_based", 1): (8, (0, 1), (None, None), (None, None), 7,
                           "bounded_unfinished"),
}


def native_configs(policy, limit, tmp):
    """The driver's ``EngineArgs`` configurations, model metadata mocked."""
    model_config = mock.Mock()
    model_config.get_max_model_len.return_value = driver.MAX_MODEL_LEN
    model_config.get_total_num_layers.return_value = 2
    with mock.patch.object(arg_utils, "ModelConfig",
                           return_value=model_config) as cls:
        args = driver.engine_args(policy, limit, Path(tmp) / "model",
                                  Path(tmp) / "metrics")
        configs = args.create_engine_configs()
    return cls.call_args.kwargs, configs


class CpuRuntime:
    """``run_comparison`` runtime with native central and worker replay."""

    def __init__(self, policy, limit):
        with tempfile.TemporaryDirectory() as tmp:
            _, (_, cache_config, _, sched_config, _) = native_configs(
                policy, limit, tmp)
        cache_config.num_gpu_blocks = driver.NUM_GPU_BLOCKS  # as _init_cache
        self.scheduler = SchedulerRegistry.get(sched_config.type, sched_config,
                                               cache_config)
        self.engine_seqs = lpt._EngineSequenceManager()
        self.seq_map = self.engine_seqs.seq_map
        self.worker_seqs = WorkerSequenceManager(cache_config, sched_config)
        self.worker_ops = []
        driver.instrument_block_manager(self.worker_seqs.block_manager,
                                        self.worker_ops)
        self.lp = policy != "adapted_vllm"

    def submit(self, prompt_ids, sampling_params):
        # Same construction, arrival clock, and LP check as add_request.
        seq = Sequence(len(self.seq_map), None, prompt_ids, driver.BLOCK_SIZE,
                       2, time.perf_counter(), sampling_params)
        if self.lp:
            self.scheduler.check_new_request(seq)
        self.engine_seqs.add_seq(seq)
        self.worker_seqs.add_seq(copy.deepcopy(seq))
        self.scheduler.add_seq(seq)

    def step(self):
        outputs = self.scheduler.schedule()
        if outputs.is_empty():
            return [], None
        ignored, metadata = self.engine_seqs.on_schedule(outputs)
        self.worker_seqs.on_schedule(outputs)
        samples = [SamplerOutput(m.seq_id, SAMPLED_TOKEN)
                   for m in outputs.scheduled_seq_metadata_list]
        self.worker_seqs.on_step_completed(outputs, samples)
        self.engine_seqs.on_step_completed(outputs, samples)
        self.scheduler.on_step_completed()
        return (self.engine_seqs.generate_request_outputs(ignored, metadata),
                [[s.seq_id, s.output_token] for s in samples])

    def worker_state(self):
        return driver.adapted.worker_validation_state(self.worker_seqs,
                                                      self.worker_ops)


def _run(policy, limit):
    runtime, trace = CpuRuntime(policy, limit), {}
    outcome = driver.run_comparison(policy, limit, runtime, trace)
    return runtime, trace, outcome


def _row(trace, outcome):
    acc = trace["final"]["accounting"]
    pair = lambda key: (acc["A"][key], acc["B"][key])  # noqa: E731
    return (len(trace["decisions"]), pair("first_prefill"),
            pair("first_decode"), pair("completion"),
            acc["A"]["preemptions"] + acc["B"]["preemptions"], outcome)


class DriverImportAndHelpTest(unittest.TestCase):
    MODULES = ("torch", "ray", "sarathi.engine.base_llm_engine",
               "sarathi.worker.base_worker", "sarathi.engine.arg_utils",
               "lp_relaxation_scheduler")

    def test_import_and_help_load_no_engine_worker_ray_or_torch(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import compare_scheduler_gpu as d\n"
            "try:\n    d.main(['--help'])\nexcept SystemExit as e:\n"
            "    code = e.code\n"
            f"print('RESULT', code, sorted(m for m in {self.MODULES!r} "
            "if m in sys.modules))")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=REPO_ROOT, env=env, capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--policy {adapted_vllm,lp_uniform,lp_rank_based}",
                      result.stdout)
        self.assertIn("--resident-limit {2,1}", result.stdout)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "RESULT 0 []")


class RunConfigurationTest(unittest.TestCase):

    def test_shared_inputs(self):
        self.assertEqual(
            (driver.BLOCK_SIZE, driver.NUM_GPU_BLOCKS, driver.MAX_MODEL_LEN,
             driver.GPU_MEMORY_UTILIZATION, driver.SEED, driver.DTYPE,
             driver.ATTENTION_BACKEND, driver.TENSOR_PARALLEL_SIZE,
             driver.PIPELINE_PARALLEL_SIZE, driver.TEMPERATURE,
             driver.IGNORE_EOS, driver.MAX_TOKENS, driver.TOKEN_BUDGET,
             driver.MAX_DECISIONS, driver.RESIDENT_LIMITS),
            (16, 8, 32, 0.5, 42, "float16", "flash_attention", 1, 1, 0.0,
             True, 2, 8, 8, (2, 1)))
        a, b = driver.PROMPTS["A"], driver.PROMPTS["B"]
        self.assertEqual((len(a), len(b)), (5, 8))
        self.assertFalse(set(a) & set(b))
        self.assertEqual((driver.LP_B_MAX, driver.LP_C_MAX, driver.LP_S_MAX),
                         (8, 8, 1))
        self.assertEqual(driver.NUMERICAL_POLICY_FIELDS,
                         dataclasses.asdict(lrs.NumericalPolicy()))

    def test_all_six_native_configurations(self):
        for policy, limit in RUNS:
            with self.subTest(policy=policy, limit=limit), \
                    tempfile.TemporaryDirectory() as tmp:
                kwargs, (_, cache, parallel, sched, _) = native_configs(
                    policy, limit, tmp)
                self.assertEqual((kwargs["load_format"], kwargs["dtype"],
                                  kwargs["max_model_len"],
                                  kwargs["attention_backend"]),
                                 ("dummy", "float16", 32, "flash_attention"))
                self.assertEqual((parallel.tensor_parallel_size,
                                  parallel.pipeline_parallel_size,
                                  cache.block_size, cache.gpu_memory_utilization),
                                 (1, 1, 16, 0.5))
                self.assertEqual((sched.max_num_seqs, sched.max_model_len,
                                  sched.num_pipeline_stages,
                                  sched.max_num_batched_tokens),
                                 (limit, 32, 1, 8))
                params = driver.run_parameters(policy, limit)
                self.assertEqual(params["native_watermark_blocks"], 0)
                if policy == "adapted_vllm":
                    self.assertIs(type(sched), AdaptedVLLMSchedulerConfig)
                    self.assertEqual((params["chunk_cap"], params["action_cap"]),
                                     (None, None))
                else:
                    self.assertIs(type(sched), LPSchedulerConfig)
                    self.assertEqual(
                        (sched.b_max, sched.c_max, sched.s_max,
                         sched.decode_utility, sched.prefill_token_utility,
                         sched.preemption_penalty),
                        (8, 8, 1, 1.0, 1.0, 1.0))
                    self.assertEqual(sched.numerical_policy,
                                     lrs.NumericalPolicy(
                                         **driver.NUMERICAL_POLICY_FIELDS))

    def test_coefficient_policy_selection(self):
        uniform = CpuRuntime("lp_uniform", 2).scheduler
        driver.select_coefficient_policy(uniform, "lp_uniform")
        self.assertEqual(uniform._coefficient_policy(),
                         lsm.RequestUtility(1.0, 1.0, 1.0))
        ranked = CpuRuntime("lp_rank_based", 2).scheduler
        driver.select_coefficient_policy(ranked, "lp_rank_based")
        policy = ranked._coefficient_policy()
        self.assertIs(policy.func, lsm.rank_based_utility)
        self.assertEqual(policy.keywords, driver.RANK_SCALES)
        # Instance override only; the class hook is unchanged.
        self.assertNotIn("_coefficient_policy", vars(uniform))
        self.assertEqual(CpuRuntime("lp_rank_based", 1).scheduler
                         ._coefficient_policy(), lsm.RequestUtility(1.0, 1.0, 1.0))


class NativeCpuReplayTest(unittest.TestCase):

    def test_six_runs_through_native_central_and_worker_replay(self):
        originals = [getattr(m, n) for m, n in (
            (lsm, "map_scheduler_state"), (lrs, "solve_and_extract"),
            (lpe, "execute_plan"))]
        for policy, limit in RUNS:
            with self.subTest(policy=policy, limit=limit):
                runtime, trace, outcome = _run(policy, limit)
                self.assertEqual(_row(trace, outcome), EARLIER_CPU[policy, limit])
                for d in trace["decisions"]:
                    if policy == "adapted_vllm":
                        self.assertEqual(len(d["proposals"]), 1)
                        self.assertEqual(d["lp_events"], [])
                    else:
                        self.assertEqual([e["call"] for e in d["lp_events"]],
                                         driver.LP_CALLS)
                        plan = d["lp_events"][1]["result"]["plan"]
                        self.assertEqual(
                            plan["extraction"]["fractional_request_count"], 0)
                acc = trace["final"]["accounting"]
                if outcome == driver.DRAINED:
                    self.assertEqual(trace["final"]["unfinished"], [])
                    self.assertEqual(trace["idle"]["after"]["free_blocks"], 8)
                    self.assertEqual((acc["A"]["generated_token_events"],
                                      acc["B"]["generated_token_events"]), (2, 2))
                else:
                    self.assertEqual(trace["final"]["unfinished"], ["A", "B"])
                    self.assertIsNone(trace["idle"])
                    self.assertTrue(runtime.scheduler.has_unfinished_seqs())
                    # Restarts happen before any output, so nothing generated.
                    self.assertEqual(acc["A"]["discarded_generated_tokens"]
                                     + acc["B"]["discarded_generated_tokens"], 0)
                    self.assertGreater(acc["A"]["prefill_tokens_executed"]
                                       + acc["B"]["prefill_tokens_executed"], 13)
        # The wrapped module functions are restored after every run.
        self.assertEqual([lsm.map_scheduler_state, lrs.solve_and_extract,
                          lpe.execute_plan], originals)

    def test_limit_one_preemptions_are_voluntary(self):
        """Every LP victim at limit 1 has decode demand 0 and the pool has
        free blocks, so no preemption is needed for memory."""
        for policy in ("lp_uniform", "lp_rank_based"):
            _, trace, _ = _run(policy, 1)
            victims = 0
            for d in trace["decisions"][1:]:
                snap = d["lp_events"][0]["result"]
                self.assertGreaterEqual(snap["free_physical_blocks"], 6)
                self.assertTrue(all(r["decode_demand"] == 0
                                    for r in snap["requests"]))
                victims += len(d["schedules"][0]["outputs"]["preempted_seq_ids"])
            self.assertEqual(victims, 7)

    def test_wrappers_return_the_native_result_unchanged(self):
        runtime = CpuRuntime("lp_uniform", 2)
        sentinel = object()
        native = mock.Mock(return_value=sentinel)
        with mock.patch.object(lrs, "solve_and_extract", native):
            observer = driver.DecisionObserver(runtime.scheduler,
                                               runtime.seq_map, "lp_uniform")
            self.assertIs(lrs.solve_and_extract("problem", k=1), sentinel)
            native.assert_called_once_with("problem", k=1)
            self.assertEqual(observer.take()["lp_events"][0]["call"],
                             "solve_and_extract")
            observer.close()
            self.assertIs(lrs.solve_and_extract, native)


class BoundAndFailureTest(unittest.TestCase):

    def test_decision_bound_leaves_work_visibly_unfinished(self):
        with mock.patch.object(driver, "MAX_DECISIONS", 3):
            runtime, trace, outcome = _run("adapted_vllm", 2)
        self.assertEqual(outcome, driver.BOUNDED_UNFINISHED)
        self.assertEqual(len(trace["decisions"]), 3)
        self.assertEqual(trace["final"]["unfinished"], ["B"])
        self.assertEqual(trace["final"]["central"]["running"], [1])
        self.assertIsNone(trace["idle"])
        self.assertTrue(runtime.scheduler.has_unfinished_seqs())

    def test_solver_failure_is_a_failure_with_partial_trace(self):
        def failing(problem):
            return lrs.Failure(problem.problem_id, lrs.STAGE_SOLVER,
                               lrs.OUTCOME_NUMERICAL, "injected", "test")
        trace = {}
        runtime = CpuRuntime("lp_rank_based", 2)
        with mock.patch.object(lrs, "solve_and_extract", failing):
            with self.assertRaises(LPSchedulingError) as caught:
                driver.run_comparison("lp_rank_based", 2, runtime, trace)
        self.assertEqual(caught.exception.category, "injected")
        record = trace["decisions"][0]
        self.assertEqual([e["call"] for e in record["lp_events"]],
                         driver.LP_CALLS[:2])
        self.assertEqual(record["lp_events"][1]["result_type"], "Failure")
        self.assertEqual(record["central_at_failure"]["waiting"], [0, 1])
        self.assertEqual(record["central_ops"], [])

    def test_check_detects_a_contract_violation(self):
        # An emitted order other than the reference's prompt-first layout.
        native = driver.check_reference_decision

        def reversed_layout(*args):
            victims, layout = native(*args)
            return victims, layout[::-1]
        with mock.patch.object(driver, "check_reference_decision",
                               reversed_layout):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "decision 0: emitted"):
                _run("adapted_vllm", 2)
        # Rank-based coefficients checked against the uniform triple.
        with mock.patch.object(driver, "expected_coefficients",
                               lambda policy, snap, r: driver.UNIFORM):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "decision 0: coefficients of"):
                _run("lp_rank_based", 2)


class CacheCapacityTest(unittest.TestCase):
    def _init(self, profiled):
        selection = driver.CacheCapacitySelection(chosen=driver.NUM_GPU_BLOCKS)
        engine_class = driver.build_engine_class(selection)
        engine = object.__new__(engine_class)
        engine.cache_config = CacheConfig(block_size=16,
                                          gpu_memory_utilization=0.5)
        engine.model_config = SimpleNamespace(max_model_len=32)
        calls = []

        def native(self, method, *args, **kwargs):
            calls.append(method)
            return [profiled] if method == driver.PROFILE_METHOD else [None]
        with mock.patch.object(engine_class.__mro__[1], "_run_workers", native):
            engine._init_cache()
        return engine, selection, calls

    def test_chosen_pool_reaches_native_initialization(self):
        engine, selection, calls = self._init(900)
        self.assertEqual(calls, [driver.PROFILE_METHOD, driver.INIT_CACHE_METHOD])
        self.assertEqual(engine.cache_config.num_gpu_blocks, 8)
        self.assertEqual(selection.record()["profiled_available_blocks"], [900])

    def test_shortfall_fails_before_initialization(self):
        with self.assertRaisesRegex(driver.CheckFailure, "below the chosen 8"):
            self._init(7)


class WorkerSerializationTest(unittest.TestCase):
    """The reused worker observation helpers load in a fresh interpreter
    that cannot import the script modules, only when pickled by value."""

    DUMP = (
        "import sys; sys.path[:0] = ['{root}', '{root}/scripts']\n"
        "import compare_scheduler_gpu as d\n"
        "from ray import cloudpickle\n"
        "cls = d.adapted.build_worker_class()\n"
        "open('{tmp}/by_reference.pkl', 'wb').write(cloudpickle.dumps(cls))\n"
        "d.register_worker_helpers_by_value()\n"
        "open('{tmp}/by_value.pkl', 'wb').write(cloudpickle.dumps(cls))\n"
        "print('ray started:', __import__('ray').is_initialized())\n")
    LOAD = (
        "import importlib.util, pickle, sys; sys.path.insert(0, '{root}')\n"
        "assert importlib.util.find_spec('check_adapted_vllm_gpu') is None\n"
        "try:\n"
        "    pickle.load(open('{tmp}/by_reference.pkl', 'rb'))\n"
        "    print('by_reference: loaded')\n"
        "except ModuleNotFoundError as e:\n"
        "    print('by_reference:', e.name)\n"
        "cls = pickle.load(open('{tmp}/by_value.pkl', 'rb'))\n"
        "from sarathi.config import AdaptedVLLMSchedulerConfig, CacheConfig\n"
        "from sarathi.core.datatypes.sampling_params import SamplingParams\n"
        "from sarathi.core.datatypes.sequence import Sequence\n"
        "from sarathi.core.sequence_manager.worker_sequence_manager import "
        "WorkerSequenceManager\n"
        "g = cls.validation_state.__globals__\n"
        "cache = CacheConfig(block_size=16, gpu_memory_utilization=0.5)\n"
        "cache.num_gpu_blocks = 8\n"
        "manager = WorkerSequenceManager(cache, "
        "AdaptedVLLMSchedulerConfig(2, 32, 1, 8))\n"
        "ops = []\n"
        "cls.init_cache_engine.__globals__['instrument_block_manager']("
        "manager.block_manager, ops)\n"
        "seq = Sequence(0, None, [1, 2, 3], 16, 2, 0.0, SamplingParams())\n"
        "manager.add_seq(seq); manager.block_manager.allocate(seq)\n"
        "state = g['worker_validation_state'](manager, ops)\n"
        "print('by_value:', cls.__name__, [o[:2] for o in state['block_ops']],"
        " state['free_blocks'], 'check_adapted_vllm_gpu' in sys.modules)\n")

    def test_worker_helpers_pickle_by_value(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as tmp:
            fmt = dict(root=REPO_ROOT, tmp=tmp)
            dump = subprocess.run(
                [sys.executable, "-B", "-c", self.DUMP.format(**fmt)],
                cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(dump.returncode, 0, dump.stderr)
            self.assertIn("ray started: False", dump.stdout)
            load = subprocess.run(
                [sys.executable, "-B", "-c", self.LOAD.format(**fmt)],
                cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(load.returncode, 0, load.stderr)
        self.assertIn("by_reference: check_adapted_vllm_gpu", load.stdout)
        self.assertIn("by_value: ValidationWorker [['allocate', 0]] 7 False",
                      load.stdout)


class EvidenceTest(unittest.TestCase):
    def test_trace_serialization(self):
        _, trace, outcome = _run("lp_rank_based", 1)
        loaded = json.loads(json.dumps(driver.to_jsonable(trace)))
        self.assertEqual(outcome, driver.BOUNDED_UNFINISHED)
        self.assertEqual(len(loaded["decisions"]), 8)
        lp = loaded["decisions"][1]["lp_events"]
        self.assertEqual(lp[1]["result"]["plan"]["victim_order"], ["0"])
        self.assertIn("relaxed", lp[1]["result"])
        self.assertIn("raw_status", lp[1]["result"]["solver"]["diagnostics"])
        self.assertIn("utility", lp[0]["result"]["requests"][0])

    def test_failure_is_recorded_without_a_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            cmd = [sys.executable, "-B", "scripts/compare_scheduler_gpu.py",
                   "--policy", "lp_uniform", "--resident-limit", "1",
                   "--model-path", str(Path(tmp) / "absent"),
                   "--output-dir", str(out)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=str(REPO_ROOT))
            result = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("RESULT: FAIL workload: failure", result.stdout)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual((summary["validation_passed"],
                              summary["workload_outcome"]), (False, "failure"))
            self.assertIn("lacks config.json", summary["failure"]["message"])
            self.assertEqual(summary["run"]["s_max"], 1)
            self.assertIn("git_head", summary["provenance"])
            self.assertFalse(json.loads(
                (out / "cleanup.json").read_text())["engine_constructed"])
            again = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                   capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)
            self.assertIn("already holds run artifacts", again.stderr)


if __name__ == "__main__":
    unittest.main()
