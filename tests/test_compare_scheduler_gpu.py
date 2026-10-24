"""CPU checks for the scheduler comparison GPU driver.

No engine, Ray worker, model, or GPU is used. Each of the six two-request
runs and the three memory-pressure runs goes
through the driver's ``run_comparison`` with the registered scheduler built
from the driver's native ``EngineArgs`` (model metadata mocked), the central
block manager, the engine sequence manager, and a worker sequence manager
holding separate copies, replayed in the engine's single-stage order at the
run's block size 16, pool, budgets, and output limits. Sampled tokens are
synthetic and distinct per request and step; no token ID or tied LP choice
is prescribed by the driver. The expected two-request summaries below are
the earlier bounded CPU comparison (block size 4) rows, which this
block-size-16 fixture is checked against. Memory-pressure expectations come
from the reference specification and its earlier preemption-case run, the
LP decode prerequisite, and the strictly ordered rank-based coefficients;
the LP runs are not required to drain.
"""

import contextlib
import copy
import dataclasses
import io
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

# The earlier three policies, whose rows are preserved below.
EARLIER_POLICIES = ("adapted_vllm", "lp_uniform", "lp_rank_based")
RUNS = [(p, n) for n in driver.RESIDENT_LIMITS for p in EARLIER_POLICIES]
PRESSURE = driver.MEMORY_PRESSURE

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


def native_configs(policy, limit, tmp, case=driver.TWO_REQUEST):
    """The driver's ``EngineArgs`` configurations, model metadata mocked."""
    model_config = mock.Mock()
    model_config.get_max_model_len.return_value = driver.case_inputs(
        case)["max_model_len"]
    model_config.get_total_num_layers.return_value = 2
    with mock.patch.object(arg_utils, "ModelConfig",
                           return_value=model_config) as cls:
        args = driver.engine_args(policy, limit, Path(tmp) / "model",
                                  Path(tmp) / "metrics", case)
        configs = args.create_engine_configs()
    return cls.call_args.kwargs, configs


class CpuRuntime:
    """``run_comparison`` runtime with native central and worker replay."""

    def __init__(self, policy, limit, case=driver.TWO_REQUEST):
        with tempfile.TemporaryDirectory() as tmp:
            _, (_, cache_config, _, sched_config, _) = native_configs(
                policy, limit, tmp, case)
        # As _init_cache with the case's chosen pool.
        cache_config.num_gpu_blocks = driver.case_inputs(case)["num_gpu_blocks"]
        self.scheduler = SchedulerRegistry.get(sched_config.type, sched_config,
                                               cache_config)
        self.engine_seqs = lpt._EngineSequenceManager()
        self.seq_map = self.engine_seqs.seq_map
        self.worker_seqs = WorkerSequenceManager(cache_config, sched_config)
        self.worker_ops = []
        driver.instrument_block_manager(self.worker_seqs.block_manager,
                                        self.worker_ops)
        self.lp = policy != "adapted_vllm"
        self.steps = 0
        # As the engine's request counter: finished requests leave seq_map,
        # so its length cannot name the next request.
        self.next_seq_id = 0

    def sample(self, seq_id):
        # Synthetic and distinct per request and step, so a misassociated
        # sample is detectable.
        return 100 * (self.steps + 1) + seq_id

    def submit(self, prompt_ids, sampling_params):
        # Same construction, arrival clock, and LP check as add_request.
        seq = Sequence(self.next_seq_id, None, prompt_ids, driver.BLOCK_SIZE,
                       2, time.perf_counter(), sampling_params)
        self.next_seq_id += 1
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
        samples = [SamplerOutput(m.seq_id, self.sample(m.seq_id))
                   for m in outputs.scheduled_seq_metadata_list]
        self.steps += 1
        self.worker_seqs.on_step_completed(outputs, samples)
        self.engine_seqs.on_step_completed(outputs, samples)
        self.scheduler.on_step_completed()
        return (self.engine_seqs.generate_request_outputs(ignored, metadata),
                [[s.seq_id, s.output_token] for s in samples])

    def worker_state(self):
        return driver.adapted.worker_validation_state(self.worker_seqs,
                                                      self.worker_ops)


def _run(policy, limit, case=driver.TWO_REQUEST, runtime=None):
    runtime, trace = runtime or CpuRuntime(policy, limit, case), {}
    outcome = driver.run_comparison(policy, limit, runtime, trace, case)
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
        self.assertIn("--policy {adapted_vllm,lp_uniform,lp_rank_based,"
                      "lp_coordinated,lp_reciprocal_square}", result.stdout)
        self.assertIn("--resident-limit {2,1,4,8}", result.stdout)
        self.assertIn("--case {two_request,memory_pressure,"
                      "queue48_burst_pool12}", result.stdout)
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
                    self.assertEqual(
                        acc["A"]["generated_tokens_moved_to_context"]
                        + acc["B"]["generated_tokens_moved_to_context"], 0)
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
                               lambda policy, snap, r, c: driver.UNIFORM):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "decision 0: coefficients of"):
                _run("lp_rank_based", 2)


class CacheCapacityTest(unittest.TestCase):
    def _init(self, profiled, chosen=driver.NUM_GPU_BLOCKS):
        selection = driver.CacheCapacitySelection(chosen=chosen)
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

    def test_memory_pressure_pool_of_four(self):
        chosen = driver.case_inputs(PRESSURE)["num_gpu_blocks"]
        engine, selection, calls = self._init(4, chosen)
        self.assertEqual(calls, [driver.PROFILE_METHOD, driver.INIT_CACHE_METHOD])
        self.assertEqual(engine.cache_config.num_gpu_blocks, 4)
        self.assertEqual(selection.record()["initializations"][0]
                         ["num_gpu_blocks"], 4)
        with self.assertRaisesRegex(driver.CheckFailure, "below the chosen 4"):
            self._init(3, chosen)


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

    def test_memory_pressure_command_line(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                   PYTHONPATH=str(REPO_ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            def run(*extra, out="out"):
                cmd = [sys.executable, "-B", "scripts/compare_scheduler_gpu.py",
                       "--policy", "lp_rank_based", "--model-path",
                       str(Path(tmp) / "absent"), "--output-dir",
                       str(Path(tmp) / out), *extra]
                return subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                      capture_output=True, text=True)
            # The limit defaults to 4 for this case; other limits are rejected,
            # and the original case still requires 2 or 1.
            result = run("--case", "memory_pressure")
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((Path(tmp) / "out/summary.json").read_text())
            self.assertEqual(
                {k: summary["run"][k] for k in (
                    "case", "resident_limit", "num_gpu_blocks", "max_model_len",
                    "token_budget", "b_max", "c_max", "s_max", "max_tokens",
                    "max_decisions")},
                dict(case="memory_pressure", resident_limit=4, num_gpu_blocks=4,
                     max_model_len=64, token_budget=64, b_max=64, c_max=64,
                     s_max=4, max_tokens=3, max_decisions=16))
            self.assertEqual(summary["workload_outcome"], "failure")
            for extra, out in ((("--case", "memory_pressure",
                                 "--resident-limit", "2"), "bad1"),
                               ((), "bad2"), (("--resident-limit", "4"), "bad3")):
                bad = run(*extra, out=out)
                self.assertEqual(bad.returncode, 2, (extra, bad.stderr))
                self.assertIn("--resident-limit for case", bad.stderr)


def _pair(trace, key):
    acc = trace["final"]["accounting"]
    return tuple(acc[n][key] for n in "ABCD")


def _emitted(d):
    return d["schedules"][0]["outputs"]


class MemoryPressureConfigurationTest(unittest.TestCase):

    def test_three_native_configurations(self):
        inputs = driver.case_inputs(PRESSURE)
        self.assertEqual({n: (p[0], len(p)) for n, p in inputs["prompts"].items()},
                         dict(A=(1000, 16), B=(2000, 16), C=(3000, 16),
                              D=(4000, 16)))
        flat = [t for p in inputs["prompts"].values() for t in p]
        self.assertEqual(len(flat), len(set(flat)))
        for policy in EARLIER_POLICIES:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                kwargs, (_, cache, parallel, sched, _) = native_configs(
                    policy, 4, tmp, PRESSURE)
                self.assertEqual((kwargs["load_format"], kwargs["dtype"],
                                  kwargs["max_model_len"],
                                  kwargs["attention_backend"]),
                                 ("dummy", "float16", 64, "flash_attention"))
                self.assertEqual((parallel.tensor_parallel_size,
                                  parallel.pipeline_parallel_size,
                                  cache.block_size, cache.gpu_memory_utilization),
                                 (1, 1, 16, 0.5))
                self.assertEqual((sched.max_num_seqs, sched.max_model_len,
                                  sched.num_pipeline_stages,
                                  sched.max_num_batched_tokens), (4, 64, 1, 64))
                params = driver.run_parameters(policy, 4, PRESSURE)
                self.assertEqual((params["native_watermark_blocks"],
                                  params["max_decisions"]), (0, 16))
                if policy == "adapted_vllm":
                    self.assertIs(type(sched), AdaptedVLLMSchedulerConfig)
                else:
                    self.assertIs(type(sched), LPSchedulerConfig)
                    self.assertEqual(
                        (sched.b_max, sched.c_max, sched.s_max,
                         sched.decode_utility, sched.prefill_token_utility,
                         sched.preemption_penalty),
                        (64, 64, 4, 1.0, 1.0, 1.0))
                    self.assertEqual(sched.numerical_policy,
                                     lrs.NumericalPolicy(
                                         **driver.NUMERICAL_POLICY_FIELDS))
        # The original case's LP caps and limits are unchanged.
        self.assertEqual(driver.run_parameters("lp_uniform", 1)["s_max"], 1)
        with self.assertRaises(driver.CheckFailure):
            driver.run_parameters("lp_uniform", 4)
        with self.assertRaises(driver.CheckFailure):
            driver.run_parameters("lp_uniform", 2, PRESSURE)


class MemoryPressureReplayTest(unittest.TestCase):
    """The three policies through real schedulers and native central and
    worker replay. Every decision passed the driver's checks if ``_run``
    returned."""

    @classmethod
    def setUpClass(cls):
        cls.runs = {p: _run(p, 4, PRESSURE) for p in EARLIER_POLICIES}

    def test_reference_follows_its_specification(self):
        # Matches the earlier reference preemption case (same workload).
        runtime, trace, outcome = self.runs["adapted_vllm"]
        self.assertEqual(outcome, driver.DRAINED)
        d = trace["decisions"]
        self.assertEqual([_emitted(x)["preempted_seq_ids"] for x in d],
                         [[], [], [3, 2], [], [], [], [], []])
        self.assertEqual([_emitted(x)["scheduled"] for x in d][:5], [
            [[i, 16] for i in range(4)], [[i, 0] for i in range(4)],
            [[0, 0], [1, 0]], [[0, 0], [1, 0]], [[2, 17], [3, 17]]])
        # Zero-demand decodes with zero free blocks: four appends, no block.
        self.assertEqual(d[1]["before"]["free_blocks"], 0)
        self.assertEqual([op[:2] + [len(op[2]), len(op[3])]
                          for op in d[1]["central_ops"]],
                         [["append", i, 1, 1] for i in range(4)])
        self.assertTrue(d[1]["victim_need"]["preemption_free_alternative"])
        # Actual one-block demand: tail victims 3 then 2, then growth 1 -> 2.
        need = d[2]["victim_need"]
        self.assertEqual((need["reasons"], need["decode_demands"],
                          need["preemption_free_alternative"]),
                         (["decode_block_demand"],
                          {str(i): 1 for i in range(4)}, False))
        self.assertEqual([op[:2] + [len(op[2]), len(op[3])]
                          for op in d[2]["central_ops"]],
                         [["free", 3, 1, 0], ["free", 2, 1, 0],
                          ["append", 0, 1, 2], ["append", 1, 1, 2]])
        self.assertEqual(d[2]["proposals"][0]["waiting_stop_reason"],
                         "skipped_after_preemption")
        # Recomputation after generated output: prompt 17, counter reset,
        # victims front-inserted in reverse preemption order.
        after = d[2]["after"]
        self.assertEqual(after["waiting"], [2, 3])
        for sid in (2, 3):
            seq = after["seqs"][str(sid)]
            self.assertEqual((seq["prompt_len"], seq["output_token_ids"],
                              seq["prompt_tokens_processed"], seq["logical_blocks"]),
                             (17, [], 0, 2))
            self.assertEqual(seq["prompt_token_ids"][16], 200 + sid)
        self.assertEqual(_pair(trace, "generated_token_events"), (3, 3, 4, 4))
        self.assertEqual(_pair(trace, "overgeneration"), (0, 0, 1, 1))
        self.assertEqual(_pair(trace, "generated_tokens_moved_to_context"),
                         (0, 0, 1, 1))
        self.assertEqual(_pair(trace, "prefill_tokens_replayed"), (0, 0, 16, 16))
        self.assertEqual(_pair(trace, "prefill_tokens_expanded_context"),
                         (0, 0, 1, 1))
        self.assertEqual(trace["idle"]["after"]["free_blocks"], 4)

    def test_lp_decode_prerequisite_at_zero_free_blocks(self):
        """Same state as the reference's decision 1: four residents with
        complete prompts, decode demand 0, no free block. No LP plan without
        a victim is useful, so exactly one victim funds H = 1 and the other
        three decode."""
        ref_state = self.runs["adapted_vllm"][1]["decisions"][1]["before"]
        for policy in ("lp_uniform", "lp_rank_based"):
            with self.subTest(policy=policy):
                d = self.runs[policy][1]["decisions"]
                self.assertEqual(
                    {k: v for k, v in d[1]["before"].items()
                     if k in ("running", "waiting", "free_blocks")},
                    {k: v for k, v in ref_state.items()
                     if k in ("running", "waiting", "free_blocks")})
                need = d[1]["victim_need"]
                self.assertEqual(
                    (need["reasons"], need["decode_demands"],
                     need["victim_blocks_needed_physical"],
                     need["victim_blocks_needed_with_lp_prerequisite"],
                     need["non_preemptive_actions"]),
                    (["lp_decode_prerequisite"], {str(i): 0 for i in range(4)},
                     0, 1, []))
                victims = _emitted(d[1])["preempted_seq_ids"]
                self.assertEqual(len(victims), 1)
                self.assertEqual(sorted(s for s, _ in _emitted(d[1])["scheduled"]),
                                 sorted(set(range(4)) - set(victims)))
                plan = d[1]["lp_events"][1]["result"]["plan"]
                self.assertEqual(plan["free_blocks_before_decodes"], 1)
                # The relaxation funds H fractionally; extraction rounds it.
                comparison = d[1]["lp_comparison"]
                self.assertTrue(comparison["extraction_changed_relaxed"])
                self.assertGreater(comparison["relaxed_objective"],
                                   comparison["integer_objective"])
        # Rank-based: penalty and decode reward are both 1/(1+s), so dropping
        # the tail (s = 3) is the unique integer optimum: 1 + 1/2 + 1/3 - 1/4.
        d1 = self.runs["lp_rank_based"][1]["decisions"][1]
        self.assertEqual(_emitted(d1)["preempted_seq_ids"], [3])
        self.assertAlmostEqual(d1["lp_comparison"]["integer_objective"],
                               1 + 1 / 2 + 1 / 3 - 1 / 4)
        # Uniform: every single-victim plan ties at 3 - 1; the victim is the
        # solver's vertex and is not prescribed.
        d1 = self.runs["lp_uniform"][1]["decisions"][1]
        self.assertAlmostEqual(d1["lp_comparison"]["integer_objective"], 2.0)

    def test_lp_runs_are_consistent_where_observed(self):
        for policy in ("lp_uniform", "lp_rank_based"):
            with self.subTest(policy=policy):
                runtime, trace, outcome = self.runs[policy]
                self.assertIn(outcome, (driver.DRAINED, driver.BOUNDED_UNFINISHED))
                acc = trace["final"]["accounting"]
                for name, a in acc.items():
                    self.assertEqual(a["prefill_tokens_executed"],
                                     a["prefill_tokens_first_time"]
                                     + a["prefill_tokens_replayed"]
                                     + a["prefill_tokens_expanded_context"])
                    self.assertLessEqual(a["prefill_tokens_first_time"], 16)
                    self.assertLessEqual(a["prefill_tokens_expanded_context"],
                                         a["generated_tokens_moved_to_context"])
                    if a["completion"] is None:
                        self.assertIsNone(a["overgeneration"])
                for d in trace["decisions"]:
                    victims = _emitted(d)["preempted_seq_ids"]
                    self.assertEqual(bool(d["victim_need"]["reasons"]),
                                     bool(victims))
                    comparison = d["lp_comparison"]
                    if comparison["fractional_request_count"]:
                        self.assertTrue(comparison["extraction_changed_relaxed"])
                if outcome == driver.BOUNDED_UNFINISHED:
                    final = trace["final"]
                    self.assertEqual(len(trace["decisions"]), 16)
                    self.assertIsNone(trace["idle"])
                    self.assertTrue(runtime.scheduler.has_unfinished_seqs())
                    self.assertEqual(
                        final["residual_allocations"],
                        {"ABCD"[int(k)]: len(v)
                         for k, v in final["central"]["block_tables"].items()})
                    self.assertEqual(sorted(final["unfinished"]),
                                     sorted("ABCD"[int(k)]
                                            for k in final["central"]["seqs"]))
                # Preemption after generated output and expanded readmission
                # are reached in these traces.
                self.assertGreater(sum(a["preemptions_after_output"]
                                       for a in acc.values()), 0)
                self.assertGreater(sum(a["prefill_tokens_expanded_context"]
                                       for a in acc.values()), 0)

    def test_multiple_actions_keep_order_and_sample_association(self):
        for policy, (_, trace, _) in self.runs.items():
            multi = [d for d in trace["decisions"]
                     if len(_emitted(d)["scheduled"]) > 1]
            self.assertTrue(multi, policy)
            for d in multi:
                # Sampler outputs follow the emitted order, one per entry, with
                # the synthetic per-request token of that step.
                step = d["index"] + 1
                self.assertEqual(d["sampler_outputs"],
                                 [[s, 100 * step + s]
                                  for s, _ in _emitted(d)["scheduled"]])


class MemoryPressureDetectionTest(unittest.TestCase):

    def test_misassociated_sample_is_detected(self):
        runtime = CpuRuntime("adapted_vllm", 4, PRESSURE)
        native = runtime.step

        def swapped():
            outputs, samples = native()
            if samples and len(samples) > 1 and runtime.steps == 2:
                # Report the first two samples under each other's IDs.
                # The native state keeps the true tokens.
                samples[0][1], samples[1][1] = samples[1][1], samples[0][1]
            return outputs, samples
        runtime.step = swapped
        with self.assertRaisesRegex(driver.CheckFailure,
                                    "decision 1: request 0 progress"):
            _run("adapted_vllm", 4, PRESSURE, runtime)

    def test_mid_run_failure_keeps_the_partial_trace(self):
        native = lrs.solve_and_extract
        calls = []

        def fail_third(problem):
            calls.append(problem.problem_id)
            if len(calls) == 3:
                return lrs.Failure(problem.problem_id, lrs.STAGE_SOLVER,
                                   lrs.OUTCOME_NUMERICAL, "injected", "test")
            return native(problem)
        trace = {}
        runtime = CpuRuntime("lp_uniform", 4, PRESSURE)
        with mock.patch.object(lrs, "solve_and_extract", fail_third):
            with self.assertRaises(LPSchedulingError):
                driver.run_comparison("lp_uniform", 4, runtime, trace, PRESSURE)
        self.assertEqual(len(trace["decisions"]), 3)
        self.assertIn("victim_need", trace["decisions"][1])
        self.assertEqual(trace["decisions"][2]["central_ops"], [])
        self.assertIsNone(trace["final"])


import exact_integer_plan as eip  # noqa: E402

COORDINATED = ("lp_coordinated", driver.INTEGER_POLICY)


class CoordinatedPolicyTest(unittest.TestCase):
    """The coordinated LP and the CPU-only integer method through the
    driver's checks and native CPU replay."""

    def test_configuration_and_hook_selection(self):
        for case, limit, caps in ((driver.TWO_REQUEST, 1, (8, 8)),
                                  (PRESSURE, 4, (64, 64)),
                                  ("mixed_staggered_tight", 4, (32, 32))):
            for policy in COORDINATED:
                with self.subTest(case=case, policy=policy):
                    runtime = CpuRuntime(policy, limit, case)
                    scheduler = runtime.scheduler
                    driver.select_coefficient_policy(scheduler, policy, case)
                    bound = scheduler._coefficient_policy()
                    self.assertIs(bound.func, lsm.coordinated_rank_utility)
                    self.assertEqual(bound.keywords, dict(
                        rank_base=0.25, c_max=caps[0], b_max=caps[1]))
                    self.assertEqual(
                        "_schedule" in vars(scheduler),
                        policy == driver.INTEGER_POLICY)
                    params = driver.run_parameters(policy, limit, case)
                    self.assertEqual(params["coefficient_policy"][
                        "coordinated_rank_utility"]["rank_base"], 0.25)
        self.assertNotIn("_schedule", vars(CpuRuntime("lp_coordinated", 1)
                                           .scheduler))

    def test_two_request_limit_one_drains_without_preemption(self):
        rows = {}
        for policy in COORDINATED:
            _, trace, outcome = _run(policy, 1)
            rows[policy] = [_emitted(d) for d in trace["decisions"]]
            self.assertEqual(_row(trace, outcome),
                             (6, (0, 3), (1, 4), (2, 5), 0, "drained"))
            calls = driver.decision_calls(policy)
            for d in trace["decisions"]:
                self.assertEqual([e["call"] for e in d["lp_events"]], calls)
            if policy == driver.INTEGER_POLICY:
                d = trace["decisions"][0]
                self.assertEqual(d["lp_events"][1]["result_type"],
                                 "IntegerOptimum")
                self.assertIsNone(d["lp_comparison"])
                self.assertEqual(d["integer_solution"]["objective"], 1.0)
        # Integer optimization does not change this trajectory.
        self.assertEqual(rows["lp_coordinated"], rows[driver.INTEGER_POLICY])

    def test_coordinated_coefficients_are_checked(self):
        with mock.patch.object(driver, "expected_coefficients",
                               lambda policy, snap, r, c: driver.UNIFORM):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "decision 0: coefficients of"):
                _run("lp_coordinated", 1)

    def test_zero_free_block_prerequisite_selects_the_tail(self):
        for policy in COORDINATED:
            with self.subTest(policy=policy):
                _, trace, outcome = _run(policy, 4, PRESSURE)
                self.assertIn(outcome, (driver.DRAINED,
                                        driver.BOUNDED_UNFINISHED))
                d1 = trace["decisions"][1]
                self.assertEqual(d1["before"]["free_blocks"], 0)
                self.assertEqual(d1["victim_need"]["reasons"],
                                 ["lp_decode_prerequisite"])
                self.assertEqual(_emitted(d1)["preempted_seq_ids"], [3])
                self.assertEqual(_emitted(d1)["scheduled"],
                                 [[0, 0], [1, 0], [2, 0]])

    def test_mixed_pressure_uses_one_block_decode_demand(self):
        # The batch diagnostic: native construction, one mapping per method
        # on separately built states (checked equal), the LP and the direct
        # integer plan each executed by the native executor (which validates
        # the complete plan before mutation), the reference proposal, and
        # bounded enumeration.
        import compare_coefficient_policies_cpu as batch
        record = batch.run_diagnostic(
            "mixed_pressure", batch.DIAGNOSTIC_STATES["mixed_pressure"])
        mapped = record["mapped_requests"]
        self.assertEqual(
            {k: mapped["1"][k] for k in ("output_len", "logical_blocks",
                                          "physical_blocks", "decode_demand")},
            dict(output_len=1, logical_blocks=2, physical_blocks=1,
                 decode_demand=1))
        self.assertEqual(mapped["2"]["decode_demand"], 0)
        self.assertEqual((mapped["0"]["prompt_tokens_remaining"],
                          mapped["0"]["prefill_fixed_charge"]), (8, 0))
        self.assertEqual(record["free_blocks"], 0)

        want = {"0": "prefill:7", "1": "decode", "2": "preempt"}
        comparison = record["lp_vs_integer"]
        self.assertEqual((comparison["lp_actions"],
                          comparison["integer_actions"]), (want, want))
        for key in ("lp_plan_objective", "integer_objective",
                    "lp_relaxed_objective"):
            self.assertAlmostEqual(comparison[key], 7 / 8 + 1 / 4 - 1 / 16,
                                   msg=key)
        self.assertEqual(record["enumeration"]["best"], 1.0625)
        for method, plan in (("lp", record["lp"]["plan"]),
                             ("integer", record["integer"]["plan"])):
            with self.subTest(method=method):
                self.assertEqual((plan["decode_demand"],
                                  plan["free_blocks_before_decodes"]), (1, 1))
                self.assertEqual(record["executed"][method]["scheduled"],
                                 [[0, 7], [1, 0]])
                self.assertEqual(
                    record["executed"][method]["preempted_seq_ids"], [2])
                blocks = record["block_counts"][method]
                self.assertEqual(blocks["before"], dict(
                    free=0, physical={"0": 1, "1": 1, "2": 1}))
                # B grows by one block; C's block is released and reused.
                self.assertEqual(blocks["after"], dict(
                    free=0, physical={"0": 1, "1": 2}))
        reference = record["reference_proposal"]
        self.assertEqual(([(a["seq_id"], a["kind"], a["num_tokens"])
                           for a in reference["actions"]],
                          reference["victims"]),
                         ([(0, driver.PREFILL, 8)], []))
        self.assertEqual(record["reference_vs_integer"]["classification"],
                         "strict_objective_preference")

    def test_integer_failure_stops_the_rollout_without_fallback(self):
        native = eip.solve_integer_plan
        calls = []

        def fail_second(problem):
            calls.append(problem.problem_id)
            if len(calls) == 2:
                return eip.IntegerSolveFailure(
                    problem.problem_id, eip.STAGE_INTEGER_SOLVER,
                    "solver_limit", "injected", 1, "time limit",
                    incumbent_objective=0.5)
            return native(problem)
        trace = {}
        runtime = CpuRuntime(driver.INTEGER_POLICY, 1)
        with mock.patch.object(eip, "solve_integer_plan", fail_second), \
                mock.patch.object(lrs, "solve_and_extract") as lp:
            with self.assertRaises(LPSchedulingError) as caught:
                driver.run_comparison(driver.INTEGER_POLICY, 1, runtime, trace)
        lp.assert_not_called()
        self.assertEqual(caught.exception.category, "solver_limit")
        record = trace["decisions"][1]
        self.assertEqual([e["call"] for e in record["lp_events"]],
                         driver.INTEGER_CALLS[:2])
        self.assertEqual(record["lp_events"][1]["result"]
                         ["incumbent_objective"], 0.5)
        self.assertEqual(record["central_ops"], [])
        self.assertEqual(record["central_at_failure"]["running"], [0])
        self.assertIsNone(trace["final"])

    def test_cli_accepts_no_integer_method_or_cpu_only_case(self):
        for argv in (["--policy", driver.INTEGER_POLICY],
                     ["--policy", "lp_coordinated", "--case",
                      "mixed_burst_ample"]):
            with self.subTest(argv=argv), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    driver.parse_args(argv + ["--model-path", "m",
                                              "--output-dir", "o"])
                self.assertEqual(caught.exception.code, 2)


class VirtualArrivalTest(unittest.TestCase):

    def test_representative_inputs(self):
        c = driver.case_inputs("mixed_staggered_tight")
        self.assertEqual([len(p) for p in c["prompts"].values()],
                         [5, 8, 16, 17, 31, 32])
        self.assertEqual(list(c["max_tokens"].values()), [2, 3, 2, 4, 3, 4])
        self.assertEqual(list(c["arrivals"].values()), [0, 2, 4, 6, 8, 10])
        self.assertEqual((c["num_gpu_blocks"], c["max_model_len"],
                          c["resident_limits"], c["token_budget"], c["b_max"],
                          c["c_max"], c["s_max"], c["max_decisions"]),
                         (6, 64, (4,), 32, 32, 32, 4, 64))
        equal = driver.case_inputs("equal_burst_ample")
        self.assertEqual(([len(p) for p in equal["prompts"].values()],
                          equal["max_tokens"], set(equal["arrivals"].values()),
                          equal["num_gpu_blocks"]), ([16] * 6, 3, {0}, 16))
        flat = [t for p in c["prompts"].values() for t in p]
        self.assertEqual(len(flat), len(set(flat)))
        # Existing cases have every request at virtual time 0.
        self.assertEqual(set(driver.case_inputs(PRESSURE)["arrivals"]
                             .values()), {0})

    def test_staggered_submission_and_per_request_limits(self):
        case = "mixed_staggered_tight"
        c = driver.case_inputs(case)
        _, trace, outcome = _run("adapted_vllm", 4, case)
        acc = trace["final"]["accounting"]
        for name, when in c["arrivals"].items():
            submitted = trace["submitted_at_virtual_time"][name]
            self.assertGreaterEqual(submitted, when)
            sid = list(c["prompts"]).index(name)
            # Never mapped or proposed before submission.
            for d in trace["decisions"]:
                present = sid in d["before"]["waiting"] + d["before"]["running"]
                self.assertEqual(present and d["virtual_time"] < submitted,
                                 False)
            if acc[name]["completion"] is not None:
                self.assertEqual(acc[name]["native_output_count"],
                                 c["max_tokens"][name])
        self.assertEqual([d["virtual_time"] for d in trace["decisions"]],
                         list(range(len(trace["decisions"]))))
        self.assertIn(outcome, (driver.DRAINED, driver.BOUNDED_UNFINISHED))

    def test_idle_interval_jumps_without_a_scheduler_call(self):
        inputs = json.loads(json.dumps(
            driver.REPRESENTATIVE_INPUTS["mixed_burst_ample"]))
        inputs["arrivals"] = dict(zip("ABCDEF", [0, 40, 40, 40, 40, 40]))
        with mock.patch.dict(driver.REPRESENTATIVE_INPUTS,
                             {"mixed_burst_ample": inputs}):
            runtime, trace, outcome = _run("lp_coordinated", 4,
                                           "mixed_burst_ample")
        self.assertEqual(outcome, driver.DRAINED)
        # A (5 tokens, 2 outputs) drains after 3 decisions; the clock then
        # jumps to 40 and the next decision sees B..F only.
        self.assertEqual(trace["idle_intervals"], [[3, 40]])
        times = [d["virtual_time"] for d in trace["decisions"]]
        self.assertEqual(times[:4], [0, 1, 2, 40])
        self.assertEqual(trace["decisions"][3]["before"]["waiting"],
                         [1, 2, 3, 4, 5])
        self.assertEqual(sorted(trace["submitted_at_virtual_time"].values()),
                         [0, 40, 40, 40, 40, 40])
        # Sequence IDs stay unique after A leaves seq_map.
        self.assertEqual(runtime.next_seq_id, 6)


RECIPROCAL = (driver.RECIPROCAL_SQUARE_POLICY,
              driver.RECIPROCAL_SQUARE_INTEGER_POLICY)
ANCHOR = "queue6_burst"


def _queue_run(method, case=ANCHOR):
    """One queue-growth rollout through the CPU batch, in its retained JSON
    form, with its offline comparisons."""
    import compare_coefficient_policies_cpu as batch
    with tempfile.TemporaryDirectory() as tmp:
        summary, trace = batch.run_queue_rollout(
            sys.modules[__name__], method, case, Path(tmp))
    return summary, trace


class QueueGrowthTest(unittest.TestCase):
    """The reciprocal-square policy, its integer-driven CPU method, and the
    per-decision comparison records of the queue-growth batch."""

    def test_queue_growth_inputs(self):
        sizes = dict(queue6_burst=(6, 128, False), queue12_burst=(12, 128, False),
                     queue24_burst=(24, 128, False),
                     queue48_burst=(48, 128, False),
                     queue48_staggered=(48, 128, True),
                     queue48_burst_pool12=(48, 12, False))
        self.assertEqual(driver.QUEUE_GROWTH_CASES, tuple(sizes))
        for case, (count, pool, staggered) in sizes.items():
            with self.subTest(case=case):
                c = driver.case_inputs(case)
                prompts = list(c["prompts"].values())
                self.assertEqual([len(p) for p in prompts],
                                 [[8, 16, 17, 32, 64, 128][k % 6]
                                  for k in range(count)])
                self.assertEqual([p[0] for p in prompts],
                                 [1000 + 256 * k for k in range(count)])
                self.assertEqual(list(c["max_tokens"].values()),
                                 [[2, 4, 8][k % 3] for k in range(count)])
                self.assertEqual(list(c["arrivals"].values()),
                                 [2 * k if staggered else 0
                                  for k in range(count)])
                self.assertEqual(
                    (c["num_gpu_blocks"], c["max_model_len"],
                     c["resident_limits"], c["token_budget"], c["b_max"],
                     c["c_max"], c["s_max"], c["max_decisions"]),
                    (pool, 192 if case == driver.CONSTRAINED_QUEUE_CASE
                     else 256, (8,), 128, 128, 128, 8, 8 * count))
                flat = [t for p in prompts for t in p]
                self.assertEqual(len(flat), len(set(flat)))
                self.assertLess(max(flat), 32000)
        # Only the constrained 48-request case is a GPU (command-line) case.
        self.assertEqual([c for c in driver.GPU_CASES
                          if c in driver.QUEUE_GROWTH_CASES],
                         [driver.CONSTRAINED_QUEUE_CASE])

    def test_only_the_constrained_context_changed(self):
        # Every other input of the constrained case is the 256-token
        # construction's; every other queue case keeps 256.
        before = driver._queue_growth(48, False, 12)
        after = driver.QUEUE_GROWTH_INPUTS[driver.CONSTRAINED_QUEUE_CASE]
        self.assertEqual({k for k in before if before[k] != after[k]},
                         {"max_model_len"})
        self.assertEqual((before["max_model_len"], after["max_model_len"]),
                         (256, 192))
        self.assertEqual({c: driver.case_inputs(c)["max_model_len"]
                          for c in driver.QUEUE_GROWTH_CASES
                          if c != driver.CONSTRAINED_QUEUE_CASE},
                         dict.fromkeys(("queue6_burst", "queue12_burst",
                                        "queue24_burst", "queue48_burst",
                                        "queue48_staggered"), 256))
        # Every request still fits well below the new limit.
        longest = max(len(p) + driver.max_tokens(after, n)
                      for n, p in after["prompts"].items())
        self.assertEqual(longest, 136)

    def test_native_capacity_preflight_agrees_with_native_initialization(self):
        from sarathi.engine.base_llm_engine import BaseLLMEngine

        def native_accepts(pool, max_model_len):
            stub = SimpleNamespace(
                cache_config=SimpleNamespace(block_size=driver.BLOCK_SIZE,
                                             gpu_memory_utilization=0.5,
                                             num_gpu_blocks=None),
                model_config=SimpleNamespace(max_model_len=max_model_len),
                _run_workers=lambda method, **kw: [pool])
            try:
                BaseLLMEngine._init_cache(stub)
            except ValueError:
                return False
            return True
        corrected = driver.case_inputs(driver.CONSTRAINED_QUEUE_CASE)
        self.assertEqual(driver.check_native_capacity(corrected), 12)
        self.assertTrue(native_accepts(12, 192))
        old = dict(corrected, max_model_len=256)
        with self.assertRaisesRegex(driver.CheckFailure,
                                    "below the 16 blocks native cache "
                                    "initialization requires"):
            driver.check_native_capacity(old)
        self.assertFalse(native_accepts(12, 256))
        for case in driver.GPU_CASES:
            with self.subTest(case=case):
                c = driver.case_inputs(case)
                driver.check_native_capacity(c)
                self.assertTrue(native_accepts(c["num_gpu_blocks"],
                                               c["max_model_len"]))

    def test_incompatible_pool_fails_before_engine_construction(self):
        old = dict(driver.QUEUE_GROWTH_INPUTS[driver.CONSTRAINED_QUEUE_CASE],
                   max_model_len=256)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(driver.QUEUE_GROWTH_INPUTS,
                                {driver.CONSTRAINED_QUEUE_CASE: old}), \
                mock.patch.object(driver, "engine_args") as args, \
                mock.patch.object(driver, "build_engine_class") as build, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = driver.main(["--case", driver.CONSTRAINED_QUEUE_CASE,
                                "--policy", driver.RECIPROCAL_SQUARE_POLICY,
                                "--model-path", str(Path(tmp) / "absent"),
                                "--output-dir", str(Path(tmp) / "out")])
            summary = json.loads((Path(tmp) / "out/summary.json").read_text())
        self.assertEqual(code, 1)
        args.assert_not_called()
        build.assert_not_called()
        self.assertEqual((summary["validation_passed"],
                          summary["workload_outcome"]), (False, "failure"))
        self.assertIn("native cache initialization requires",
                      summary["failure"]["message"])

    def test_queue_growth_case_selection(self):
        import compare_coefficient_policies_cpu as batch
        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "out"
            self.assertEqual(batch.main(["--batch", "queue_growth", "--cases",
                                         ANCHOR, "--output-dir", str(out)]), 0)
            record = json.loads((out / "run_record.json").read_text())
            ran = sorted(p.name for p in (out / "rollouts").iterdir())
            methods = sorted(p.name for p in (out / "rollouts" / ANCHOR)
                             .iterdir())
        self.assertEqual(record["selected_cases"], [ANCHOR])
        self.assertEqual(record["runs"], [[ANCHOR, m]
                                          for m in batch.QUEUE_ANCHOR_METHODS])
        self.assertEqual((ran, methods),
                         ([ANCHOR], sorted(batch.QUEUE_ANCHOR_METHODS)))
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (["--cases", ANCHOR],
                         ["--batch", "queue_growth", "--cases", "two_request"]):
                with self.subTest(argv=argv), \
                        self.assertRaises(SystemExit) as caught:
                    batch.main(argv + ["--output-dir", "unused"])
                self.assertEqual(caught.exception.code, 2)

    def test_configuration_hook_and_native_watermark(self):
        for case, watermark in ((ANCHOR, 1), (driver.CONSTRAINED_QUEUE_CASE, 0)):
            for policy in RECIPROCAL:
                with self.subTest(case=case, policy=policy):
                    runtime = CpuRuntime(policy, 8, case)
                    scheduler = runtime.scheduler
                    self.assertEqual(scheduler.block_manager.watermark_blocks,
                                     watermark)
                    driver.select_coefficient_policy(scheduler, policy, case)
                    bound = scheduler._coefficient_policy()
                    self.assertIs(bound.func, lsm.reciprocal_square_rank_utility)
                    self.assertEqual(bound.keywords, dict(c_max=128, b_max=128))
                    self.assertEqual("_schedule" in vars(scheduler),
                                     policy == driver.RECIPROCAL_SQUARE_INTEGER_POLICY)
                    params = driver.run_parameters(policy, 8, case)
                    self.assertIn("reciprocal_square_rank_utility",
                                  params["coefficient_policy"])
                    self.assertEqual(params["native_watermark_blocks"],
                                     watermark)
        self.assertEqual(driver.decision_calls(
            driver.RECIPROCAL_SQUARE_INTEGER_POLICY), driver.INTEGER_CALLS)

    def test_reciprocal_square_coefficients_are_checked(self):
        # The policy-specific check rejects the geometric weights.
        native = driver.expected_coefficients

        def geometric(policy, snap, request, c):
            return native("lp_coordinated", snap, request, c)
        with mock.patch.object(driver, "expected_coefficients", geometric):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "decision 0: coefficients of"):
                _run(driver.RECIPROCAL_SQUARE_POLICY, 8, ANCHOR)

    def test_every_decision_has_a_comparison_record(self):
        import compare_coefficient_policies_cpu as batch
        native = eip.solve_integer_plan
        for method in ("adapted_vllm",) + RECIPROCAL:
            with self.subTest(method=method):
                summary, trace = _queue_run(method)
                self.assertTrue(summary["validation_passed"], summary["failure"])
                self.assertEqual(summary["workload_outcome"], driver.DRAINED)
                reached = len(trace["decisions"])
                calls = []
                with mock.patch.object(
                        eip, "solve_integer_plan",
                        lambda p: calls.append(p) or native(p)):
                    records, coverage = batch.decision_comparisons(trace, method)
                self.assertEqual([r["index"] for r in records],
                                 list(range(reached)))
                integer_driven = method in driver.INTEGER_POLICIES
                self.assertEqual(coverage, dict(
                    reached_decisions=reached, valid_problems=reached,
                    mapping_failures=0, missing_problems=0,
                    round_trip_failures=0,
                    attempted_solves=0 if integer_driven else reached,
                    reused_integer_results=reached if integer_driven else 0,
                    optimal=reached, time_limits=0, other_failures=0,
                    comparison_errors=0, records=reached, missing_records=0,
                    no_valid_problem=0, complete=True))
                # The integer-driven result is reused, never solved again.
                self.assertEqual(len(calls), 0 if integer_driven else reached)
                for r in records:
                    self.assertTrue(r["round_trip_equal"])
                    self.assertEqual(r["integer"]["status"], "optimal")
                    if method == "adapted_vllm":
                        self.assertIn(r["comparison"]["classification"], (
                            "identical_actions", "tied_optimum",
                            "strict_objective_preference",
                            "contract_difference"))
                    elif not integer_driven:
                        self.assertTrue(r["comparison"]["bounds_consistent"])
                if method == "adapted_vllm":
                    # One read-only diagnostic problem per decision, none for
                    # the final ordinary idle step, with reciprocal-square
                    # coefficients and the case's caps.
                    problems = trace["reference_diagnostic_problems"]
                    self.assertEqual(len(problems), reached)
                    first = problems[0]["result"]
                    self.assertEqual((first["b_max"], first["c_max"],
                                      first["s_max"],
                                      first["admission_watermark"]),
                                     (128, 128, 8, 1))
                    self.assertEqual(
                        [r["decode_utility"] + r["prefill_token_utility"]
                         * r["prefill_upper_bound"] for r in first["requests"]],
                        [1 / (1 + k) ** 2 for k in range(6)])
                    # The capture does not change the reference's decisions.
                    _, plain, _ = _run("adapted_vllm", 8, ANCHOR)
                    self.assertEqual(
                        [_emitted(d) for d in plain["decisions"]],
                        [d["schedules"][0]["outputs"]
                         for d in trace["decisions"]])

    def test_missing_or_invalid_problem_records_are_not_silent(self):
        import compare_coefficient_policies_cpu as batch
        _, trace = _queue_run("adapted_vllm")
        dropped = copy.deepcopy(trace)
        dropped["reference_diagnostic_problems"].pop()
        _, coverage = batch.decision_comparisons(dropped, "adapted_vllm")
        self.assertEqual((coverage["missing_problems"], coverage["complete"]),
                         (1, False))
        failed = copy.deepcopy(trace)
        failed["reference_diagnostic_problems"][0] = dict(
            result_type="MappingFailure", result=dict(reason="injected"))
        records, coverage = batch.decision_comparisons(failed, "adapted_vllm")
        self.assertEqual((records[0]["problem_status"],
                          coverage["mapping_failures"], coverage["complete"]),
                         ("mapping_failure", 1, True))
        _, lp_trace = _queue_run(driver.RECIPROCAL_SQUARE_POLICY)
        corrupt = copy.deepcopy(lp_trace)
        corrupt["decisions"][1]["lp_events"][0]["result"]["lp_problem"][
            "b_max"] = 0
        records, coverage = batch.decision_comparisons(
            corrupt, driver.RECIPROCAL_SQUARE_POLICY)
        self.assertIn("comparison_error", records[1])
        self.assertEqual((coverage["comparison_errors"], coverage["complete"]),
                         (1, False))

    def test_problem_records_round_trip_through_json(self):
        _, trace = _queue_run(driver.RECIPROCAL_SQUARE_POLICY)
        for d in trace["decisions"]:
            record = d["lp_events"][0]["result"]["lp_problem"]
            problem = eip.problem_from_record(json.loads(json.dumps(record)))
            canonical = lrs.validate_problem(problem)
            self.assertIsInstance(canonical, lrs.LPProblem)
            self.assertEqual(driver.to_jsonable(canonical), record)
            # The reloaded problem solves to the recorded LP result.
            result = lrs.solve_and_extract(canonical)
            self.assertEqual(driver.to_jsonable(result.plan),
                             d["lp_events"][1]["result"]["plan"])

    def test_non_optimal_offline_comparison_is_recorded(self):
        import compare_coefficient_policies_cpu as batch
        _, trace = _queue_run(driver.RECIPROCAL_SQUARE_POLICY)
        native, calls = eip.solve_integer_plan, []

        def limit_second(problem):
            calls.append(problem.problem_id)
            if len(calls) == 2:
                return eip.IntegerSolveFailure(
                    problem.problem_id, eip.STAGE_INTEGER_SOLVER,
                    "solver_limit", "injected", 1, "time limit reached",
                    incumbent_vector=(0.0,), incumbent_objective=0.5,
                    dual_bound=0.75, reported_gap=0.5, node_count=7)
            return native(problem)
        with mock.patch.object(eip, "solve_integer_plan", limit_second):
            records, coverage = batch.decision_comparisons(
                trace, driver.RECIPROCAL_SQUARE_POLICY)
        reached = len(trace["decisions"])
        self.assertEqual((coverage["attempted_solves"], coverage["optimal"],
                          coverage["time_limits"], coverage["complete"]),
                         (reached, reached - 1, 1, True))
        integer = records[1]["integer"]
        self.assertEqual({k: integer[k] for k in (
            "status", "category", "raw_status", "incumbent_objective",
            "dual_bound", "reported_gap", "node_count")}, dict(
            status="failure", category="solver_limit", raw_status=1,
            incumbent_objective=0.5, dual_bound=0.75, reported_gap=0.5,
            node_count=7))
        self.assertEqual(records[1]["comparison"]["attribution"],
                         "not_comparable")
        self.assertNotIn("integer_objective", records[1]["comparison"])
        self.assertEqual(records[2]["integer"]["status"], "optimal")

    def test_integer_rollout_stops_on_non_optimal_without_fallback(self):
        import compare_coefficient_policies_cpu as batch
        native, calls = eip.solve_integer_plan, []

        def limit_second(problem):
            calls.append(problem.problem_id)
            if len(calls) == 2:
                return eip.IntegerSolveFailure(
                    problem.problem_id, eip.STAGE_INTEGER_SOLVER,
                    "solver_limit", "injected", 1, "time limit",
                    incumbent_objective=0.5, dual_bound=0.75)
            return native(problem)
        method = driver.RECIPROCAL_SQUARE_INTEGER_POLICY
        with mock.patch.object(eip, "solve_integer_plan", limit_second), \
                mock.patch.object(lrs, "solve_and_extract") as lp:
            summary, trace = _queue_run(method)
        lp.assert_not_called()
        self.assertEqual((summary["validation_passed"],
                          summary["workload_outcome"],
                          summary["failure"]["category"]),
                         (False, driver.FAILURE, "solver_limit"))
        self.assertEqual(len(trace["decisions"]), 2)
        self.assertEqual(trace["decisions"][1]["central_ops"], [])
        records, coverage = batch.decision_comparisons(trace, method)
        self.assertEqual((coverage["reused_integer_results"],
                          coverage["time_limits"], coverage["attempted_solves"],
                          coverage["complete"]), (2, 1, 0, True))
        self.assertEqual(records[1]["integer"]["incumbent_objective"], 0.5)

    def test_command_lines(self):
        args = driver.parse_args(["--case", driver.CONSTRAINED_QUEUE_CASE,
                                  "--policy", driver.RECIPROCAL_SQUARE_POLICY,
                                  "--model-path", "m", "--output-dir", "o"])[1]
        self.assertEqual((args.case, args.policy, args.resident_limit),
                         (driver.CONSTRAINED_QUEUE_CASE,
                          driver.RECIPROCAL_SQUARE_POLICY, None))
        import compare_coefficient_policies_cpu as batch
        for parse, argv in (
                (driver.parse_args, ["--policy",
                                     driver.RECIPROCAL_SQUARE_INTEGER_POLICY]),
                (driver.parse_args, ["--policy", "adapted_vllm", "--case",
                                     ANCHOR]),
                (batch.parse_args, ["--batch", "unknown"])):
            with self.subTest(argv=argv), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse(argv + ["--model-path", "m", "--output-dir", "o"]
                          if parse is driver.parse_args
                          else argv + ["--output-dir", "o"])
                self.assertEqual(caught.exception.code, 2)
        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stderr(io.StringIO()):
            for argv in (["--batch", "retained_runs"],
                         ["--batch", "queue_growth", "--runs", "x"]):
                with self.assertRaises(SystemExit) as caught:
                    batch.main(argv + ["--output-dir", str(Path(tmp) / "o")])
                self.assertEqual(caught.exception.code, 2)


class CpuEntrypointTest(unittest.TestCase):
    def test_help_loads_no_engine_worker_ray_or_torch(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import compare_coefficient_policies_cpu as m\n"
            "try:\n    m.main(['--help'])\nexcept SystemExit as e:\n"
            "    code = e.code\n"
            "print('RESULT', code, sorted(n for n in ('torch', 'ray', "
            "'sarathi.engine.base_llm_engine') if n in sys.modules))")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=REPO_ROOT, env=env, capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-dir", result.stdout)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "RESULT 0 []")


if __name__ == "__main__":
    unittest.main()
