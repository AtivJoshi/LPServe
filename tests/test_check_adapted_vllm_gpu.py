"""CPU checks for the adapted vLLM reference GPU validation driver.

No engine, Ray worker, model, or GPU is used. The driver's case checks run
against the registered scheduler, central block manager, engine sequence
manager, and a worker sequence manager holding separate copies, replayed in
the engine's single-stage order at the planned block size, pools, budgets,
model lengths, and output limits. Sampled tokens are synthetic; the checks
do not prescribe token IDs. Cache sizing is exercised through native
``_init_cache`` with the worker dispatch replaced by a recording double.
"""

import copy
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

import check_adapted_vllm_gpu as driver  # noqa: E402
import test_lp_scheduler as lpt  # noqa: E402

from sarathi.config import (  # noqa: E402
    AdaptedVLLMSchedulerConfig,
    CacheConfig,
    SchedulerType,
)
from sarathi.core.datatypes.sequence import SamplerOutput, Sequence  # noqa: E402
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry  # noqa: E402
from sarathi.core.sequence_manager.worker_sequence_manager import (  # noqa: E402
    WorkerSequenceManager,
)
from sarathi.engine import arg_utils  # noqa: E402

setUpModule = lpt.setUpModule
tearDownModule = lpt.tearDownModule

SAMPLED_TOKEN = 7


class CpuRuntime:
    """``run_case`` runtime with native central and worker replay on CPU."""

    def __init__(self, name):
        case = driver.CASES[name]
        cache_config = CacheConfig(block_size=driver.BLOCK_SIZE,
                                   gpu_memory_utilization=0.5)
        cache_config.num_gpu_blocks = case["num_gpu_blocks"]
        config = AdaptedVLLMSchedulerConfig(
            case["max_num_seqs"], case["max_model_len"], 1,
            case["max_num_batched_tokens"])
        self.scheduler = SchedulerRegistry.get(
            SchedulerType.ADAPTED_VLLM, config, cache_config)
        self.engine_seqs = lpt._EngineSequenceManager()
        self.seq_map = self.engine_seqs.seq_map
        self.worker_seqs = WorkerSequenceManager(cache_config, config)
        self.worker_ops = []
        driver.instrument_block_manager(self.worker_seqs.block_manager,
                                        self.worker_ops)
        self.next_id = 0

    def prompt_ids(self, lengths):
        return [list(range(1000 * (n + 1), 1000 * (n + 1) + length))
                for n, length in enumerate(lengths)]

    def submit(self, prompt_ids, sampling_params):
        # Same construction and arrival clock as BaseLLMEngine.add_request.
        seq = Sequence(self.next_id, None, prompt_ids, driver.BLOCK_SIZE, 2,
                       time.perf_counter(), sampling_params)
        self.next_id += 1
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
        return driver.worker_validation_state(self.worker_seqs, self.worker_ops)


def _run(name):
    runtime, trace = CpuRuntime(name), {}
    driver.run_case(name, runtime, trace)
    return runtime, trace


class DriverImportAndHelpTest(unittest.TestCase):
    MODULES = ("torch", "ray", "sarathi.engine.base_llm_engine",
               "sarathi.worker.base_worker", "sarathi.engine.arg_utils")

    def _python(self, code):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        return subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO_ROOT,
                              env=env, capture_output=True, text=True)

    def test_import_and_help_load_no_engine_worker_ray_or_torch(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import check_adapted_vllm_gpu as d\n"
            "try:\n    d.main(['--help'])\nexcept SystemExit as e:\n"
            "    code = e.code\n"
            f"print('RESULT', code, sorted(m for m in {self.MODULES!r} "
            "if m in sys.modules))"
        )
        result = self._python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--case {mixed,preemption}", result.stdout)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "RESULT 0 []")


class PlannedCaseTest(unittest.TestCase):

    def test_case_parameters_match_the_planned_workloads(self):
        mixed, pre = driver.CASES["mixed"], driver.CASES["preemption"]
        self.assertEqual(
            (mixed["prompt_lens"], mixed["max_tokens"],
             mixed["max_num_batched_tokens"], mixed["max_num_seqs"],
             mixed["max_model_len"], mixed["num_gpu_blocks"]),
            ([5, 8], 2, 8, 2, 32, 8))
        self.assertEqual(
            (pre["prompt_lens"], pre["max_tokens"],
             pre["max_num_batched_tokens"], pre["max_num_seqs"],
             pre["max_model_len"], pre["num_gpu_blocks"]),
            ([16] * 4, 3, 64, 4, 64, 4))
        self.assertEqual((driver.BLOCK_SIZE, driver.TENSOR_PARALLEL_SIZE,
                          driver.PIPELINE_PARALLEL_SIZE, driver.LOAD_FORMAT,
                          driver.TEMPERATURE, driver.IGNORE_EOS),
                         (16, 1, 1, "dummy", 0.0, True))
        self.assertEqual(driver.MAX_NONEMPTY_DECISIONS, 12)
        for name in driver.CASES:
            driver.validate_case_constants(name)

    def test_mixed_case_through_native_cpu_replay(self):
        runtime, trace = _run("mixed")
        decisions = trace["decisions"]
        self.assertEqual(len(decisions), 4)
        first = decisions[0]["proposals"][0]
        self.assertEqual(first["arrived_waiting_order"], [0, 1])
        self.assertEqual([(a["seq_id"], a["num_tokens"]) for a in first["actions"]],
                         [(0, 5), (1, 3)])
        # Selection A decode, B prefill; emitted prompt-first.
        second = decisions[1]
        self.assertEqual([a["seq_id"] for a in second["proposals"][0]["actions"]],
                         [0, 1])
        self.assertEqual(second["schedules"][0]["outputs"]["scheduled"],
                         [[1, 5], [0, 0]])
        self.assertEqual(trace["token_events"], {"0": 2, "1": 2})
        self.assertEqual(trace["idle"]["after"]["free_blocks"], 8)
        self.assertFalse(runtime.scheduler.has_unfinished_seqs())

    def test_preemption_case_through_native_cpu_replay(self):
        runtime, trace = _run("preemption")
        decisions = trace["decisions"]
        self.assertEqual(len(decisions), 8)
        # Prefills fill the pool; zero-demand decodes run with 0 free.
        self.assertEqual(decisions[0]["after"]["free_blocks"], 0)
        self.assertEqual(
            [a["block_demand"] for a in decisions[1]["proposals"][0]["actions"]],
            [0, 0, 0, 0])
        # Allocation-driven tail eviction: free 3, free 2, then the appends.
        boundary = decisions[2]
        self.assertEqual(boundary["proposals"][0]["victims"], [3, 2])
        self.assertEqual([op[:2] for op in boundary["central_ops"]],
                         [["free", 3], ["free", 2], ["append", 0],
                          ["append", 1]])
        self.assertEqual(boundary["after"]["waiting"], [2, 3])
        for sid in ("2", "3"):
            state = boundary["after"]["seqs"][sid]
            self.assertEqual((state["status"], state["prompt_len"],
                              state["prompt_tokens_processed"],
                              state["output_token_ids"]),
                             ("WAITING", 17, 0, []))
        # Readmission of expanded contexts after 0 and 1 finish.
        self.assertEqual(decisions[3]["proposals"][0]["waiting_stop_reason"],
                         "admission_memory")
        self.assertEqual(decisions[4]["schedules"][0]["outputs"]["scheduled"],
                         [[2, 17], [3, 17]])
        # Inherited counter reset: 4 generated-token events for max_tokens 3.
        self.assertEqual(trace["token_events"],
                         {"0": 3, "1": 3, "2": 4, "3": 4})
        final = [o for d in decisions for o in d["request_outputs"]
                 if o["finished"]]
        self.assertEqual([(o["seq_id"], len(o["token_ids"]),
                           len(o["prompt_token_ids"])) for o in final],
                         [(0, 3, 16), (1, 3, 16), (2, 3, 17), (3, 3, 17)])
        # Physical IDs are recorded for both managers, not required equal.
        self.assertIn("free_block_ids", boundary["worker_after"])
        self.assertEqual(trace["idle"]["after"]["free_blocks"], 4)

    def test_checks_reject_a_wrong_expectation(self):
        # Expecting selection order instead of the prompt-first layout.
        cases = copy.deepcopy(driver.CASES)
        cases["mixed"]["decisions"][1]["scheduled"] = [[0, 0], [1, 5]]
        with mock.patch.object(driver, "CASES", cases):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "mixed decision 1: emitted"):
                _run("mixed")
        # Expecting no preemption at the allocation-pressure decision.
        cases = copy.deepcopy(driver.CASES)
        cases["preemption"]["decisions"][2]["victims"] = []
        with mock.patch.object(driver, "CASES", cases):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "preemption decision 2: victims"):
                _run("preemption")

    def test_decision_bound_stops_an_unplanned_run(self):
        cases = copy.deepcopy(driver.CASES)
        cases["mixed"]["decisions"] = cases["mixed"]["decisions"][:2]
        with mock.patch.object(driver, "CASES", cases):
            with self.assertRaisesRegex(driver.CheckFailure,
                                        "more nonempty decisions"):
                _run("mixed")


class CacheCapacityTest(unittest.TestCase):
    def _engine(self, name, profiled):
        case = driver.CASES[name]
        selection = driver.CacheCapacitySelection(chosen=case["num_gpu_blocks"])
        engine_class = driver.build_engine_class(selection)
        engine = object.__new__(engine_class)  # attributes _init_cache reads
        engine.cache_config = CacheConfig(block_size=driver.BLOCK_SIZE,
                                          gpu_memory_utilization=0.5)
        engine.model_config = SimpleNamespace(
            max_model_len=case["max_model_len"])
        calls = []

        def native_run_workers(self, method, *args, **kwargs):
            calls.append((method, args, dict(kwargs)))
            if method == driver.PROFILE_METHOD:
                return [profiled]
            if method == driver.INIT_CACHE_METHOD:
                return [None]
            if method == "execute_model":
                return [SamplerOutput(3, 11)]
            return ("native", method, args, kwargs)

        patch = mock.patch.object(engine_class.__mro__[1], "_run_workers",
                                  native_run_workers)
        return engine, selection, calls, patch

    def test_chosen_capacity_reaches_native_initialization(self):
        for name in driver.CASES:
            with self.subTest(name):
                engine, selection, calls, patch = self._engine(name, 900)
                with patch:
                    engine._init_cache()  # native, including max-context check
                chosen = driver.CASES[name]["num_gpu_blocks"]
                self.assertEqual([c[0] for c in calls],
                                 [driver.PROFILE_METHOD, driver.INIT_CACHE_METHOD])
                self.assertIs(calls[1][2]["cache_config"], engine.cache_config)
                self.assertEqual(engine.cache_config.num_gpu_blocks, chosen)
                self.assertEqual(selection.record()["profiled_available_blocks"],
                                 [900])
                self.assertEqual(selection.profile_calls, 1)
                self.assertEqual(selection.initialized,
                                 [dict(num_gpu_blocks=chosen, result=[None])])

    def test_insufficient_capacity_fails_before_initialization(self):
        engine, selection, calls, patch = self._engine("mixed", 7)
        with patch:
            with self.assertRaisesRegex(driver.CheckFailure, "below the chosen 8"):
                engine._init_cache()
        self.assertEqual([c[0] for c in calls], [driver.PROFILE_METHOD])
        self.assertIsNone(engine.cache_config.num_gpu_blocks)
        self.assertEqual(selection.initialized, [])

    def test_second_profile_fails(self):
        engine, selection, calls, patch = self._engine("preemption", 900)
        with patch:
            engine._init_cache()
            with self.assertRaisesRegex(driver.CheckFailure, "ran 2 times"):
                engine._init_cache()

    def test_other_worker_operations_delegate_unchanged(self):
        engine, selection, calls, patch = self._engine("mixed", 900)
        with patch:
            result = engine._run_workers("add_seq", 1, seq="s")
            samples = engine._run_workers("execute_model", scheduler_outputs="o")
        self.assertEqual(result, ("native", "add_seq", (1,), dict(seq="s")))
        self.assertEqual([(s.seq_id, s.output_token) for s in samples], [(3, 11)])
        self.assertEqual(engine.validation_sampler_outputs, [[3, 11]])
        self.assertEqual(calls, [("add_seq", (1,), dict(seq="s")),
                                 ("execute_model", (), dict(scheduler_outputs="o"))])
        self.assertEqual((selection.profile_calls, selection.initialized), (0, []))


class EngineArgsTest(unittest.TestCase):
    def test_native_configuration_for_each_case(self):
        for name, case in driver.CASES.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                model_config = mock.Mock()
                model_config.get_max_model_len.return_value = case["max_model_len"]
                model_config.get_total_num_layers.return_value = 2
                with mock.patch.object(arg_utils, "ModelConfig",
                                       return_value=model_config) as cls:
                    args = driver.engine_args(name, Path(tmp) / "model",
                                              Path(tmp) / "metrics")
                    _, cache, parallel, sched, metrics = \
                        args.create_engine_configs()
                kwargs = cls.call_args.kwargs
                self.assertEqual((kwargs["load_format"], kwargs["dtype"],
                                  kwargs["max_model_len"],
                                  kwargs["attention_backend"]),
                                 ("dummy", "float16", case["max_model_len"],
                                  "flash_attention"))
                self.assertIs(type(sched), AdaptedVLLMSchedulerConfig)
                self.assertEqual((sched.max_num_seqs, sched.max_model_len,
                                  sched.num_pipeline_stages,
                                  sched.max_num_batched_tokens),
                                 (case["max_num_seqs"], case["max_model_len"],
                                  1, case["max_num_batched_tokens"]))
                self.assertEqual((parallel.tensor_parallel_size,
                                  parallel.pipeline_parallel_size,
                                  cache.block_size,
                                  cache.gpu_memory_utilization),
                                 (1, 1, 16, 0.5))
                self.assertTrue((Path(tmp) / "metrics" / "config.yml").is_file())


class EvidenceTest(unittest.TestCase):
    def test_trace_serialization(self):
        _, trace = _run("preemption")
        with tempfile.TemporaryDirectory() as tmp:
            summary = dict(passed=True, case="preemption",
                           expected_decisions=driver.CASES["preemption"]["decisions"])
            driver.write_artifacts(Path(tmp), summary, trace)
            loaded = json.loads((Path(tmp) / "decision_trace.json").read_text())
            self.assertEqual(len(loaded["decisions"]), 8)
            self.assertEqual(loaded["decisions"][2]["proposals"][0]["victims"],
                             [3, 2])
            self.assertTrue(json.loads(
                (Path(tmp) / "summary.json").read_text())["passed"])

    def test_failure_is_recorded_without_a_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            cmd = [sys.executable, "-B", "scripts/check_adapted_vllm_gpu.py",
                   "--case", "mixed", "--model-path", str(Path(tmp) / "absent"),
                   "--output-dir", str(out)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=str(REPO_ROOT))
            result = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("RESULT: FAIL", result.stdout)
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["failure"]["type"], "CheckFailure")
            self.assertIn("lacks config.json", summary["failure"]["message"])
            self.assertEqual(summary["case_parameters"]["num_gpu_blocks"], 8)
            self.assertIn("git_head", summary["provenance"])
            self.assertFalse(json.loads(
                (out / "cleanup.json").read_text())["engine_constructed"])
            # Earlier artifacts are never overwritten.
            again = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                   capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)
            self.assertIn("already holds run artifacts", again.stderr)


if __name__ == "__main__":
    unittest.main()
