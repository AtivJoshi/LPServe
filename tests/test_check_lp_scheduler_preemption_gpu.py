"""CPU checks for the four-block cache initialization in the native
preemption GPU validation driver. No engine, Ray worker, or GPU is used: the
engine subclass is exercised through native ``_init_cache`` with the native
worker dispatch replaced by a recording test double."""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_lp_scheduler_preemption_gpu as driver  # noqa: E402

from sarathi.config import CacheConfig  # noqa: E402

PROFILE = driver.PROFILE_METHOD
INIT_CACHE = driver.INIT_CACHE_METHOD


class DriverImportTest(unittest.TestCase):
    def test_import_loads_no_engine_worker_ray_or_torch(self):
        code = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import check_lp_scheduler_preemption_gpu; "
            "print(sorted(m for m in ('torch', 'ray', "
            "'sarathi.engine.base_llm_engine', 'sarathi.worker.base_worker') "
            "if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "[]")


class CacheCapacitySelectionTest(unittest.TestCase):
    def test_actual_profile_is_retained_and_chosen_capacity_returned(self):
        selection = driver.CacheCapacitySelection()
        self.assertEqual(selection.select([1234]), [4])
        self.assertEqual(selection.profiled, [1234])
        self.assertEqual(selection.record()["profiled_available_blocks"], [1234])
        self.assertEqual(selection.record()["chosen_initialized_blocks"], 4)

    def test_insufficient_capacity_fails_without_substitution(self):
        selection = driver.CacheCapacitySelection()
        with self.assertRaisesRegex(driver.CheckFailure, "below the chosen 4"):
            selection.select([3])
        self.assertEqual(selection.profiled, [3])

    def test_unexpected_result_shape_or_second_profile_fails(self):
        for results in ([4, 4], 4, [True], ["4"]):
            with self.subTest(results=results):
                with self.assertRaisesRegex(driver.CheckFailure, "shape"):
                    driver.CacheCapacitySelection().select(results)
        selection = driver.CacheCapacitySelection()
        selection.select([10])
        with self.assertRaisesRegex(driver.CheckFailure, "ran 2 times"):
            selection.select([10])


class EngineCacheInitializationTest(unittest.TestCase):
    def _engine(self, profiled):
        selection = driver.CacheCapacitySelection()
        engine_class = driver.build_engine_class(selection)
        # Bypass construction: only the attributes native _init_cache reads.
        engine = object.__new__(engine_class)
        engine.cache_config = CacheConfig(block_size=16,
                                          gpu_memory_utilization=0.5)
        engine.model_config = SimpleNamespace(max_model_len=64)
        calls = []

        def native_run_workers(self, method, *args, **kwargs):
            calls.append((method, args, dict(kwargs)))
            if method == PROFILE:
                return [profiled]
            if method == INIT_CACHE:
                return [None]
            return ("native", method, args, kwargs)

        patch = mock.patch.object(
            engine_class.__mro__[1], "_run_workers", native_run_workers,
        )
        return engine, selection, calls, patch

    def test_chosen_capacity_reaches_native_initialization(self):
        engine, selection, calls, patch = self._engine(profiled=900)
        with patch:
            engine._init_cache()
        self.assertEqual([c[0] for c in calls], [PROFILE, INIT_CACHE])
        self.assertEqual(
            calls[0][2],
            dict(get_all_outputs=True, block_size=16,
                 gpu_memory_utilization=0.5),
        )
        self.assertIs(calls[1][2]["cache_config"], engine.cache_config)
        self.assertEqual(engine.cache_config.num_gpu_blocks, 4)
        self.assertEqual(selection.record()["profiled_available_blocks"], [900])
        self.assertEqual(
            selection.record()["initializations"],
            [dict(num_gpu_blocks=4, result=[None])],
        )

    def test_insufficient_capacity_fails_before_cache_initialization(self):
        engine, selection, calls, patch = self._engine(profiled=3)
        with patch:
            with self.assertRaises(driver.CheckFailure):
                engine._init_cache()
        self.assertEqual([c[0] for c in calls], [PROFILE])
        self.assertIsNone(engine.cache_config.num_gpu_blocks)
        self.assertEqual(selection.initialized, [])

    def test_other_worker_operations_delegate_unchanged(self):
        engine, selection, calls, patch = self._engine(profiled=900)
        with patch:
            result = engine._run_workers(
                "execute_model", 1, scheduler_outputs="outputs",
            )
        self.assertEqual(
            result, ("native", "execute_model", (1,),
                     dict(scheduler_outputs="outputs")),
        )
        self.assertEqual(
            calls, [("execute_model", (1,), dict(scheduler_outputs="outputs"))],
        )
        self.assertEqual(selection.profile_calls, 0)
        self.assertEqual(selection.initialized, [])


if __name__ == "__main__":
    unittest.main()
