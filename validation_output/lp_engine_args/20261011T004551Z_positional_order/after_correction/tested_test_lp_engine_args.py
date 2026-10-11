import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

import lp_relaxation_scheduler as lrs  # noqa: E402
from sarathi.config import (  # noqa: E402
    LPSchedulerConfig,
    SarathiSchedulerConfig,
    SchedulerType,
    VLLMSchedulerConfig,
)
from sarathi.core.block_space_manager.block_space_manager_registry import (  # noqa: E402
    BlockSpaceManagerRegistry,
)
from sarathi.core.block_space_manager.vllm_block_space_manager import (  # noqa: E402
    VLLMBlockSpaceManager,
)
from sarathi.core.scheduler.lp_scheduler import LPScheduler  # noqa: E402
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry  # noqa: E402
from sarathi.engine import arg_utils  # noqa: E402
from sarathi.engine import llm_engine  # noqa: E402
from sarathi.engine.arg_utils import EngineArgs  # noqa: E402

# Provisional test inputs consistent with the earlier CPU integration
# fixture. They are not defaults or project decisions.
MODEL = "unused-model-name"
RESIDENT_LIMIT = 4
MAX_MODEL_LEN = 32
MODEL_NUM_LAYERS = 2
LP_ARGS = dict(
    lp_b_max=8,
    lp_c_max=4,
    lp_s_max=3,
    lp_memory_reserve=1,
    lp_decode_memory_policy_id="conservative_one_block_v1",
    lp_decode_utility=1,
    lp_prefill_token_utility=1,
    lp_preemption_penalty=1,
)


def numerical_policy():
    return lrs.NumericalPolicy(
        policy_id="lp_relaxation_mvp_v1",
        feasibility_tol=1e-7,
        integrality_tol=1e-6,
        objective_abs_tol=1e-9,
        objective_rel_tol=1e-9,
    )


def fake_model_config():
    """Stands in for ModelConfig, which would load model metadata."""
    model_config = mock.Mock()
    model_config.get_max_model_len.return_value = MAX_MODEL_LEN
    model_config.get_total_num_layers.return_value = MODEL_NUM_LAYERS
    return model_config


class LPEngineArgsTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_dir = os.path.join(self.tmp.name, "run")
        patcher = mock.patch.object(
            arg_utils, "ModelConfig", side_effect=lambda **_: fake_model_config()
        )
        self.model_config_cls = patcher.start()
        self.addCleanup(patcher.stop)

    def lp_kwargs(self, **overrides):
        kwargs = dict(
            model=MODEL,
            scheduler_type="lp",
            max_num_seqs=RESIDENT_LIMIT,
            max_model_len=MAX_MODEL_LEN,
            pipeline_parallel_size=1,
            output_dir=self.output_dir,
            lp_numerical_policy=numerical_policy(),
            **LP_ARGS,
        )
        kwargs.update(overrides)
        return kwargs

    def config_path(self):
        return os.path.join(self.output_dir, "config.yml")

    def test_valid_lp_args_construct_lp_scheduler_config(self):
        policy = numerical_policy()
        args = EngineArgs(**self.lp_kwargs(lp_numerical_policy=policy))
        _, _, parallel_config, scheduler_config, _ = args.create_engine_configs()

        self.assertIs(type(scheduler_config), LPSchedulerConfig)
        self.assertEqual(scheduler_config.type, SchedulerType.LP)
        self.assertEqual(scheduler_config.b_max, 8)
        self.assertEqual(scheduler_config.c_max, 4)
        self.assertEqual(scheduler_config.s_max, 3)
        self.assertEqual(scheduler_config.memory_reserve, 1)
        self.assertEqual(
            scheduler_config.decode_memory_policy_id, "conservative_one_block_v1"
        )
        self.assertEqual(scheduler_config.decode_utility, 1)
        self.assertEqual(scheduler_config.prefill_token_utility, 1)
        self.assertEqual(scheduler_config.preemption_penalty, 1)
        # Shared engine settings, kept separate from s_max.
        self.assertEqual(scheduler_config.max_num_seqs, RESIDENT_LIMIT)
        self.assertEqual(scheduler_config.max_model_len, MAX_MODEL_LEN)
        self.assertEqual(scheduler_config.num_pipeline_stages, 1)
        self.assertEqual(parallel_config.pipeline_parallel_size, 1)
        self.assertNotEqual(scheduler_config.s_max, scheduler_config.max_num_seqs)
        self.assertEqual(scheduler_config.max_num_batched_tokens, 8)
        self.assertEqual(
            self.model_config_cls.call_args.kwargs["max_model_len"], MAX_MODEL_LEN
        )
        self.assertIs(scheduler_config.numerical_policy, policy)

    def test_each_missing_lp_field_fails_before_model_and_config_write(self):
        for name in arg_utils._LP_REQUIRED_FIELDS:
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, name):
                    EngineArgs(**self.lp_kwargs(**{name: None}))
                self.model_config_cls.assert_not_called()
                self.assertFalse(os.path.exists(self.config_path()))

    def test_zero_reserve_and_zero_penalty_are_supplied_values(self):
        args = EngineArgs(
            **self.lp_kwargs(lp_memory_reserve=0, lp_preemption_penalty=0)
        )
        scheduler_config = args.create_engine_configs()[3]
        self.assertEqual(scheduler_config.memory_reserve, 0)
        self.assertEqual(scheduler_config.preemption_penalty, 0)

    def test_configuration_errors_fail_before_model_engine_and_config_write(self):
        cases = [
            (
                TypeError,
                "lp_numerical_policy",
                dict(lp_numerical_policy={"policy_id": "lp_relaxation_mvp_v1"}),
            ),
            (ValueError, "pipeline_parallel_size=2", dict(pipeline_parallel_size=2)),
            (ValueError, "lp_b_max", dict(max_num_batched_tokens=8)),
        ]
        with mock.patch.object(llm_engine, "BaseLLMEngine") as base_engine, \
                mock.patch.object(
                    llm_engine, "PipelineParallelLLMEngine"
                ) as pipeline_engine:
            for error, message, overrides in cases:
                with self.subTest(overrides=list(overrides)):
                    with self.assertRaisesRegex(error, message):
                        llm_engine.LLMEngine.from_engine_args(
                            **self.lp_kwargs(**overrides)
                        )
                    self.model_config_cls.assert_not_called()
                    base_engine.assert_not_called()
                    pipeline_engine.assert_not_called()
                    self.assertFalse(os.path.exists(self.config_path()))

    def test_config_file_records_lp_settings_as_plain_data(self):
        EngineArgs(**self.lp_kwargs())
        with open(self.config_path()) as f:
            text = f.read()
        self.assertNotIn("!!python", text)
        recorded = yaml.safe_load(text)
        for name, value in LP_ARGS.items():
            self.assertEqual(recorded[name], value)
        self.assertEqual(
            recorded["lp_numerical_policy"],
            {
                "policy_id": "lp_relaxation_mvp_v1",
                "feasibility_tol": 1e-7,
                "integrality_tol": 1e-6,
                "objective_abs_tol": 1e-9,
                "objective_rel_tol": 1e-9,
            },
        )
        self.assertEqual(recorded["scheduler_type"], "lp")
        self.assertEqual(recorded["max_num_seqs"], RESIDENT_LIMIT)
        self.assertIsNone(recorded["max_num_batched_tokens"])

    def test_existing_schedulers_need_no_lp_inputs(self):
        common = dict(
            model=MODEL,
            max_num_seqs=RESIDENT_LIMIT,
            max_model_len=MAX_MODEL_LEN,
            output_dir=self.output_dir,
        )
        vllm = EngineArgs(
            scheduler_type="vllm", max_num_batched_tokens=MAX_MODEL_LEN, **common
        ).create_engine_configs()[3]
        self.assertIs(type(vllm), VLLMSchedulerConfig)
        self.assertEqual(vllm.max_num_batched_tokens, MAX_MODEL_LEN)
        self.assertEqual(vllm.max_num_seqs, RESIDENT_LIMIT)
        self.assertEqual(vllm.max_model_len, MAX_MODEL_LEN)

        sarathi = EngineArgs(
            scheduler_type="sarathi", chunk_size=8, **common
        ).create_engine_configs()[3]
        self.assertIs(type(sarathi), SarathiSchedulerConfig)
        self.assertEqual(sarathi.chunk_size, 8)
        self.assertEqual(sarathi.max_num_batched_tokens, 8)
        self.assertEqual(sarathi.max_num_seqs, RESIDENT_LIMIT)

    def test_engine_facade_forwards_lp_configs_to_single_stage_engine(self):
        policy = numerical_policy()
        with mock.patch.object(llm_engine, "BaseLLMEngine") as base_engine, \
                mock.patch.object(
                    llm_engine, "PipelineParallelLLMEngine"
                ) as pipeline_engine:
            engine = llm_engine.LLMEngine.from_engine_args(
                **self.lp_kwargs(lp_numerical_policy=policy)
            )
        pipeline_engine.assert_not_called()
        base_engine.assert_called_once()
        self.assertIs(engine, base_engine.return_value)
        configs = base_engine.call_args.args
        self.assertEqual(len(configs), 5)
        self.assertEqual(base_engine.call_args.kwargs, {})
        _, cache_config, parallel_config, scheduler_config, metrics_config = configs
        self.assertEqual(cache_config.block_size, 16)
        self.assertEqual(parallel_config.pipeline_parallel_size, 1)
        self.assertIs(type(scheduler_config), LPSchedulerConfig)
        self.assertEqual(
            (scheduler_config.b_max, scheduler_config.c_max, scheduler_config.s_max),
            (8, 4, 3),
        )
        self.assertIs(scheduler_config.numerical_policy, policy)
        self.assertEqual(metrics_config.output_dir, self.output_dir)
        self.assertEqual(metrics_config.model_num_layers, MODEL_NUM_LAYERS)

    def test_legacy_positional_metrics_arguments_keep_their_bindings(self):
        # Positional EngineArgs order through output_dir before LP fields were
        # added (commit 09e05151dd76dd4e9ef557b7c4c03a24871d6b5b). Frozen here
        # so the check does not adapt to the current declaration.
        output_dir = os.path.join(self.tmp.name, "legacy_output")
        legacy = (
            ("model", MODEL),
            ("replica_id", 0),
            ("replica_resource_mapping", []),
            ("tokenizer", MODEL),
            ("tokenizer_mode", "auto"),
            ("trust_remote_code", False),
            ("download_dir", None),
            ("load_format", "auto"),
            ("dtype", "auto"),
            ("seed", 0),
            ("pipeline_parallel_size", 1),
            ("tensor_parallel_size", 1),
            ("block_size", 16),
            ("gpu_memory_utilization", 0.85),
            ("revision", None),
            ("scheduler_type", "sarathi"),
            ("max_model_len", None),
            ("max_num_seqs", 256),
            ("max_num_batched_tokens", None),
            ("chunk_size", None),
            ("enable_dynamic_chunking_schedule", False),
            ("low_chunk_size", None),
            ("high_chunk_size", None),
            ("chunk_schedule_max_tokens", None),
            ("chunk_schedule_stages", None),
            ("token_budget", None),
            ("fcfs", None),
            ("fixed_offset", None),
            ("below_memory_limit_offset", None),
            ("above_memory_limit_offset", None),
            ("memory_limit", None),
            ("user_priority", None),
            ("time_between_tokens", None),
            ("process_smallest_prefill", None),
            ("limit_total_decodes", None),
            ("hold_n_scheduler_hold_n", None),
            ("hold_n_scheduler_token_budget", None),
            ("hold_n_scheduler_prefill_factor", None),
            ("write_metrics", False),
            ("output_dir", output_dir),
        )
        with mock.patch.object(arg_utils, "open", create=True) as open_mock, \
                mock.patch.object(arg_utils.os, "makedirs") as makedirs:
            args = EngineArgs(*(value for _, value in legacy))
        for name, value in legacy:
            self.assertEqual(getattr(args, name), value, name)
        for name in arg_utils._LP_REQUIRED_FIELDS:
            self.assertIsNone(getattr(args, name), name)
        open_mock.assert_not_called()
        makedirs.assert_not_called()
        self.assertFalse(os.path.exists(output_dir))

    def test_registries_resolve_lp_scheduler_and_block_manager(self):
        scheduler_config = EngineArgs(**self.lp_kwargs()).create_engine_configs()[3]
        self.assertIs(SchedulerRegistry.get_class(scheduler_config.type), LPScheduler)
        self.assertIs(
            BlockSpaceManagerRegistry.get_class(scheduler_config.type),
            VLLMBlockSpaceManager,
        )


if __name__ == "__main__":
    unittest.main()
