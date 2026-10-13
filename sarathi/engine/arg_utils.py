import dataclasses
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import yaml

from sarathi.config import (
    LP_OBSOLETE_ARGUMENT_MIGRATION,
    BaseSchedulerConfig,
    CacheConfig,
    FasterTransformerSchedulerConfig,
    LPSchedulerConfig,
    MetricsConfig,
    ModelConfig,
    OrcaSchedulerConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
    SchedulerType,
    SimpleChunkingSchedulerConfig,
    VLLMSchedulerConfig,
    SLAISchedulerConfig,
)

from sarathi.core.scheduler.hold_n_scheduler import Hold_NSchedulerConfig

if TYPE_CHECKING:
    from lp_relaxation_scheduler import NumericalPolicy

_LP_REQUIRED_FIELDS = (
    "lp_b_max",
    "lp_c_max",
    "lp_s_max",
    "lp_decode_utility",
    "lp_prefill_token_utility",
    "lp_preemption_penalty",
    "lp_numerical_policy",
)
# Removed LP fields, kept only to reject supplied values with migration
# guidance and to preserve the positions of later fields.
_LP_OBSOLETE_FIELDS = ("lp_memory_reserve", "lp_decode_memory_policy_id")


class _Omitted:
    """Private default of a removed field, distinguishing omission from any
    explicitly supplied value, including ``None``. Copies keep its identity."""

    def __repr__(self):
        return "<omitted>"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return "_OMITTED"


_OMITTED = _Omitted()

@dataclass
class EngineArgs:
    """Arguments for Sarathi engine."""

    model: str
    replica_id: int = 0
    replica_resource_mapping: List[Tuple[str, int]] = dataclasses.field(
        default_factory=list
    )
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    dtype: str = "auto"
    seed: int = 0
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    gpu_memory_utilization: float = 0.85
    revision: Optional[str] = None
    # scheduler parameters
    scheduler_type: str = "sarathi"
    max_model_len: Optional[int] = None
    max_num_seqs: int = 256
    # vllm scheduler parameters
    max_num_batched_tokens: Optional[int] = None
    # sarathi scheduler parameters
    chunk_size: Optional[int] = None
    enable_dynamic_chunking_schedule: bool = False
    low_chunk_size: Optional[int] = None
    high_chunk_size: Optional[int] = None
    chunk_schedule_max_tokens: Optional[int] = None
    chunk_schedule_stages: Optional[int] = None
    # SLAI scheduler parameters
    token_budget : Optional[int] = None # Add your token budget argument
    fcfs: Optional[bool] = None # Add your fcfs argument
    fixed_offset: Optional[bool] = None # Add your fixed offset argument
    below_memory_limit_offset: Optional[int] = None # Add your below memory limit offset argument
    above_memory_limit_offset: Optional[int] = None # Add your above memory limit offset argument
    memory_limit: Optional[float] = None # Add your memory limit argument
    user_priority: Optional[bool] = None # Add your user priority argument
    time_between_tokens: Optional[float] = None # Add your tbt argument
    process_smallest_prefill : Optional[bool] = None # Add your process_smallest_prefill argument
    limit_total_decodes: Optional[int] = None  # Add your limit_total_decodes argument
    # hold_n scheduler parameters
    hold_n_scheduler_hold_n: Optional[int] = None   # flag created by ConfigParser
    hold_n_scheduler_token_budget: Optional[int] = None  # flag created by ConfigParser
    hold_n_scheduler_prefill_factor: Optional[int] = None  # flag created by Config
    # Metrics store parameters
    write_metrics: bool = True
    output_dir: str = "."
    wandb_project: Optional[str] = None
    wandb_sweep_id: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_run_name: Optional[str] = None
    enable_op_level_metrics: bool = False
    enable_cpu_op_level_metrics: bool = False
    enable_chrome_trace: bool = False
    enable_request_outputs: bool = False
    keep_individual_batch_metrics: bool = False
    attention_backend: str = "flash_attention"
    # LP scheduler parameters: all required for scheduler_type="lp", no
    # defaults. The resident limit, model length, and stage count come from
    # the shared max_num_seqs, max_model_len, and pipeline_parallel_size.
    lp_b_max: Optional[int] = None  # combined prefill/decode token budget
    lp_c_max: Optional[int] = None  # per-request prefill chunk cap
    lp_s_max: Optional[int] = None  # execution-action limit
    lp_memory_reserve: Any = _OMITTED  # removed; any supplied value is rejected
    lp_decode_memory_policy_id: Any = _OMITTED  # removed; rejected
    lp_decode_utility: Optional[float] = None
    lp_prefill_token_utility: Optional[float] = None
    lp_preemption_penalty: Optional[float] = None
    lp_numerical_policy: Optional["NumericalPolicy"] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model
        supplied = [
            n for n in _LP_OBSOLETE_FIELDS if getattr(self, n) is not _OMITTED
        ]
        if supplied:
            raise ValueError(
                f"EngineArgs {', '.join(supplied)}: "
                + LP_OBSOLETE_ARGUMENT_MIGRATION
            )
        if self.scheduler_type == SchedulerType.LP.name.lower():
            self._verify_lp_args()
        if self.write_metrics:
            os.makedirs(self.output_dir, exist_ok=True)
            recorded = {
                k: v for k, v in asdict(self).items()
                if k not in _LP_OBSOLETE_FIELDS
            }
            with open(f"{self.output_dir}/config.yml", "w") as f:
                yaml.dump(recorded, f, default_flow_style=False, sort_keys=False)

    def _verify_lp_args(self) -> None:
        missing = [name for name in _LP_REQUIRED_FIELDS if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"LP scheduler requires explicit values for: {', '.join(missing)}"
            )
        from lp_relaxation_scheduler import NumericalPolicy

        if not isinstance(self.lp_numerical_policy, NumericalPolicy):
            raise TypeError(
                "lp_numerical_policy must be an lp_relaxation_scheduler."
                f"NumericalPolicy, got {type(self.lp_numerical_policy).__name__}"
            )
        if self.pipeline_parallel_size != 1:
            raise ValueError(
                "LP scheduler supports exactly one pipeline stage, got "
                f"pipeline_parallel_size={self.pipeline_parallel_size!r}"
            )
        if self.max_num_batched_tokens is not None:
            raise ValueError(
                "max_num_batched_tokens is not used by the LP scheduler; "
                "set the combined token budget with lp_b_max instead"
            )

    def _get_scheduler_config(
        self, model_config: ModelConfig, num_pipeline_stages: int
    ) -> BaseSchedulerConfig:
        if self.scheduler_type == SchedulerType.VLLM.name.lower():
            scheduler_config = VLLMSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.max_num_batched_tokens,
            )
        elif self.scheduler_type == SchedulerType.ORCA.name.lower():
            scheduler_config = OrcaSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
            )
        elif self.scheduler_type == SchedulerType.FASTER_TRANSFORMER.name.lower():
            scheduler_config = FasterTransformerSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
            )
        elif self.scheduler_type == SchedulerType.SARATHI.name.lower():
            scheduler_config = SarathiSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.chunk_size,
                self.fcfs,
                self.enable_dynamic_chunking_schedule,
                self.low_chunk_size,
                self.high_chunk_size,
                self.chunk_schedule_max_tokens,
                self.chunk_schedule_stages,
            )
        elif self.scheduler_type == SchedulerType.SIMPLE_CHUNKING.name.lower():
            scheduler_config = SimpleChunkingSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.chunk_size,
            )
        elif self.scheduler_type == SchedulerType.HOLD_N.name.lower():
            assert self.hold_n_scheduler_hold_n is not None, "hold_n must be set"
            assert self.hold_n_scheduler_token_budget is not None, "token_budget must be set"
            assert self.hold_n_scheduler_prefill_factor is not None, "prefill_factor must be set"
    
            scheduler_config = Hold_NSchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.hold_n_scheduler_hold_n,
                self.hold_n_scheduler_token_budget,
                self.hold_n_scheduler_prefill_factor,
            )
        elif self.scheduler_type == SchedulerType.SLAI_SCHEDULER.name.lower():
            assert self.token_budget is not None, "token_budget must be set for slai scheduler"
            assert self.fcfs is not None, "fcfs must be set for slai scheduler"
            assert self.fixed_offset is not None, "fixed_offset must be set for slai scheduler"
            assert self.below_memory_limit_offset is not None, "below_memory_limit_offset must be set for slai scheduler"
            assert self.above_memory_limit_offset is not None, "above_memory_limit_offset must be set for slai scheduler"
            assert self.memory_limit is not None, "memory_limit must be set for slai scheduler"
            assert self.user_priority is not None, "user_priority must be set for slai scheduler"
            assert self.time_between_tokens is not None, "time_between_tokens must be set for slai scheduler"
            assert self.limit_total_decodes is not None, "limit_total_decodes must be set for slai scheduler"

            scheduler_config = SLAISchedulerConfig(
                self.max_num_seqs,
                model_config.get_max_model_len(),
                num_pipeline_stages,
                self.token_budget,
                self.fcfs,
                self.fixed_offset,
                self.below_memory_limit_offset,
                self.above_memory_limit_offset,
                self.memory_limit,
                self.user_priority,
                self.time_between_tokens,
                self.limit_total_decodes,
            )
        elif self.scheduler_type == SchedulerType.LP.name.lower():
            scheduler_config = LPSchedulerConfig(
                max_num_seqs=self.max_num_seqs,
                max_model_len=model_config.get_max_model_len(),
                num_pipeline_stages=num_pipeline_stages,
                b_max=self.lp_b_max,
                c_max=self.lp_c_max,
                s_max=self.lp_s_max,
                decode_utility=self.lp_decode_utility,
                prefill_token_utility=self.lp_prefill_token_utility,
                preemption_penalty=self.lp_preemption_penalty,
                numerical_policy=self.lp_numerical_policy,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        return scheduler_config

    def create_engine_configs(
        self,
    ) -> Tuple[
        ModelConfig, CacheConfig, ParallelConfig, BaseSchedulerConfig, MetricsConfig
    ]:
        model_config = ModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            download_dir=self.download_dir,
            load_format=self.load_format,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            max_model_len=self.max_model_len,
            attention_backend=self.attention_backend,
        )
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            replica_resource_mapping=self.replica_resource_mapping,
        )
        scheduler_config = self._get_scheduler_config(
            model_config=model_config, num_pipeline_stages=self.pipeline_parallel_size
        )
        metrics_config = MetricsConfig(
            replica_id=self.replica_id,
            write_metrics=self.write_metrics,
            output_dir=self.output_dir,
            wandb_project=self.wandb_project,
            wandb_group=self.wandb_group,
            wandb_run_name=self.wandb_run_name,
            wandb_sweep_id=self.wandb_sweep_id,
            wandb_run_id=self.wandb_run_id,
            enable_op_level_metrics=self.enable_op_level_metrics,
            enable_cpu_op_level_metrics=self.enable_cpu_op_level_metrics,
            enable_chrome_trace=self.enable_chrome_trace,
            enable_request_outputs=self.enable_request_outputs,
            keep_individual_batch_metrics=self.keep_individual_batch_metrics,
            model_num_layers=model_config.get_total_num_layers(),
        )
        return (
            model_config,
            cache_config,
            parallel_config,
            scheduler_config,
            metrics_config,
        )
