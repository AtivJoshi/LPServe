"""Bounded dummy-weight GPU check of native preemption under the LP scheduler.

Runs two requests through a real ``BaseLLMEngine`` with the registered
``LPScheduler``, one GPU worker (tensor parallel 1, one pipeline stage), dummy
TinyLlama weights, and native replay and completion. The cache is initialized
with exactly four 16-token blocks so that admitting B (48 prompt tokens, three
blocks) requires recovering the two blocks held by partially prefilled A (17
prompt tokens).

Request A is submitted alone and prefilled for 16 tokens. B is submitted only
after that step completes. The next real mapping, solve, and extraction must
preempt A and admit B. The run then requires B to finish, A to be readmitted
and recomputed from prompt token zero, A to finish, and one ordinary idle
call. Every expected emitted schedule entry is asserted; tied relaxed values
at later steps are recorded, not prescribed.

Cache sizing: normal worker profiling runs unchanged. Its actual result is
recorded and must be at least four blocks; the value four is then supplied to
the engine's native cache initialization, which builds the worker cache and
worker block manager. The central scheduler is built from the same cache
configuration. A validation worker subclass, selected through the engine's
existing worker-selection hook, records its own block-manager free/allocate
calls and exposes read-only primitive state; it changes no native behavior.

This establishes the bounded dummy-weight path only. It does not establish
numerical agreement with real weights, useful generation, interruption after
generated tokens, arbitrary-workload completion, fairness, pipeline support,
corrected output semantics, or performance. Native text is not evidence of
correct token conversion.

Run from the repository root with the root on PYTHONPATH:

    PYTHONPATH="$PWD" timeout 300s python -B \\
        scripts/check_lp_scheduler_preemption_gpu.py \\
        --model-path <local TinyLlama snapshot directory> \\
        --output-dir validation_output/lp_scheduler_preemption_gpu/<UTC timestamp>

Exit status is zero only if every assertion passes.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Read-only helpers from the single-request check; its main() is not used.
import check_lp_scheduler_gpu as single
from check_lp_scheduler_gpu import (CheckFailure, Observer, check,
                                    environment_record, outputs_record,
                                    request_output_record, to_jsonable)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Approved scoped test inputs (provisional, not project-wide policy).
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOAD_FORMAT = "dummy"
DTYPE = "float16"
ATTENTION_BACKEND = "flash_attention"
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 64
BLOCK_SIZE = 16
GPU_MEMORY_UTILIZATION = 0.5
SEED = 42
TRUST_REMOTE_CODE = True
TOKENIZER_MODE = "auto"
INITIALIZED_NUM_GPU_BLOCKS = 4
MAX_NUM_SEQS = 1
B_MAX = 64
C_MAX = 16
S_MAX = 1
MEMORY_RESERVE = 0
DECODE_POLICY_ID = "conservative_one_block_v1"
DECODE_UTILITY = 40.0
PREFILL_TOKEN_UTILITY = 1.0
PREEMPTION_PENALTY = 1.0
NUMERICAL_POLICY_FIELDS = dict(
    policy_id="lp_relaxation_mvp_v1",
    feasibility_tol=1e-7,
    integrality_tol=1e-6,
    objective_abs_tol=1e-9,
    objective_rel_tol=1e-9,
)
TEMPERATURE = 0.0
MAX_TOKENS = 1
IGNORE_EOS = True
# Distinct search starts give disjoint, non-special prompt-token lists.
PROMPTS = dict(A=dict(length=17, search_start=1000),
               B=dict(length=48, search_start=2000))
FINISHED = "FINISHED_LENGTH_CAPPED"
PROFILE_METHOD = "profile_num_available_blocks"
INIT_CACHE_METHOD = "init_cache_engine"
WORKER_STATE_METHOD = "validation_state"
PIPELINE_CALLS = ["map_scheduler_state", "solve_and_extract", "execute_plan",
                  "schedule", "_run_workers"]

# Expected emitted (preempted IDs, scheduled [seq_id, prompt_chunk_len]) per
# nonempty step; A is seq 0 and B is seq 1.
EXPECTED_STEPS = [
    dict(name="a_admission_prefill", preempted=[], scheduled=[[0, 16]]),
    dict(name="a_preemption_b_admission", preempted=[0], scheduled=[[1, 16]]),
    dict(name="b_resident_prefill", preempted=[], scheduled=[[1, 16]]),
    dict(name="b_prompt_completion", preempted=[], scheduled=[[1, 16]]),
    dict(name="b_decode_finish", preempted=[], scheduled=[[1, 0]]),
    dict(name="a_readmission_prefill", preempted=[], scheduled=[[0, 16]]),
    dict(name="a_prompt_completion", preempted=[], scheduled=[[0, 1]]),
    dict(name="a_decode_finish", preempted=[], scheduled=[[0, 0]]),
]
BOUNDARY_STEP = 1
READMISSION_STEP = 5


class CacheCapacitySelection:
    """Retains the actual profiled capacity and supplies the deliberately
    chosen capacity to native cache initialization, exactly once."""

    def __init__(self, chosen=INITIALIZED_NUM_GPU_BLOCKS):
        self.chosen = chosen
        self.profiled = None
        self.profile_calls = 0
        self.initialized = []

    def select(self, results):
        self.profile_calls += 1
        check(self.profile_calls == 1,
              f"cache profiling ran {self.profile_calls} times; expected once")
        check(isinstance(results, list) and len(results) == TENSOR_PARALLEL_SIZE
              and all(isinstance(r, int) and not isinstance(r, bool)
                      for r in results),
              f"unexpected profiling result shape {results!r}; expected one "
              "integer per worker")
        self.profiled = list(results)
        check(min(results) >= self.chosen,
              f"actual profiled capacity {min(results)} blocks is below the "
              f"chosen {self.chosen}; no larger pool is substituted")
        return [self.chosen] * len(results)

    def record_initialization(self, num_gpu_blocks, result):
        check(self.profiled is not None,
              "cache initialization ran before profiling")
        self.initialized.append(dict(num_gpu_blocks=num_gpu_blocks,
                                     result=to_jsonable(result)))
        check(len(self.initialized) == 1,
              "cache initialization ran more than once")
        check(num_gpu_blocks == self.chosen,
              f"cache initialization received {num_gpu_blocks} blocks; "
              f"expected {self.chosen}")

    def record(self):
        return dict(profiled_available_blocks=self.profiled,
                    chosen_initialized_blocks=self.chosen,
                    profile_calls=self.profile_calls,
                    initializations=self.initialized)


def build_worker_class():
    """Return a ``BaseWorker`` subclass that records its own block-manager
    free/allocate calls and exposes read-only primitive state. Defined
    lazily so importing this module does not import the worker stack; it is
    serialized by value for the Ray worker."""
    from sarathi.worker.base_worker import BaseWorker

    class ValidationWorker(BaseWorker):
        def init_cache_engine(self, cache_config):
            super().init_cache_engine(cache_config)
            manager = self.seq_manager.block_manager
            ops = self._validation_block_ops = []
            native_free, native_allocate = manager.free, manager.allocate

            def free(seq):
                table = manager.block_tables.get(seq.seq_id, [])
                ops.append(["free", seq.seq_id, [b.block_number for b in table]])
                return native_free(seq)

            def allocate(seq):
                result = native_allocate(seq)
                ops.append(["allocate", seq.seq_id, [
                    b.block_number for b in manager.block_tables[seq.seq_id]]])
                return result

            manager.free, manager.allocate = free, allocate

        def validation_state(self):
            manager = self.seq_manager.block_manager
            caches = [t for layer in self.gpu_cache for t in (
                layer if isinstance(layer, (tuple, list)) else (layer,))]
            seqs = {}
            for seq_id, seq in self.seq_manager.seq_map.items():
                seqs[str(seq_id)] = dict(
                    status=seq.get_status().name,
                    prompt_token_ids=list(seq.prompt_token_ids),
                    prompt_tokens_processed=seq.get_num_prompt_tokens_processed(),
                    prompt_processing_finished=seq.prompt_processing_finished,
                    output_token_ids=list(seq.output_token_ids),
                    logical_blocks=len(seq.logical_token_blocks),
                )
            state = dict(
                cache_config_num_gpu_blocks=self.cache_config.num_gpu_blocks,
                cache_engine_num_gpu_blocks=self.cache_engine.num_gpu_blocks,
                cache_layers=len(self.gpu_cache),
                cache_tensor_block_dims=sorted({int(t.shape[0]) for t in caches}),
                cache_tensor_shapes=sorted({tuple(t.shape) for t in caches}),
                manager_total_blocks=manager.num_total_gpu_blocks,
                allocator_num_blocks=manager.gpu_allocator.num_blocks,
                free_blocks=manager.get_num_free_gpu_blocks(),
                free_block_ids=[b.block_number
                                for b in manager.gpu_allocator.free_blocks],
                block_tables={str(k): [b.block_number for b in v]
                              for k, v in manager.block_tables.items()},
                seqs=seqs,
                block_ops=list(self._validation_block_ops),
            )
            del self._validation_block_ops[:]
            return state

    return ValidationWorker


def build_engine_class(selection):
    """Return a ``BaseLLMEngine`` subclass that uses the validation worker
    and passes the chosen capacity from ``selection`` to native cache
    initialization. Every other worker call is delegated unchanged."""
    from sarathi.engine.base_llm_engine import BaseLLMEngine

    class FourBlockValidationEngine(BaseLLMEngine):
        def _get_worker_impl(self):
            return build_worker_class()

        def _run_workers(self, method, *args, **kwargs):
            result = super()._run_workers(method, *args, **kwargs)
            if method == PROFILE_METHOD:
                return selection.select(result)
            if method == INIT_CACHE_METHOD:
                selection.record_initialization(
                    kwargs["cache_config"].num_gpu_blocks, result)
            return result

    return FourBlockValidationEngine


def select_prompt_ids(tokenizer, vocab_size, start, count):
    special = set(tokenizer.all_special_ids)
    ids = []
    candidate = start
    while len(ids) < count:
        token = tokenizer.convert_ids_to_tokens(candidate)
        if (candidate < vocab_size and candidate < len(tokenizer)
                and candidate not in special and token is not None
                and token != tokenizer.unk_token):
            ids.append(candidate)
        candidate += 1
    return ids, tokenizer.convert_ids_to_tokens(ids)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=30).stdout
    except Exception as err:  # provenance only; never affects the check
        return f"unavailable: {err!r}"


def provenance_record():
    files = ["scripts/check_lp_scheduler_preemption_gpu.py",
             "scripts/check_lp_scheduler_gpu.py",
             "tests/test_check_lp_scheduler_preemption_gpu.py",
             "tests/test_lp_scheduler.py", "tests/test_lpserve_plan_execution.py",
             "sarathi/core/scheduler/lp_scheduler.py",
             "sarathi/core/scheduler/base_scheduler.py",
             "sarathi/core/block_space_manager/base_block_space_manager.py",
             "sarathi/core/sequence_manager/base_sequence_manager.py",
             "sarathi/core/sequence_manager/worker_sequence_manager.py",
             "sarathi/engine/base_llm_engine.py",
             "sarathi/worker/base_worker.py",
             "sarathi/model_executor/model_runner.py",
             "lp_relaxation_scheduler.py", "lpserve_state_mapping.py",
             "lpserve_plan_execution.py"]
    return dict(
        git_head=git("rev-parse", "HEAD").strip(),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        git_status_short=git("status", "--short", "--untracked-files=all"),
        git_diff_head_sha256=hashlib.sha256(
            git("diff", "HEAD").encode()).hexdigest(),
        file_sha256={f: (sha256(REPO_ROOT / f) if (REPO_ROOT / f).exists()
                         else None) for f in files},
        executed_script_sha256=sha256(__file__),
        imported_helper_sha256=sha256(single.__file__),
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def central_state(engine):
    """Selected primitive central state; holds no live objects."""
    scheduler = engine.scheduler
    manager = scheduler.block_manager
    seqs = {}
    for seq_id, seq in engine.seq_manager.seq_map.items():
        seqs[str(seq_id)] = dict(
            status=seq.get_status().name,
            prompt_token_ids=list(seq.prompt_token_ids),
            prompt_tokens_processed=seq.get_num_prompt_tokens_processed(),
            prompt_processing_finished=seq.prompt_processing_finished,
            output_token_ids=list(seq.output_token_ids),
            logical_blocks=len(seq.logical_token_blocks),
        )
    return dict(
        iteration_id=scheduler._iteration_id,
        num_running_batches=scheduler.num_running_batches,
        waiting=[s.seq_id for s in scheduler.waiting],
        running=[s.seq_id for s in scheduler.running],
        total_blocks=manager.num_total_gpu_blocks,
        free_blocks=manager.get_num_free_gpu_blocks(),
        free_block_ids=[b.block_number for b in manager.gpu_allocator.free_blocks],
        block_tables={str(k): [b.block_number for b in v]
                      for k, v in manager.block_tables.items()},
        seqs=seqs,
    )


def check_pool(tag, label, total, tables, free_ids):
    allocated = [b for table in tables.values() for b in table]
    check(len(allocated) == len(set(allocated)),
          f"{tag}: {label} allocations repeat a block: {tables}")
    check(len(free_ids) == len(set(free_ids)),
          f"{tag}: {label} free list repeats a block: {free_ids}")
    check(not set(allocated) & set(free_ids),
          f"{tag}: {label} block is both allocated and free")
    check(sorted(allocated + free_ids) == list(range(total)),
          f"{tag}: {label} pool not conserved: {tables} free {free_ids}")


def check_quiescent(tag, central, worker):
    """Ownership, conservation, and central/worker agreement at a completed
    step boundary. Physical block IDs may differ between managers."""
    waiting, running = central["waiting"], central["running"]
    check(len(set(waiting)) == len(waiting) and len(set(running)) == len(running)
          and not set(waiting) & set(running),
          f"{tag}: ownership not unique/disjoint: {waiting} {running}")
    check(len(running) <= MAX_NUM_SEQS, f"{tag}: resident count {running}")
    check(central["num_running_batches"] == 0, f"{tag}: batches in flight")
    check(sorted(waiting + running) == sorted(int(k) for k in central["seqs"]),
          f"{tag}: owned set differs from central sequence map")
    check(set(central["seqs"]) == set(worker["seqs"]),
          f"{tag}: central/worker sequence maps differ")
    check(set(central["block_tables"]) == {str(k) for k in running},
          f"{tag}: central tables {central['block_tables']} vs running")
    check({k: len(v) for k, v in central["block_tables"].items()}
          == {k: len(v) for k, v in worker["block_tables"].items()},
          f"{tag}: central/worker allocation counts differ")
    check(central["free_blocks"] == worker["free_blocks"],
          f"{tag}: free counts differ {central['free_blocks']} vs "
          f"{worker['free_blocks']}")
    check_pool(tag, "central", INITIALIZED_NUM_GPU_BLOCKS,
               central["block_tables"], central["free_block_ids"])
    check_pool(tag, "worker", INITIALIZED_NUM_GPU_BLOCKS,
               worker["block_tables"], worker["free_block_ids"])
    for key, seq in central["seqs"].items():
        mirror = worker["seqs"][key]
        for field in ("status", "prompt_token_ids", "prompt_tokens_processed",
                      "prompt_processing_finished", "output_token_ids"):
            check(seq[field] == mirror[field],
                  f"{tag}: seq {key} {field} central {seq[field]} worker "
                  f"{mirror[field]}")


def check_decision(tag, expected, before, events, policy_json):
    """Pipeline sequence, mapped inputs, plan/output agreement, and limits."""
    check([e["call"] for e in events] == PIPELINE_CALLS,
          f"{tag}: call sequence {[e['call'] for e in events]}")
    mapped, solved, executed, scheduled, worker = events
    check(worker["method"] == "execute_model",
          f"{tag}: worker call {worker['method']}")
    check(mapped["result_type"] == "StateSnapshot",
          f"{tag}: mapping returned {mapped['result_type']}: {mapped['result']}")
    snap = mapped["result"]
    problem = snap["lp_problem"]
    check(mapped["num_running_batches"] == 0
          and snap["num_running_batches"] == 0
          and snap["num_pipeline_stages"] == 1,
          f"{tag}: in-flight or stage state at the mapping boundary")
    check(snap["scheduler_iteration_id"] == before["iteration_id"] + 1
          == mapped["iteration_id"], f"{tag}: mapping iteration")
    check((problem["b_max"], problem["c_max"], problem["s_max"], problem["w"])
          == (B_MAX, C_MAX, S_MAX, MEMORY_RESERVE), f"{tag}: mapped limits")
    check(snap["memory_reserve"] == MEMORY_RESERVE
          and snap["resident_limit"] == MAX_NUM_SEQS
          and snap["decode_memory_policy_id"] == DECODE_POLICY_ID
          and snap["numerical_policy"] == policy_json,
          f"{tag}: mapped policy inputs")
    check(snap["free_physical_blocks"] == problem["m_free"]
          == before["free_blocks"], f"{tag}: mapped free blocks")
    check(sorted(r["raw_seq_id"] for r in snap["requests"])
          == sorted(before["waiting"] + before["running"]),
          f"{tag}: mapped request set")
    for r in snap["requests"]:
        check(r["utility"] == dict(decode_utility=DECODE_UTILITY,
                                   prefill_token_utility=PREFILL_TOKEN_UTILITY,
                                   preemption_penalty=PREEMPTION_PENALTY),
              f"{tag}: mapped utility {r['utility']}")
    check(solved["result_type"] == "SchedulingSuccess",
          f"{tag}: solve returned {solved['result_type']}: {solved['result']}")
    plan = solved["result"]["plan"]
    raw = {r["request_id"]: r["raw_seq_id"] for r in snap["requests"]}
    order = {r["request_id"]: r["order_key"] for r in snap["requests"]}
    decisions = plan["decisions"]
    victims = sorted((d for d in decisions if d["preempt"]),
                     key=lambda d: order[d["request_id"]])
    actions = sorted((d for d in decisions if d["prefill_tokens"] or d["decode"]),
                     key=lambda d: (0 if d["prefill_tokens"] else 1,
                                    order[d["request_id"]]))
    check(executed["result_type"] == "SchedulerOutputs",
          f"{tag}: executor returned {executed['result_type']}: "
          f"{executed['result']}")
    emitted = scheduled["outputs"]
    check(executed["result"] == emitted,
          f"{tag}: scheduler output differs from executor output")
    check(emitted["id"] == snap["scheduler_iteration_id"], f"{tag}: output id")
    check(emitted["preempted_seq_ids"] == [raw[d["request_id"]] for d in victims]
          and emitted["scheduled"] == [[raw[d["request_id"]], d["prefill_tokens"]]
                                       for d in actions]
          and emitted["ignored_seq_ids"] == [],
          f"{tag}: emitted {emitted} differs from plan {decisions}")
    check(emitted["preempted_seq_ids"] == expected["preempted"]
          and emitted["scheduled"] == expected["scheduled"],
          f"{tag}: emitted preempted {emitted['preempted_seq_ids']} scheduled "
          f"{emitted['scheduled']}; expected {expected['preempted']} "
          f"{expected['scheduled']}")
    tokens = sum(c if c > 0 else 1 for _, c in emitted["scheduled"])
    check(tokens <= B_MAX and 1 <= len(emitted["scheduled"]) <= S_MAX,
          f"{tag}: token/action limits ({tokens}, {len(emitted['scheduled'])})")
    check([s for s, _ in worker["sampler_outputs"]]
          == [s for s, _ in emitted["scheduled"]],
          f"{tag}: sampler outputs {worker['sampler_outputs']} not associated "
          f"with scheduled entries {emitted['scheduled']}")
    return snap, solved["result"]


def check_boundary(tag, snap, solved, sched_state, central_ops, worker_ops,
                   before, after, worker_after, tol):
    """The preemption decision and its native execution and replay."""
    problem = snap["lp_problem"]
    reqs = {r["request_id"]: r for r in snap["requests"]}
    a, b = reqs["0"], reqs["1"]
    check(problem["legal_preemption_ids"] == ["0"],
          f"{tag}: legal victims {problem['legal_preemption_ids']}")
    check((problem["m_free"], problem["w"]) == (2, 0), f"{tag}: m_free/w")
    check((a["ownership"], a["status"], a["prompt_tokens_processed"],
           a["physical_block_count"], a["preemption_recovery"])
          == ("running", "PAUSED", 16, 2, 2), f"{tag}: mapped A {a}")
    check((b["ownership"], b["prompt_len"], b["prefill_fixed_charge"])
          == ("waiting", 48, 3), f"{tag}: mapped B {b}")
    relaxed = {d["request_id"]: d for d in solved["relaxed"]["decisions"]}
    for rid, want in (("0", (0, 0, 0, 0.5)), ("1", (16, 0, 1, 0))):
        d = relaxed[rid]
        got = (d["x"], d["y"], d["prefill_indicator"], d["z"])
        check(all(abs(g - w) <= tol for g, w in zip(got, want)),
              f"{tag}: relaxed {rid} {got}; expected {want}")
    check(abs(solved["relaxed"]["normalized_objective"] - 15.5) <= tol,
          f"{tag}: relaxed objective {solved['relaxed']['normalized_objective']}")
    plan = solved["plan"]
    check([(d["request_id"], d["prefill_tokens"], d["decode"], d["preempt"])
           for d in plan["decisions"]] == [("0", 0, 0, 1), ("1", 16, 0, 0)],
          f"{tag}: integer plan {plan['decisions']}")
    check(plan["dominant_preemption_ids"] == ["0"]
          and plan["safety_preemption_ids"] == [],
          f"{tag}: preemption classification")
    check(abs(plan["objective"] - 15.0) <= tol,
          f"{tag}: integer objective {plan['objective']}")

    # Central execution: free A, then admit B; A untouched until replay.
    check([op[:2] for op in central_ops] == [["free", 0], ["allocate", 1]],
          f"{tag}: central block operations {central_ops}")
    check(central_ops[0][2] == before["block_tables"]["0"]
          and len(central_ops[1][2]) == 3,
          f"{tag}: central free/allocate tables {central_ops}")
    check(sched_state["waiting"] == [0] and sched_state["running"] == [1]
          and set(sched_state["block_tables"]) == {"1"}
          and len(sched_state["block_tables"]["1"]) == 3
          and sched_state["free_blocks"] == 1,
          f"{tag}: central state after execution {sched_state}")
    check(sched_state["seqs"]["0"] == before["seqs"]["0"],
          f"{tag}: A changed before replay: {sched_state['seqs']['0']}")

    # Worker replay: free A's local blocks, then allocate B's.
    check([op[:2] for op in worker_ops] == [["free", 0], ["allocate", 1]]
          and len(worker_ops[0][2]) == 2 and len(worker_ops[1][2]) == 3,
          f"{tag}: worker block operations {worker_ops}")

    # Central and worker reset of A for recomputation.
    for label, seqs in (("central", after["seqs"]), ("worker", worker_after["seqs"])):
        state = seqs["0"]
        check(state["status"] == "WAITING"
              and state["prompt_tokens_processed"] == 0
              and state["prompt_processing_finished"] is False
              and state["output_token_ids"] == []
              and state["prompt_token_ids"] == before["seqs"]["0"]["prompt_token_ids"],
              f"{tag}: {label} A after replay {state}")
        check(seqs["1"]["prompt_tokens_processed"] == 16,
              f"{tag}: {label} B progress {seqs['1']}")
    check(after["waiting"] == [0] and after["running"] == [1]
          and len(worker_after["block_tables"]["1"]) == 3
          and "0" not in worker_after["block_tables"],
          f"{tag}: ownership/allocation after replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True,
                        help="existing local directory with the TinyLlama "
                        "config and tokenizer")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "decision_trace.json"
    summary_path = out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")
    model_path = Path(args.model_path).resolve()

    # Local assets only; never download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    trace = dict(steps=[], idle=None)
    summary = dict(passed=False, failure=None,
                   provenance=provenance_record(),
                   argv=sys.argv, model_path=str(model_path))
    engine = None
    try:
        check((model_path / "config.json").is_file()
              and (model_path / "tokenizer.json").is_file(),
              f"model path {model_path} lacks config.json/tokenizer.json")
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import the LP modules")

        import lp_relaxation_scheduler as lrs
        import lpserve_plan_execution as lpe
        import lpserve_state_mapping as lsm
        from sarathi.config import (CacheConfig, LPSchedulerConfig,
                                    MetricsConfig, ModelConfig, ParallelConfig)
        from sarathi.core.datatypes.sampling_params import SamplingParams
        from sarathi.core.scheduler.lp_scheduler import LPScheduler

        numerical_policy = lrs.NumericalPolicy(**NUMERICAL_POLICY_FIELDS)
        policy_json = to_jsonable(numerical_policy)
        model_config = ModelConfig(
            model=str(model_path), tokenizer=str(model_path),
            tokenizer_mode=TOKENIZER_MODE, trust_remote_code=TRUST_REMOTE_CODE,
            download_dir=None, load_format=LOAD_FORMAT, dtype=DTYPE, seed=SEED,
            revision=None, max_model_len=MAX_MODEL_LEN,
            attention_backend=ATTENTION_BACKEND,
        )
        cache_config = CacheConfig(block_size=BLOCK_SIZE,
                                   gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE)
        scheduler_config = LPSchedulerConfig(
            max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN,
            num_pipeline_stages=PIPELINE_PARALLEL_SIZE, b_max=B_MAX,
            c_max=C_MAX, s_max=S_MAX, memory_reserve=MEMORY_RESERVE,
            decode_memory_policy_id=DECODE_POLICY_ID,
            decode_utility=DECODE_UTILITY,
            prefill_token_utility=PREFILL_TOKEN_UTILITY,
            preemption_penalty=PREEMPTION_PENALTY,
            numerical_policy=numerical_policy,
        )
        # Enabled metrics with every optional output off, as in the earlier
        # bounded checks: the disabled mode lacks attributes the engine reads
        # on every step. Nothing is plotted or written.
        metrics_config = MetricsConfig(
            replica_id=0, write_metrics=True,
            output_dir=str(out_dir / "metrics_store_unused"),
            wandb_project=None, wandb_group=None, wandb_run_name=None,
            wandb_sweep_id=None, wandb_run_id=None,
            enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
            enable_chrome_trace=False, enable_request_outputs=False,
            keep_individual_batch_metrics=True,
            model_num_layers=model_config.get_total_num_layers(),
        )
        # The native generic profiler builds max_num_seqs prompts totalling
        # max_num_batched_tokens (= b_max): one 64-token profiling request.
        check(scheduler_config.max_num_batched_tokens == B_MAX,
              "profiling token count must equal b_max")

        summary["environment"] = environment_record()
        print("environment:", json.dumps(summary["environment"], indent=1),
              flush=True)

        selection = CacheCapacitySelection()
        engine_class = build_engine_class(selection)
        t0 = time.monotonic()
        engine = engine_class(model_config, cache_config, parallel_config,
                              scheduler_config, metrics_config)
        init_seconds = time.monotonic() - t0
        scheduler = engine.scheduler
        central_manager = scheduler.block_manager
        check(type(scheduler) is LPScheduler,
              f"registered scheduler is {type(scheduler).__name__}")

        # Capacity confirmation in all three places.
        worker_init = engine._run_workers(WORKER_STATE_METHOD,
                                          get_all_outputs=True)
        check(len(worker_init) == 1, "expected one worker")
        worker_init = worker_init[0]
        capacity = dict(
            **selection.record(),
            engine_cache_config_num_gpu_blocks=cache_config.num_gpu_blocks,
            central_manager_total_blocks=central_manager.num_total_gpu_blocks,
            central_allocator_num_blocks=central_manager.gpu_allocator.num_blocks,
            central_free_blocks=central_manager.get_num_free_gpu_blocks(),
            central_watermark_blocks=central_manager.watermark_blocks,
            worker={k: worker_init[k] for k in (
                "cache_config_num_gpu_blocks", "cache_engine_num_gpu_blocks",
                "cache_layers", "cache_tensor_block_dims",
                "cache_tensor_shapes", "manager_total_blocks",
                "allocator_num_blocks", "free_blocks", "block_ops")},
        )
        summary["cache_capacity"] = to_jsonable(capacity)
        print("cache capacity:", json.dumps(summary["cache_capacity"]),
              flush=True)
        n = INITIALIZED_NUM_GPU_BLOCKS
        check(len(selection.initialized) == 1 and selection.profile_calls == 1,
              "profiling/initialization did not each run exactly once")
        check(cache_config.num_gpu_blocks == n
              and central_manager.num_total_gpu_blocks == n
              and central_manager.gpu_allocator.num_blocks == n
              and central_manager.get_num_free_gpu_blocks() == n
              and not central_manager.block_tables,
              f"central capacity {capacity}")
        check(central_manager.watermark_blocks == 0, "central watermark")
        check(worker_init["cache_config_num_gpu_blocks"] == n
              and worker_init["cache_engine_num_gpu_blocks"] == n
              and worker_init["cache_tensor_block_dims"] == [n]
              and worker_init["cache_layers"] > 0
              and worker_init["manager_total_blocks"] == n
              and worker_init["allocator_num_blocks"] == n
              and worker_init["free_blocks"] == n
              and worker_init["block_tables"] == {}
              and worker_init["block_ops"] == [],
              f"worker capacity {worker_init}")

        vocab_size = model_config.hf_config.vocab_size
        prompts = {}
        for name, spec in PROMPTS.items():
            ids, tokens = select_prompt_ids(engine.tokenizer, vocab_size,
                                            spec["search_start"], spec["length"])
            check(len(ids) == spec["length"], f"prompt {name} length")
            prompts[name] = dict(token_ids=ids, tokens=tokens)
        check(not set(prompts["A"]["token_ids"]) & set(prompts["B"]["token_ids"]),
              "prompts must be distinct")
        summary["config"] = dict(
            model=MODEL_NAME, model_path=str(model_path),
            load_format=LOAD_FORMAT, requested_dtype=DTYPE,
            resolved_dtype=str(model_config.dtype),
            hf_config_commit_hash=getattr(model_config.hf_config,
                                          "_commit_hash", None),
            attention_backend=ATTENTION_BACKEND,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
            max_model_len=model_config.max_model_len, block_size=BLOCK_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=SEED,
            trust_remote_code=TRUST_REMOTE_CODE, tokenizer_mode=TOKENIZER_MODE,
            vocab_size=vocab_size, tokenizer_len=len(engine.tokenizer),
            eos_token_id=engine.tokenizer.eos_token_id, prompts=prompts,
            sampling=dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                          ignore_eos=IGNORE_EOS, stop=[]),
            b_max=B_MAX, c_max=C_MAX, s_max=S_MAX, max_num_seqs=MAX_NUM_SEQS,
            memory_reserve=MEMORY_RESERVE,
            decode_memory_policy_id=DECODE_POLICY_ID,
            utilities=dict(decode=DECODE_UTILITY,
                           prefill_token=PREFILL_TOKEN_UTILITY,
                           preemption_penalty=PREEMPTION_PENALTY),
            numerical_policy=policy_json,
            profiling_max_num_batched_tokens=(
                scheduler_config.max_num_batched_tokens),
            profiling_max_num_seqs=scheduler_config.max_num_seqs,
            metrics_mode="enabled, all optional outputs off, never plotted",
            engine_init_seconds=round(init_seconds, 3),
        )
        print("config:", json.dumps(summary["config"]), flush=True)

        # Observation only: every wrapper delegates once and returns the
        # native result unchanged.
        observer = Observer()
        observer.wrap(lsm, "map_scheduler_state", lambda a, k, r: dict(
            num_running_batches=a[0].num_running_batches,
            iteration_id=a[0]._iteration_id,
            result_type=type(r).__name__, result=to_jsonable(r)))
        observer.wrap(lrs, "solve_and_extract", lambda a, k, r: dict(
            result_type=type(r).__name__, result=to_jsonable(r)))
        observer.wrap(lpe, "execute_plan", lambda a, k, r: dict(
            result_type=type(r).__name__,
            result=(outputs_record(r) if type(r).__name__ == "SchedulerOutputs"
                    else to_jsonable(r))))
        observer.wrap(scheduler, "schedule", lambda a, k, r: dict(
            outputs=outputs_record(r), state=central_state(engine),
            block_ops=list(central_ops)))
        observer.wrap(engine, "_run_workers", lambda a, k, r: dict(
            method=a[0],
            sampler_outputs=([[o.seq_id, o.output_token] for o in r]
                             if a[0] == "execute_model" and r is not None
                             else None)))

        central_ops = []
        native_free = central_manager.free
        native_allocate = central_manager.allocate

        def central_free(seq):
            table = central_manager.block_tables.get(seq.seq_id, [])
            central_ops.append(["free", seq.seq_id,
                                [b.block_number for b in table]])
            return native_free(seq)

        def central_allocate(seq):
            result = native_allocate(seq)
            central_ops.append(["allocate", seq.seq_id, [
                b.block_number
                for b in central_manager.block_tables[seq.seq_id]]])
            return result

        central_manager.free = central_free
        central_manager.allocate = central_allocate

        def worker_state():
            (state,) = engine._run_workers(WORKER_STATE_METHOD,
                                           get_all_outputs=True)
            return state

        def submit(name):
            engine.add_request(
                prompt=None,
                sampling_params=SamplingParams(
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                    ignore_eos=IGNORE_EOS, stop=None),
                prompt_token_ids=list(prompts[name]["token_ids"]),
            )

        tol = numerical_policy.feasibility_tol
        finish_order = []
        final_outputs = {}
        for index, expected in enumerate(EXPECTED_STEPS):
            tag = f"step {index} {expected['name']}"
            if index == 0:
                submit("A")
                check(sorted(engine.seq_manager.seq_map) == [0],
                      "A must be seq 0")
            if index == BOUNDARY_STEP:
                submit("B")
                check(sorted(engine.seq_manager.seq_map) == [0, 1],
                      "B must be seq 1")
            before = central_state(engine)
            worker_before = worker_state()
            check_quiescent(f"{tag} (before)", before, worker_before)
            observer.events.clear()
            del central_ops[:]
            step_outputs = engine.step()
            events = list(observer.events)
            after = central_state(engine)
            worker_after = worker_state()
            step_ops = list(central_ops)  # whole step, including completion
            record = dict(index=index, name=expected["name"], before=before,
                          worker_before=worker_before, events=events,
                          after=after, worker_after=worker_after,
                          central_block_ops=step_ops,
                          request_outputs=[request_output_record(o)
                                           for o in step_outputs])
            trace["steps"].append(record)

            snap, solved = check_decision(tag, expected, before, events,
                                          policy_json)
            sched_state = events[3]["state"]
            step_central_ops = events[3]["block_ops"]
            worker_ops = worker_after["block_ops"]
            check(len(sched_state["running"]) <= MAX_NUM_SEQS
                  and not set(sched_state["waiting"]) & set(sched_state["running"]),
                  f"{tag}: central ownership after execution")
            check(after["iteration_id"] == before["iteration_id"] + 1,
                  f"{tag}: iteration advanced "
                  f"{after['iteration_id'] - before['iteration_id']}")
            check_quiescent(f"{tag} (after)", after, worker_after)

            if index == 0:
                a = after["seqs"]["0"]
                check(a["status"] == "PAUSED"
                      and a["prompt_tokens_processed"] == 16
                      and not a["prompt_processing_finished"]
                      and a["output_token_ids"] == []
                      and len(after["block_tables"]["0"]) == 2
                      and after["free_blocks"] == 2
                      and worker_after["free_blocks"] == 2,
                      f"{tag}: A after first prefill {after}")
                check([op[:2] for op in step_ops] == [["allocate", 0]]
                      and [op[:2] for op in worker_ops] == [["allocate", 0]],
                      f"{tag}: admission ops {step_ops} {worker_ops}")
            elif index == BOUNDARY_STEP:
                check_boundary(tag, snap, solved, sched_state, step_central_ops,
                               worker_ops, before, after, worker_after, tol)
            elif index == READMISSION_STEP:
                a_snap = [r for r in snap["requests"] if r["raw_seq_id"] == 0][0]
                check((a_snap["ownership"], a_snap["status"],
                       a_snap["physical_block_count"],
                       a_snap["prompt_tokens_processed"],
                       a_snap["prefill_fixed_charge"])
                      == ("waiting", "WAITING", 0, 0, 2),
                      f"{tag}: mapped A before readmission {a_snap}")
                check([op[:2] for op in step_central_ops] == [["allocate", 0]]
                      and len(step_central_ops[0][2]) == 2,
                      f"{tag}: central readmission ops {step_central_ops}")
                check([op[:2] for op in worker_ops] == [["allocate", 0]]
                      and len(worker_ops[0][2]) == 2,
                      f"{tag}: worker readmission ops {worker_ops}")
                check(after["seqs"]["0"]["prompt_tokens_processed"] == 16
                      and worker_after["seqs"]["0"]["prompt_tokens_processed"] == 16,
                      f"{tag}: A recomputed prompt progress")
            elif index not in (4, 7):
                check(step_ops == [] and worker_ops == [],
                      f"{tag}: unexpected block operations "
                      f"{step_ops} {worker_ops}")
            if index in (BOUNDARY_STEP, READMISSION_STEP):
                check(step_ops == step_central_ops,
                      f"{tag}: central block operations after execution "
                      f"{step_ops}")

            for out in step_outputs:
                if out.finished:
                    finish_order.append(out.seq_id)
                    final_outputs[out.seq_id] = request_output_record(out)
            if index in (4, 7):
                finished_id = expected["scheduled"][0][0]
                check([o.seq_id for o in step_outputs if o.finished]
                      == [finished_id], f"{tag}: finished outputs "
                      f"{record['request_outputs']}")
                fin = final_outputs[finished_id]
                check(len(fin["token_ids"]) == MAX_TOKENS
                      and fin["finish_reason"] == "length",
                      f"{tag}: finished output {fin}")
                check(str(finished_id) not in after["seqs"]
                      and str(finished_id) not in worker_after["seqs"]
                      and str(finished_id) not in after["block_tables"]
                      and str(finished_id) not in worker_after["block_tables"],
                      f"{tag}: finished request still held")
                # Decode at gap zero allocates nothing; completion frees.
                check(step_central_ops == []
                      and [op[:2] for op in step_ops] == [["free", finished_id]]
                      and [op[:2] for op in worker_ops] == [["free", finished_id]],
                      f"{tag}: finish block operations central "
                      f"{step_ops} worker {worker_ops}")
            print(f"{tag}: preempted {events[3]['outputs']['preempted_seq_ids']}"
                  f" scheduled {events[3]['outputs']['scheduled']} central "
                  f"free {after['free_blocks']} worker free "
                  f"{worker_after['free_blocks']} sampler "
                  f"{events[4]['sampler_outputs']}", flush=True)

        check(finish_order == [1, 0], f"finish order {finish_order}")
        drained = central_state(engine)
        worker_drained = worker_state()
        check(drained["waiting"] == [] and drained["running"] == []
              and drained["seqs"] == {} and drained["block_tables"] == {}
              and drained["free_blocks"] == INITIALIZED_NUM_GPU_BLOCKS,
              f"central not drained: {drained}")
        check(worker_drained["seqs"] == {} and worker_drained["block_tables"] == {}
              and worker_drained["free_blocks"] == INITIALIZED_NUM_GPU_BLOCKS,
              f"worker not drained: {worker_drained}")
        check(not engine.has_unfinished_requests(), "unfinished requests")

        # One ordinary idle call: no mapping, solving, execution, or worker.
        before = central_state(engine)
        observer.events.clear()
        idle_outputs = engine.step()
        after = central_state(engine)
        idle_events = list(observer.events)
        trace["idle"] = dict(before=before, events=idle_events, after=after,
                             request_outputs=[request_output_record(o)
                                              for o in idle_outputs])
        check(idle_outputs == [], f"idle step returned {idle_outputs}")
        check([e["call"] for e in idle_events] == ["schedule"]
              and idle_events[0]["outputs"]["scheduled"] == []
              and idle_events[0]["outputs"]["preempted_seq_ids"] == [],
              f"idle step calls {[e['call'] for e in idle_events]}")
        check(after["iteration_id"] == before["iteration_id"] + 1,
              "idle step must advance the iteration once")
        check({**after, "iteration_id": None} == {**before, "iteration_id": None},
              f"idle step changed state: {before} -> {after}")
        print(f"idle: no outputs, calls {[e['call'] for e in idle_events]}, "
              f"iteration {before['iteration_id']} -> {after['iteration_id']}",
              flush=True)

        summary.update(
            passed=True,
            finish_order=finish_order,
            final_outputs={str(k): v for k, v in final_outputs.items()},
            decisions=[dict(name=s["name"],
                            preempted=s["events"][3]["outputs"]["preempted_seq_ids"],
                            scheduled=s["events"][3]["outputs"]["scheduled"],
                            sampler_outputs=s["events"][4]["sampler_outputs"])
                       for s in trace["steps"]],
            boundary=dict(
                relaxed=trace["steps"][BOUNDARY_STEP]["events"][1]["result"]["relaxed"],
                plan=trace["steps"][BOUNDARY_STEP]["events"][1]["result"]["plan"],
                central_block_ops=trace["steps"][BOUNDARY_STEP]["events"][3]["block_ops"],
                worker_block_ops=trace["steps"][BOUNDARY_STEP]["worker_after"]["block_ops"],
            ),
        )
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        print("CHECK FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        cleanup = dict(engine_constructed=engine is not None)
        try:
            trace_path.write_text(json.dumps(trace, indent=1))
            summary_path.write_text(json.dumps(summary, indent=1))
        finally:
            # Stop only the Ray runtime this process started.
            if "ray" in sys.modules:
                ray = sys.modules["ray"]
                cleanup["ray_initialized_before_shutdown"] = ray.is_initialized()
                ray.shutdown()
                cleanup["ray_initialized_after_shutdown"] = ray.is_initialized()
            print("cleanup:", json.dumps(cleanup), flush=True)
            (out_dir / "cleanup.json").write_text(json.dumps(cleanup, indent=1))
    print("RESULT:", "PASS" if summary["passed"] else "FAIL", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
