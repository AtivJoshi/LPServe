"""Serve one recorded LP-inference request alone and compare generated IDs.

Diagnostic driver only. It replays the first input of the recorded run
``validation_output/lp_inference/20261006T064204Z`` (raw prompt text, no chat
template) through a fresh ``BaseLLMEngine`` with one scheduler:

- ``--scheduler vllm``: the registered ``VLLMScheduler`` with
  ``max_num_seqs=1`` and ``max_num_batched_tokens=32``. Expected shape: one
  nine-token prefill, then four decodes.
- ``--scheduler lp``: ``run_lp_inference.create_engine`` unchanged (resident
  limit 3, b_max 96, c_max 8, s_max 2, reserve 1). Its decisions are recorded
  and checked with the recorded run's per-step checks, not prescribed. If
  ``<output-dir>/../reference/summary.json`` exists, its tokens are compared
  too.

Model, sampling, and metrics settings are the constants of
``scripts/run_lp_inference.py``. The request is submitted through native
``add_request`` with the original text and the verified recorded IDs. After
the run, the native incremental detokenizer is replayed on CPU over the same
IDs to explain the saved text. Observers delegate once and return results
unchanged. Any failure stops the run; the engine is never reused.

    PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    timeout 300s python -B <this file> --scheduler vllm|lp \\
        --model-path <snapshot> --recorded-results <results.json> \\
        --output-dir <case dir>
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEBUG_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_lp_inference as rli  # noqa: E402
from check_lp_scheduler_contention_gpu import (capture_state,  # noqa: E402
                                               outputs_with_counts,
                                               worker_record)
from check_lp_scheduler_gpu import (Observer, check,  # noqa: E402
                                    environment_record, request_output_record,
                                    to_jsonable)
from check_lp_scheduler_reference_gpu import (compare_tokens,  # noqa: E402
                                              inspect_snapshot, sha256)

RECORDED_RUN = REPO_ROOT / "validation_output/lp_inference/20261006T064204Z"
RECORDED_INDEX = 0
RECORDED_PROMPT = "Explain why plants need sunlight."
RECORDED_PROMPT_IDS = [1, 12027, 7420, 2020, 18577, 817, 6575, 4366, 29889]
RECORDED_GENERATED_IDS = [2, 29871, 13, 29966]
EXPECTED_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
EXPECTED_WEIGHT_SHA256 = (
    "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933")

VLLM_MAX_NUM_SEQS = 1
VLLM_MAX_NUM_BATCHED_TOKENS = 32
FINISHED = "FINISHED_LENGTH_CAPPED"

PROVENANCE_FILES = [
    "scripts/run_lp_inference.py",
    "scripts/check_lp_scheduler_gpu.py",
    "scripts/check_lp_scheduler_contention_gpu.py",
    "scripts/check_lp_scheduler_reference_gpu.py",
    "sarathi/engine/base_llm_engine.py",
    "sarathi/core/scheduler/vllm_scheduler.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/core/sequence_manager/engine_sequence_manager.py",
    "sarathi/core/datatypes/sequence.py",
    "sarathi/transformers_utils/tokenizer.py",
    "sarathi/model_executor/model_runner.py",
    "sarathi/model_executor/attention/flash_attention_wrapper.py",
    "sarathi/model_executor/layers/sampler.py",
    "sarathi/config.py",
    "lp_relaxation_scheduler.py",
    "lpserve_state_mapping.py",
    "lpserve_plan_execution.py",
    "validation_output/lp_inference/20261006T064204Z/validate_run.py",
]


def git(*args):
    import subprocess
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=REPO_ROOT, timeout=30).stdout
    except Exception as err:  # provenance only
        return f"unavailable: {err!r}"


def provenance_record():
    diff = git("diff", "HEAD")
    return dict(
        git_head=git("rev-parse", "HEAD").strip(),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        git_status_short=git("status", "--short", "--untracked-files=all"),
        git_diff_head_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        driver_sha256=sha256(__file__),
        file_sha256={f: sha256(REPO_ROOT / f) for f in PROVENANCE_FILES
                     if (REPO_ROOT / f).exists()},
        loaded_modules=os.environ.get("LOADEDMODULES"),
    )


def load_recorded(path):
    """The recorded first input and its LP result; checked against the
    values this diagnosis was asked about."""
    path = Path(path).resolve()
    results = json.loads(path.read_text())
    entry = results[RECORDED_INDEX]
    prompts_path = path.parent.parent / "prompts.json"
    prompts = json.loads(prompts_path.read_text())
    check(entry["index"] == RECORDED_INDEX
          and entry["prompt"] == prompts[RECORDED_INDEX] == RECORDED_PROMPT
          and entry["prompt_token_ids"] == RECORDED_PROMPT_IDS
          and entry["generated_token_ids"] == RECORDED_GENERATED_IDS,
          f"recorded entry differs from the expected input: {entry}")
    return dict(
        results_path=str(path), results_sha256=sha256(path),
        prompts_path=str(prompts_path), prompts_sha256=sha256(prompts_path),
        all_generated_token_ids=[r["generated_token_ids"] for r in results],
        entry=entry)


def load_validate_run():
    """The recorded run's per-step LP checks (import-safe module)."""
    spec = importlib.util.spec_from_file_location(
        "recorded_validate_run", RECORDED_RUN / "validate_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_vllm_engine(model_path, metrics_dir):
    """Same model, cache, parallel, and metrics settings as
    ``run_lp_inference.create_engine``, with the registered VLLMScheduler."""
    from sarathi.config import (CacheConfig, MetricsConfig, ModelConfig,
                                ParallelConfig, VLLMSchedulerConfig)
    from sarathi.core.scheduler.vllm_scheduler import VLLMScheduler
    from sarathi.engine.base_llm_engine import BaseLLMEngine

    model_config = ModelConfig(
        model=model_path, tokenizer=model_path,
        tokenizer_mode=rli.TOKENIZER_MODE,
        trust_remote_code=rli.TRUST_REMOTE_CODE, download_dir=None,
        load_format=rli.LOAD_FORMAT, dtype=rli.DTYPE, seed=rli.SEED,
        revision=None, max_model_len=rli.MAX_MODEL_LEN,
        attention_backend=rli.ATTENTION_BACKEND,
    )
    cache_config = CacheConfig(block_size=rli.BLOCK_SIZE,
                               gpu_memory_utilization=rli.GPU_MEMORY_UTILIZATION)
    parallel_config = ParallelConfig(
        pipeline_parallel_size=rli.PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=rli.TENSOR_PARALLEL_SIZE)
    scheduler_config = VLLMSchedulerConfig(
        max_num_seqs=VLLM_MAX_NUM_SEQS, max_model_len=rli.MAX_MODEL_LEN,
        num_pipeline_stages=rli.PIPELINE_PARALLEL_SIZE,
        max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS)
    metrics_config = MetricsConfig(
        replica_id=0, write_metrics=True, output_dir=str(metrics_dir),
        wandb_project=None, wandb_group=None, wandb_run_name=None,
        wandb_sweep_id=None, wandb_run_id=None,
        enable_op_level_metrics=False, enable_cpu_op_level_metrics=False,
        enable_chrome_trace=False, enable_request_outputs=False,
        keep_individual_batch_metrics=True,
        model_num_layers=model_config.get_total_num_layers(),
    )
    engine = BaseLLMEngine(model_config, cache_config, parallel_config,
                           scheduler_config, metrics_config)
    check(type(engine.scheduler) is VLLMScheduler,
          f"registered scheduler is {type(engine.scheduler).__name__}")
    return engine


def engine_facts(engine):
    mc, sc = engine.model_config, engine.scheduler_config
    keys = ["max_num_seqs", "max_model_len", "num_pipeline_stages",
            "max_num_batched_tokens", "b_max", "c_max", "s_max",
            "memory_reserve", "decode_memory_policy_id", "decode_utility",
            "prefill_token_utility", "preemption_penalty"]
    return dict(
        scheduler_type=type(engine.scheduler).__name__,
        scheduler_config_type=type(sc).__name__,
        scheduler_config={k: getattr(sc, k) for k in keys if hasattr(sc, k)},
        numerical_policy=to_jsonable(getattr(sc, "numerical_policy", None)),
        model=mc.model, tokenizer=mc.tokenizer, load_format=mc.load_format,
        dtype=str(mc.dtype), max_model_len=mc.max_model_len, seed=mc.seed,
        attention_backend=str(mc.attention_backend),
        tokenizer_mode=mc.tokenizer_mode,
        trust_remote_code=mc.trust_remote_code,
        hf_config_commit_hash=getattr(mc.hf_config, "_commit_hash", None),
        block_size=engine.cache_config.block_size,
        gpu_memory_utilization=engine.cache_config.gpu_memory_utilization,
        tensor_parallel_size=engine.parallel_config.tensor_parallel_size,
        pipeline_parallel_size=engine.parallel_config.pipeline_parallel_size,
        metrics=dict(
            write_metrics=engine.metrics_config.write_metrics,
            enable_request_outputs=engine.metrics_config.enable_request_outputs,
            enable_chrome_trace=engine.metrics_config.enable_chrome_trace),
        tokenizer_class=type(engine.tokenizer).__name__,
        eos_token_id=engine.tokenizer.eos_token_id,
        eos_token=engine.tokenizer.eos_token,
        num_total_gpu_blocks=engine.scheduler.block_manager.num_total_gpu_blocks,
        watermark_blocks=engine.scheduler.block_manager.watermark_blocks,
    )


def check_vllm_step(tag, number, seq_id, prompt_len, before, events, after,
                    outputs):
    """Expected shape: step 1 prefills the whole prompt; steps 2-5 decode."""
    prefill = number == 1
    req_b, req_a = before["requests"]["0"], after["requests"]["0"]
    calls = [e["call"] for e in events]
    check(calls == ["schedule", "_run_workers"]
          and events[1]["method"] == "execute_model", f"{tag}: calls {calls}")
    emitted = events[0]["outputs"]
    want = [[seq_id, prompt_len if prefill else 0]]
    check(emitted["scheduled"] == want, f"{tag}: emitted "
          f"{emitted['scheduled']}, expected {want}")
    check(not emitted["ignored_seq_ids"] and not emitted["preempted_seq_ids"],
          f"{tag}: controls emitted")
    check([s for s, _ in events[1]["sampler_outputs"]] == [seq_id],
          f"{tag}: sampler ids {events[1]['sampler_outputs']}")
    check(before["num_running_batches"] == 0
          and before["engine_seq_ids"] == [seq_id], f"{tag}: boundary")
    if prefill:
        check(before["waiting"] == [seq_id] and before["running"] == []
              and req_b["block_ids"] is None, f"{tag}: waiting {before}")
        check(req_a["prompt_tokens_processed"] == prompt_len
              and req_a["prompt_processing_finished"]
              and req_a["output_token_ids"] == [],
              f"{tag}: prefill completion {req_a}")
    else:
        check(before["waiting"] == [] and before["running"] == [seq_id]
              and req_b["block_ids"], f"{tag}: resident {before}")
        check(req_a["generated"] == req_b["generated"] + 1
              and req_a["output_token_ids"][:-1] == req_b["output_token_ids"],
              f"{tag}: decode completion {req_b} -> {req_a}")
    finished = number == 1 + rli.MAX_TOKENS
    check(req_a["is_finished"] == finished, f"{tag}: finished {req_a}")
    check(len(outputs) == 1 and outputs[0]["seq_id"] == seq_id
          and outputs[0]["finished"] == finished
          and outputs[0]["token_ids"] == req_a["output_token_ids"],
          f"{tag}: request outputs {outputs}")
    if finished:
        check(req_a["status"] == FINISHED and req_a["block_ids"] is None
              and outputs[0]["finish_reason"] == "length",
              f"{tag}: finish {req_a}")
    sched = events[0]["state"]
    return dict(emitted=emitted["scheduled"],
                sampler_outputs=events[1]["sampler_outputs"],
                free=dict(before=before["free_blocks"],
                          sched=sched["free_blocks"],
                          after=after["free_blocks"]),
                finished=["0"] if finished else [])


def replay_detokenization(tokenizer, prompt_ids, generated_ids):
    """Replay ``EngineSequenceManager._decode_seq`` on CPU for each generated
    token, with the same native function and arguments."""
    from sarathi.transformers_utils.tokenizer import detokenize_incrementally
    tokens, prefix_offset, read_offset, text, steps = None, 0, 0, "", []
    for i in range(1, len(generated_ids) + 1):
        all_ids = list(prompt_ids) + list(generated_ids[:i])
        new_tokens, new_text, prefix_offset, read_offset = (
            detokenize_incrementally(tokenizer, all_input_ids=all_ids,
                                     prev_tokens=tokens,
                                     prefix_offset=prefix_offset,
                                     read_offset=read_offset,
                                     skip_special_tokens=True))
        tokens = new_tokens if tokens is None else tokens + new_tokens
        text += new_text
        steps.append(dict(token_id=generated_ids[i - 1],
                          new_tokens=list(new_tokens), new_text=new_text,
                          prefix_offset=prefix_offset,
                          read_offset=read_offset, tokens=list(tokens)))
    return dict(text=text, steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheduler", required=True, choices=["vllm", "lp"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--recorded-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    mode = args.scheduler

    out_dir = Path(args.output_dir)
    trace_path, summary_path = out_dir / "trace.json", out_dir / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error(f"{out_dir} already holds run artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = dict(submission=None, steps=[])
    summary = dict(scheduler=mode, passed=False, failure=None,
                   command=[sys.executable, *sys.argv],
                   provenance=provenance_record())
    ray_was_initialized = ("ray" in sys.modules
                           and sys.modules["ray"].is_initialized())
    engine = None
    try:
        root_on_path = str(REPO_ROOT) in [
            str(Path(p).resolve()) for p in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        check(root_on_path, f"repository root {REPO_ROOT} must be on "
              "PYTHONPATH so the Ray worker can import the LP modules")
        check(all(os.environ.get(k) == "1" for k in
                  ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
              "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be 1")

        recorded = load_recorded(args.recorded_results)
        summary["recorded"] = recorded
        asset = inspect_snapshot(os.path.abspath(args.model_path))
        weight = asset["weight_files"][0]
        weight["sha256"] = sha256(weight["resolved"])
        check(asset["snapshot_revision"] == EXPECTED_REVISION
              and len(asset["weight_files"]) == 1
              and weight["sha256"] == EXPECTED_WEIGHT_SHA256,
              f"asset identity {asset}")
        summary["asset"] = asset
        reference = None
        if mode == "lp":
            ref_path = out_dir.resolve().parent / "reference" / "summary.json"
            if ref_path.exists():
                ref = json.loads(ref_path.read_text())
                reference = dict(path=str(ref_path), sha256=sha256(ref_path),
                                 passed=ref.get("passed"),
                                 generated_token_ids=ref.get(
                                     "generated_token_ids"))
            summary["reference"] = reference
            validate_run = load_validate_run()
        summary["environment"] = environment_record()
        print("environment:", json.dumps(summary["environment"]), flush=True)

        metrics_dir = out_dir / "metrics_store_unused"
        t0 = time.monotonic()
        engine = (create_vllm_engine(asset["snapshot_path"], metrics_dir)
                  if mode == "vllm"
                  else rli.create_engine(asset["snapshot_path"], metrics_dir))
        summary["engine_init_seconds"] = round(time.monotonic() - t0, 3)
        facts = engine_facts(engine)
        summary["engine_facts"] = facts
        print("engine:", json.dumps(facts), flush=True)
        check(facts["load_format"] == "auto"
              and facts["dtype"] == "torch.float16"
              and facts["max_model_len"] == 32 and facts["seed"] == 42
              and facts["block_size"] == 16
              and facts["gpu_memory_utilization"] == 0.5
              and facts["tensor_parallel_size"] == 1
              and facts["pipeline_parallel_size"] == 1
              and facts["tokenizer_mode"] == "auto"
              and facts["trust_remote_code"] is True
              and facts["model"] == facts["tokenizer"]
              == asset["snapshot_path"]
              and facts["hf_config_commit_hash"] == EXPECTED_REVISION
              and facts["metrics"] == dict(write_metrics=True,
                                           enable_request_outputs=False,
                                           enable_chrome_trace=False),
              f"engine facts {facts}")
        if mode == "vllm":
            check(facts["scheduler_type"] == "VLLMScheduler"
                  and facts["scheduler_config"]["max_num_seqs"] == 1
                  and facts["scheduler_config"]["max_num_batched_tokens"] == 32
                  and facts["scheduler_config"]["num_pipeline_stages"] == 1,
                  "reference scheduler config")
        else:
            import lp_relaxation_scheduler as lrs
            check(facts["scheduler_type"] == "LPScheduler"
                  and {k: facts["scheduler_config"][k] for k in (
                      "max_num_seqs", "b_max", "c_max", "s_max",
                      "memory_reserve", "decode_memory_policy_id",
                      "decode_utility", "prefill_token_utility",
                      "preemption_penalty", "num_pipeline_stages")}
                  == dict(max_num_seqs=3, b_max=96, c_max=8, s_max=2,
                          memory_reserve=1,
                          decode_memory_policy_id="conservative_one_block_v1",
                          decode_utility=1.0, prefill_token_utility=1.0,
                          preemption_penalty=1.0, num_pipeline_stages=1)
                  and facts["numerical_policy"] == to_jsonable(
                      lrs.NumericalPolicy(**rli.NUMERICAL_POLICY_FIELDS)),
                  "LP scheduler config")

        tokenizer = engine.tokenizer
        prompt_ids = list(tokenizer.encode(RECORDED_PROMPT))
        check(prompt_ids == RECORDED_PROMPT_IDS,
              f"native encoding {prompt_ids} differs from the recorded IDs")
        summary["prompt_tokens"] = tokenizer.convert_ids_to_tokens(prompt_ids)

        bm = engine.scheduler.block_manager
        n_free = bm.get_num_free_gpu_blocks()
        check(n_free == bm.num_total_gpu_blocks and not bm.block_tables,
              "block pool not fully free before the request")
        summary["n_free"] = n_free

        seqs = {}
        observer = Observer()
        if mode == "lp":
            import lp_relaxation_scheduler as lrs
            import lpserve_plan_execution as lpe
            import lpserve_state_mapping as lsm
            from check_lp_scheduler_gpu import outputs_record
            policy_json = to_jsonable(lrs.NumericalPolicy(
                **rli.NUMERICAL_POLICY_FIELDS))
            observer.wrap(lsm, "map_scheduler_state", lambda a, k, r: dict(
                result_type=type(r).__name__, result=to_jsonable(r)))
            observer.wrap(lrs, "solve_and_extract", lambda a, k, r: dict(
                problem_id_in=a[0].problem_id, result_type=type(r).__name__,
                result=to_jsonable(r)))
            observer.wrap(lpe, "execute_plan", lambda a, k, r: dict(
                snapshot_id_in=a[1].snapshot_id,
                problem_id_in=a[2].problem_id, result_type=type(r).__name__,
                result=(outputs_record(r)
                        if type(r).__name__ == "SchedulerOutputs"
                        else to_jsonable(r))))
        observer.wrap(engine.scheduler, "schedule", lambda a, k, r: dict(
            outputs=outputs_with_counts(r), state=capture_state(engine, seqs)))
        observer.wrap(engine, "_run_workers", worker_record)

        sampling_params = rli.make_sampling_params()
        observer.events = []
        engine.add_request(RECORDED_PROMPT, sampling_params,
                           prompt_token_ids=list(prompt_ids))
        check(len(engine.seq_manager.seq_map) == 1, "expected one sequence")
        seq_id = next(iter(engine.seq_manager.seq_map))
        seq = seqs["0"] = engine.seq_manager.seq_map[seq_id]
        sp = seq.sampling_params
        sampling = dict(temperature=sp.temperature, max_tokens=sp.max_tokens,
                        ignore_eos=sp.ignore_eos, stop=list(sp.stop),
                        sampling_type=sp.sampling_type.name)
        check(sampling == dict(temperature=0.0, max_tokens=4, ignore_eos=True,
                               stop=[], sampling_type="GREEDY")
              and seq.prompt == RECORDED_PROMPT
              and list(seq.prompt_token_ids) == RECORDED_PROMPT_IDS
              and seq.eos_token_id == tokenizer.eos_token_id,
              f"submitted sequence {sampling}")
        trace["submission"] = dict(
            seq_id=seq_id, prompt=seq.prompt, prompt_token_ids=prompt_ids,
            sampling=sampling, calls=[e["call"] for e in observer.events],
            state=capture_state(engine, seqs))
        summary.update(seq_id=seq_id, sampling=sampling)
        print(f"submitted seq {seq_id}: {prompt_ids}", flush=True)

        ids, prompt_lens = {"0": seq_id}, {"0": len(prompt_ids)}
        max_steps = len(prompt_ids) + rli.MAX_TOKENS
        finished_outputs = []
        while engine.has_unfinished_requests():
            number = len(trace["steps"]) + 1
            check(number <= max_steps, f"unfinished after {max_steps} steps")
            tag = f"step {number}"
            before = capture_state(engine, seqs)
            record = dict(step=number, before=before)
            trace["steps"].append(record)
            events = record["events"] = observer.events = []
            step_outputs = engine.step()
            after = capture_state(engine, seqs)
            outputs = [request_output_record(o) for o in step_outputs]
            record.update(after=after, request_outputs=outputs)
            if mode == "vllm":
                checked = check_vllm_step(tag, number, seq_id,
                                          len(prompt_ids), before, events,
                                          after, outputs)
            else:
                checked = validate_run.check_step(
                    tag, ids, prompt_lens, before, events, after,
                    step_outputs, policy_json)
            record["checked"] = checked
            finished_outputs += [o for o in outputs if o["finished"]]
            print(f"{tag}: emitted {checked['emitted']}, sampler "
                  f"{checked['sampler_outputs']}, free {checked['free']}, "
                  f"prompt {after['requests']['0']['prompt_tokens_processed']}"
                  f", output {after['requests']['0']['output_token_ids']}, "
                  f"finished {checked['finished']}", flush=True)

        final = capture_state(engine, seqs)
        check(len(finished_outputs) == 1, f"finished {finished_outputs}")
        out = finished_outputs[0]
        check(out["seq_id"] == seq_id
              and out["prompt_token_ids"] == RECORDED_PROMPT_IDS
              and out["finish_reason"] == "length"
              and len(out["token_ids"]) == rli.MAX_TOKENS
              and out["token_ids"] == final["requests"]["0"]["output_token_ids"],
              f"finished output {out}")
        check(final["waiting"] == [] and final["running"] == []
              and final["block_tables"] == {} and final["engine_seq_ids"] == []
              and final["num_unfinished"] == 0
              and final["free_blocks"] == n_free, f"final state {final}")
        if mode == "vllm":
            check(len(trace["steps"]) == 1 + rli.MAX_TOKENS,
                  f"{len(trace['steps'])} steps")
        summary.update(steps=len(trace["steps"]), final_state=final,
                       finished_output=out,
                       decisions=[dict(step=s["step"], **s["checked"])
                                  for s in trace["steps"]])

        generated = out["token_ids"]
        replay = replay_detokenization(tokenizer, prompt_ids, generated)
        text_record = dict(
            native_text=out["text"],
            decode=tokenizer.decode(generated),
            decode_skip_special_tokens=tokenizer.decode(
                generated, skip_special_tokens=True),
            generated_tokens=tokenizer.convert_ids_to_tokens(generated),
            prompt_tail_tokens=tokenizer.convert_ids_to_tokens(prompt_ids[-6:]),
            replay=replay,
            replay_equals_native=replay["text"] == out["text"],
            eos_token_id=tokenizer.eos_token_id,
            eos_positions=[i for i, t in enumerate(generated)
                           if t == tokenizer.eos_token_id],
            ignore_eos=sampling["ignore_eos"],
        )
        summary["text"] = text_record
        summary["comparisons"] = dict(
            recorded_lp_three_request=compare_tokens(
                RECORDED_GENERATED_IDS, generated))
        if reference is not None and reference["generated_token_ids"]:
            summary["comparisons"]["reference"] = compare_tokens(
                reference["generated_token_ids"], generated)
        summary["generated_token_ids"] = generated
        summary["passed"] = True
        print("generated:", generated, flush=True)
        print("text:", json.dumps({k: v for k, v in text_record.items()
                                   if k != "replay"}), flush=True)
        print("replayed text:", repr(replay["text"]), flush=True)
        print("comparisons:", json.dumps(summary["comparisons"]), flush=True)
    except BaseException as err:
        summary["failure"] = dict(type=type(err).__name__, message=str(err),
                                  traceback=traceback.format_exc())
        for name in ("stage", "category", "reason"):
            if hasattr(err, name):
                summary["failure"][name] = repr(getattr(err, name))
        print("RUN FAILED:", type(err).__name__, err, file=sys.stderr,
              flush=True)
        traceback.print_exc()
    finally:
        engine = None
        started = (not ray_was_initialized and "ray" in sys.modules
                   and sys.modules["ray"].is_initialized())
        if started:
            sys.modules["ray"].shutdown()
        summary["ray"] = dict(
            started_by_this_run=started,
            initialized_after_shutdown=("ray" in sys.modules and
                                        sys.modules["ray"].is_initialized()))
        trace_path.write_text(json.dumps(trace, indent=1))
        summary_path.write_text(json.dumps(summary, indent=1,
                                           ensure_ascii=False))
    print("RESULT:", "COMPLETED" if summary["passed"] else "FAILED",
          flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
