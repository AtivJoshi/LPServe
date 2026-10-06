"""Validation driver for one run of scripts/run_lp_inference.py.

Calls ``run_lp_inference.main`` with the given arguments, unchanged. Before
that, it replaces ``run_lp_inference.create_engine`` with a wrapper that
calls the original and then installs read-only observers (the existing
``Observer`` helper) on the LP mapping, solve, and execution functions and on
the engine's ``add_request``, ``step``, ``scheduler.schedule``, and
``_run_workers``. Observers delegate exactly once and return results
unchanged. A failed per-step check raises inside ``step`` so the run stops.

Validation machinery only; not part of the reusable program.

    PYTHONPATH="$PWD" timeout 300s python -B <this file> \\
        --model-path ... --prompts-file ... --output-dir <run dir> \\
        --driver-dir <evidence dir>
"""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_lp_inference as rli  # noqa: E402
from check_lp_scheduler_gpu import (Observer, check, outputs_record,  # noqa: E402
                                    request_output_record, to_jsonable)
from check_lp_scheduler_contention_gpu import (capture_state,  # noqa: E402
                                               outputs_with_counts,
                                               worker_record)


def remaining_work(state, prompt_lens):
    return sum(prompt_lens[n] - r["prompt_tokens_processed"]
               + rli.MAX_TOKENS - r["generated"]
               for n, r in state["requests"].items())


def check_step(tag, ids, prompt_lens, before, events, after, step_outputs,
               policy_json):
    """Per-step checks derived from the observed pre-step state."""
    names = {s: n for n, s in ids.items()}
    unfinished = sorted(ids[n] for n, r in before["requests"].items()
                        if not r["is_finished"])
    check(before["num_pipeline_stages"] == 1
          and before["num_running_batches"] == 0, f"{tag}: not quiescent")
    check(sorted(before["waiting"] + before["running"]) == unfinished
          and before["engine_seq_ids"] == unfinished
          and before["num_unfinished"] == len(unfinished),
          f"{tag}: ownership {before}")
    for s in before["waiting"]:
        r = before["requests"][names[s]]
        check(r["status"] == "WAITING" and r["block_ids"] is None,
              f"{tag}: waiting {s} state {r}")
    for s in before["running"]:
        r = before["requests"][names[s]]
        check(r["status"] == "PAUSED" and r["block_ids"],
              f"{tag}: resident {s} state {r}")

    calls = [e["call"] for e in events]
    check(calls == ["map_scheduler_state", "solve_and_extract",
                    "execute_plan", "schedule", "_run_workers"],
          f"{tag}: call sequence {calls}")
    mapped, solved, executed, scheduled, worker = events
    check(worker["method"] == "execute_model", f"{tag}: worker call")

    check(mapped["result_type"] == "StateSnapshot",
          f"{tag}: mapping {mapped['result_type']}: {mapped['result']}")
    snap = mapped["result"]
    problem = snap["lp_problem"]
    check((problem["b_max"], problem["c_max"], problem["s_max"])
          == (rli.B_MAX, rli.C_MAX, rli.S_MAX)
          and snap["memory_reserve"] == rli.MEMORY_RESERVE
          and snap["resident_limit"] == rli.MAX_NUM_SEQS
          and snap["decode_memory_policy_id"] == rli.DECODE_POLICY_ID
          and snap["numerical_policy"] == policy_json
          and snap["free_physical_blocks"] == before["free_blocks"]
          and problem["m_free"] == before["free_blocks"],
          f"{tag}: mapped inputs")
    check(sorted(r["raw_seq_id"] for r in snap["requests"]) == unfinished,
          f"{tag}: mapped request set")
    for r in snap["requests"]:
        check(r["utility"] == dict(
            decode_utility=rli.DECODE_UTILITY,
            prefill_token_utility=rli.PREFILL_TOKEN_UTILITY,
            preemption_penalty=rli.PREEMPTION_PENALTY), f"{tag}: utilities")
    raw = {r["request_id"]: r["raw_seq_id"] for r in snap["requests"]}

    check(solved["result_type"] == "SchedulingSuccess",
          f"{tag}: solve {solved['result_type']}: {solved['result']}")
    plan = solved["result"]["plan"]
    pid = problem["problem_id"]
    check(solved["problem_id_in"] == pid == solved["result"]["problem_id"]
          == plan["problem_id"] == executed["problem_id_in"]
          and executed["snapshot_id_in"] == snap["snapshot_id"],
          f"{tag}: identities differ")
    check(not plan["dominant_preemption_ids"]
          and not plan["safety_preemption_ids"], f"{tag}: plan preemption")
    selected = []
    for d in plan["decisions"]:
        check(d["preempt"] == 0
              and not (d["prefill_tokens"] > 0 and d["decode"]),
              f"{tag}: decision {d}")
        if d["prefill_tokens"] > 0 or d["decode"]:
            selected.append(d)
    selected.sort(key=lambda d: (0 if d["prefill_tokens"] > 0 else 1,
                                 tuple(d["order_key"])))
    want = [[raw[d["request_id"]], d["prefill_tokens"]] for d in selected]

    check(executed["result_type"] == "SchedulerOutputs",
          f"{tag}: executor {executed['result_type']}")
    emitted = scheduled["outputs"]
    check(executed["result"] == {k: emitted[k] for k in (
        "id", "scheduled", "ignored_seq_ids", "preempted_seq_ids")},
        f"{tag}: scheduler output differs from executor output")
    check(emitted["scheduled"] == want,
          f"{tag}: emitted {emitted['scheduled']}, plan {want}")
    check(not emitted["ignored_seq_ids"] and not emitted["preempted_seq_ids"],
          f"{tag}: controls emitted")
    sel = [s for s, _ in emitted["scheduled"]]
    check(len(sel) == len(set(sel)) and 1 <= len(sel) <= rli.S_MAX,
          f"{tag}: selected {sel}")
    chunks = dict(emitted["scheduled"])
    prefill_ids = {s for s, c in chunks.items() if c > 0}
    decode_ids = {s for s, c in chunks.items() if c == 0}
    for s, c in chunks.items():
        r = before["requests"][names[s]]
        rem = prompt_lens[names[s]] - r["prompt_tokens_processed"]
        if c > 0:
            check(c <= min(rem, rli.C_MAX), f"{tag}: chunk {c} for {s}")
        else:
            check(rem == 0 and r["prompt_processing_finished"]
                  and s in before["running"], f"{tag}: decode for {s}: {r}")
    tokens = sum(chunks.values()) + len(decode_ids)
    check(emitted["num_batched_prompt_tokens"] == sum(chunks.values())
          and emitted["num_batched_output_tokens"] == len(decode_ids)
          and tokens <= rli.B_MAX, f"{tag}: counts {emitted}")
    check([s for s, _ in worker["sampler_outputs"]] == sel,
          f"{tag}: sampler ids {worker['sampler_outputs']}")

    sched = scheduled["state"]
    check(sched["num_running_batches"] == 1
          and len(sched["running"]) <= rli.MAX_NUM_SEQS,
          f"{tag}: state after scheduling")
    expected_free = before["free_blocks"]
    for s in unfinished:
        n = names[s]
        b, m = before["requests"][n], sched["requests"][n]
        check((m["prompt_tokens_processed"], m["generated"])
              == (b["prompt_tokens_processed"], b["generated"]),
              f"{tag}: {n} progressed during scheduling")
        if s in prefill_ids and s in before["waiting"]:
            check(s in sched["running"]
                  and m["physical_blocks"] == m["logical_blocks"],
                  f"{tag}: {n} admission {m}")
            expected_free -= m["physical_blocks"]
        elif s in decode_ids:
            gap = b["logical_blocks"] - b["physical_blocks"]
            check(gap in (0, 1)
                  and m["block_ids"][:b["physical_blocks"]] == b["block_ids"]
                  and m["physical_blocks"] == b["physical_blocks"] + gap,
                  f"{tag}: {n} decode append {b} -> {m}")
            expected_free -= gap
        else:
            check(m["block_ids"] == b["block_ids"],
                  f"{tag}: {n} blocks changed")
    check(sched["free_blocks"] == expected_free,
          f"{tag}: free after scheduling {sched['free_blocks']}, "
          f"expected {expected_free}")

    check(after["iteration_id"] == before["iteration_id"] + 1
          and after["num_running_batches"] == 0, f"{tag}: after state")
    freed, finished_now = 0, []
    for s in unfinished:
        n = names[s]
        b, a = before["requests"][n], after["requests"][n]
        if s in prefill_ids:
            done = b["prompt_tokens_processed"] + chunks[s]
            check(a["prompt_tokens_processed"] == done
                  and a["prompt_processing_finished"]
                  == (done == prompt_lens[n])
                  and a["output_token_ids"] == b["output_token_ids"],
                  f"{tag}: {n} prefill completion {b} -> {a}")
        elif s in decode_ids:
            check(a["prompt_tokens_processed"] == b["prompt_tokens_processed"]
                  and a["generated"] == b["generated"] + 1
                  and a["output_token_ids"][:-1] == b["output_token_ids"],
                  f"{tag}: {n} decode completion {b} -> {a}")
        else:
            keys = ("status", "prompt_tokens_processed",
                    "prompt_processing_finished", "output_token_ids",
                    "generated", "logical_blocks", "block_ids")
            check(all(a[k] == b[k] for k in keys)
                  and (s in after["waiting"]) == (s in before["waiting"])
                  and (s in after["running"]) == (s in before["running"]),
                  f"{tag}: unselected {n} changed {b} -> {a}")
        if a["is_finished"]:
            check(s in decode_ids and a["status"] == "FINISHED_LENGTH_CAPPED"
                  and a["generated"] == rli.MAX_TOKENS
                  and a["block_ids"] is None
                  and s not in after["waiting"] + after["running"]
                  + after["engine_seq_ids"], f"{tag}: {n} finish {a}")
            freed += sched["requests"][n]["physical_blocks"]
            finished_now.append(s)
    check(after["free_blocks"] == sched["free_blocks"] + freed,
          f"{tag}: free after completion {after['free_blocks']}")
    reduced = (remaining_work(before, prompt_lens)
               - remaining_work(after, prompt_lens))
    check(reduced == tokens >= 1, f"{tag}: work reduced by {reduced}")
    check([(o.seq_id, len(o.token_ids), o.finished) for o in step_outputs]
          == [(s, after["requests"][names[s]]["generated"],
               s in finished_now) for s in sel],
          f"{tag}: request outputs")
    for o in step_outputs:
        check(not o.finished or o.finish_reason == "length",
              f"{tag}: finish reason {o.finish_reason}")
    return dict(
        emitted=emitted["scheduled"],
        selected=[names[s] for s in sel],
        omitted=[names[s] for s in unfinished if s not in sel],
        plan={names[raw[d["request_id"]]]: [d["prefill_tokens"], d["decode"],
                                            d["preempt"]]
              for d in plan["decisions"]},
        sampler_outputs=worker["sampler_outputs"],
        free=dict(before=before["free_blocks"], sched=sched["free_blocks"],
                  after=after["free_blocks"]),
        work=dict(before=remaining_work(before, prompt_lens),
                  after=remaining_work(after, prompt_lens)),
        finished=[names[s] for s in finished_now],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--driver-dir", required=True)
    args = parser.parse_args()
    driver_dir = Path(args.driver_dir)
    trace_path = driver_dir / "decision_trace.json"
    summary_path = driver_dir / "driver_summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error("driver artifacts already exist")
    main_argv = ["--model-path", args.model_path,
                 "--prompts-file", args.prompts_file,
                 "--output-dir", args.output_dir]

    import lp_relaxation_scheduler as lrs
    import lpserve_plan_execution as lpe
    import lpserve_state_mapping as lsm
    from sarathi.core.scheduler.lp_scheduler import LPScheduler

    policy_json = to_jsonable(lrs.NumericalPolicy(
        **rli.NUMERICAL_POLICY_FIELDS))
    trace = dict(add_requests=[], steps=[])
    obs = dict(engine_facts=None, n_free=None, ids={}, prompt_lens={},
               sampling={}, finished_outputs=[], final_state=None)
    seqs = {}
    observer = Observer()
    observer.wrap(lsm, "map_scheduler_state", lambda a, k, r: dict(
        result_type=type(r).__name__, result=to_jsonable(r)))
    observer.wrap(lrs, "solve_and_extract", lambda a, k, r: dict(
        problem_id_in=a[0].problem_id, result_type=type(r).__name__,
        result=to_jsonable(r)))
    observer.wrap(lpe, "execute_plan", lambda a, k, r: dict(
        snapshot_id_in=a[1].snapshot_id, problem_id_in=a[2].problem_id,
        result_type=type(r).__name__,
        result=(outputs_record(r) if type(r).__name__ == "SchedulerOutputs"
                else to_jsonable(r))))

    original_create = rli.create_engine

    def observed_create_engine(model_path, metrics_dir):
        engine = original_create(model_path, metrics_dir)
        sched, bm = engine.scheduler, engine.scheduler.block_manager
        mc = engine.model_config
        obs["engine_facts"] = dict(
            scheduler_type=type(sched).__name__,
            scheduler_is_lp=type(sched) is LPScheduler,
            model=mc.model, tokenizer=mc.tokenizer,
            load_format=mc.load_format, dtype=str(mc.dtype),
            max_model_len=mc.max_model_len, seed=mc.seed,
            attention_backend=str(mc.attention_backend),
            tokenizer_mode=mc.tokenizer_mode,
            trust_remote_code=mc.trust_remote_code,
            hf_config_commit_hash=getattr(mc.hf_config, "_commit_hash", None),
            block_size=engine.cache_config.block_size,
            gpu_memory_utilization=engine.cache_config.gpu_memory_utilization,
            tensor_parallel_size=engine.parallel_config.tensor_parallel_size,
            pipeline_parallel_size=(
                engine.parallel_config.pipeline_parallel_size),
            scheduler_config={k: getattr(engine.scheduler_config, k) for k in (
                "max_num_seqs", "b_max", "c_max", "s_max", "memory_reserve",
                "decode_memory_policy_id", "decode_utility",
                "prefill_token_utility", "preemption_penalty",
                "num_pipeline_stages")},
            numerical_policy=to_jsonable(
                engine.scheduler_config.numerical_policy),
            metrics=dict(
                write_metrics=engine.metrics_config.write_metrics,
                enable_request_outputs=(
                    engine.metrics_config.enable_request_outputs),
                enable_chrome_trace=engine.metrics_config.enable_chrome_trace),
            num_total_gpu_blocks=bm.num_total_gpu_blocks,
            watermark_blocks=bm.watermark_blocks,
        )
        obs["n_free"] = bm.get_num_free_gpu_blocks()
        check(obs["n_free"] == bm.num_total_gpu_blocks and not bm.block_tables,
              "pool not fully free before requests")
        observer.wrap(sched, "schedule", lambda a, k, r: dict(
            outputs=outputs_with_counts(r), state=capture_state(engine, seqs)))
        observer.wrap(engine, "_run_workers", worker_record)

        original_add = engine.add_request
        original_step = engine.step

        def add_request(prompt, sampling_params, *a, **k):
            check(not a and not k, f"unexpected add_request args {a} {k}")
            known = set(engine.seq_manager.seq_map)
            observer.events = []
            original_add(prompt, sampling_params)
            new = set(engine.seq_manager.seq_map) - known
            check(len(new) == 1, f"add_request created {new}")
            s = new.pop()
            name = str(len(seqs))
            seq = engine.seq_manager.seq_map[s]
            seqs[name] = seq
            obs["ids"][name] = s
            obs["prompt_lens"][name] = len(seq.prompt_token_ids)
            sp = seq.sampling_params
            obs["sampling"][name] = dict(
                temperature=sp.temperature, max_tokens=sp.max_tokens,
                ignore_eos=sp.ignore_eos, stop=list(sp.stop),
                sampling_type=sp.sampling_type.name)
            trace["add_requests"].append(dict(
                index=name, seq_id=s, prompt=seq.prompt,
                prompt_token_ids=list(seq.prompt_token_ids),
                tokens=engine.tokenizer.convert_ids_to_tokens(
                    list(seq.prompt_token_ids)),
                calls=[e["call"] for e in observer.events],
                state=capture_state(engine, seqs)))

        def step():
            tag = f"step {len(trace['steps']) + 1}"
            check(len(seqs) == len(json.loads(Path(
                args.prompts_file).read_text())),
                f"{tag}: stepped before all requests were added")
            before = capture_state(engine, seqs)
            record = dict(step=len(trace["steps"]) + 1, before=before)
            trace["steps"].append(record)
            events = record["events"] = observer.events = []
            outputs = original_step()
            after = capture_state(engine, seqs)
            record.update(after=after, request_outputs=[
                request_output_record(o) for o in outputs])
            record["checked"] = check_step(
                tag, obs["ids"], obs["prompt_lens"], before, events, after,
                outputs, policy_json)
            obs["finished_outputs"].extend(
                request_output_record(o) for o in outputs if o.finished)
            obs["final_state"] = after
            print(f"{tag}: emitted {record['checked']['emitted']}, omitted "
                  f"{record['checked']['omitted']}, sampler "
                  f"{record['checked']['sampler_outputs']}, free "
                  f"{record['checked']['free']}, work "
                  f"{record['checked']['work']}, finished "
                  f"{record['checked']['finished']}", flush=True)
            return outputs

        engine.add_request = add_request
        engine.step = step
        return engine

    rli.create_engine = observed_create_engine
    driver = dict(main_argv=main_argv, driver_argv=[sys.executable,
                                                   *sys.argv],
                  driver_sha256=hashlib.sha256(
                      Path(__file__).read_bytes()).hexdigest(),
                  program_exit_status=None, checks_passed=False,
                  failure=None)
    try:
        status = rli.main(main_argv)
        driver["program_exit_status"] = status
        check(status == 0, f"program exit status {status}")
        out = Path(args.output_dir)
        results = json.loads((out / rli.RESULTS_FILE).read_text())
        summary = json.loads((out / rli.SUMMARY_FILE).read_text())
        prompts = json.loads(Path(args.prompts_file).read_text())
        facts = obs["engine_facts"]

        # Configuration actually used.
        check(facts["scheduler_is_lp"], "scheduler is not LPScheduler")
        check(facts["load_format"] == "auto" and facts["dtype"]
              == "torch.float16" and facts["max_model_len"] == 32
              and facts["seed"] == 42 and facts["block_size"] == 16
              and facts["gpu_memory_utilization"] == 0.5
              and facts["tensor_parallel_size"] == 1
              and facts["pipeline_parallel_size"] == 1
              and facts["tokenizer_mode"] == "auto"
              and facts["trust_remote_code"] is True
              and facts["model"] == facts["tokenizer"] == summary["model_path"]
              and facts["hf_config_commit_hash"]
              == "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
              f"engine facts {facts}")
        check(facts["scheduler_config"] == dict(
            max_num_seqs=3, b_max=96, c_max=8, s_max=2, memory_reserve=1,
            decode_memory_policy_id="conservative_one_block_v1",
            decode_utility=1.0, prefill_token_utility=1.0,
            preemption_penalty=1.0, num_pipeline_stages=1)
            and facts["numerical_policy"] == policy_json
            and policy_json["policy_id"] == "lp_relaxation_mvp_v1",
            "scheduler config")
        check(facts["metrics"] == dict(write_metrics=True,
                                       enable_request_outputs=False,
                                       enable_chrome_trace=False),
              "metrics config")
        check(all(v == dict(temperature=0.0, max_tokens=4, ignore_eos=True,
                            stop=[], sampling_type="GREEDY")
                  for v in obs["sampling"].values())
              and len(obs["sampling"]) == len(prompts), "sampling params")
        check(summary["settings"] == json.loads(json.dumps(rli.SETTINGS))
              and summary["success"] is True, "summary settings/success")

        # Association and saved answers.
        native = {o["seq_id"]: o for o in obs["finished_outputs"]}
        check(len(obs["finished_outputs"]) == len(prompts) == len(native),
              "each request must finish exactly once")
        check([r["index"] for r in results] == list(range(len(prompts))),
              "results not in input order")
        for r in results:
            i = r["index"]
            add = trace["add_requests"][i]
            o = native[r["seq_id"]]
            check(r["seq_id"] == obs["ids"][str(i)] == add["seq_id"]
                  and summary["seq_ids"][str(i)] == r["seq_id"]
                  and r["prompt"] == prompts[i] == add["prompt"]
                  and r["prompt_token_ids"] == add["prompt_token_ids"]
                  == o["prompt_token_ids"]
                  and r["generated_text"] == o["text"]
                  and r["generated_token_ids"] == o["token_ids"]
                  and r["finish_reason"] == o["finish_reason"] == "length"
                  and len(r["generated_token_ids"]) == rli.MAX_TOKENS,
                  f"result {i} association")

        # Final drained state.
        final = obs["final_state"]
        check(final["waiting"] == [] and final["running"] == []
              and final["block_tables"] == {}
              and final["engine_seq_ids"] == []
              and final["num_unfinished"] == 0
              and final["free_blocks"] == obs["n_free"],
              f"final state {final}")
        check(summary["ray"] == dict(started_by_this_run=True,
                                     initialized_after_shutdown=False),
              f"ray record {summary['ray']}")
        check(not ("ray" in sys.modules
                   and sys.modules["ray"].is_initialized()),
              "Ray still initialized")
        driver["checks_passed"] = True
    except BaseException as err:
        driver["failure"] = dict(type=type(err).__name__, message=str(err),
                                 traceback=traceback.format_exc())
        print("DRIVER CHECK FAILED:", type(err).__name__, err,
              file=sys.stderr, flush=True)
    finally:
        driver.update(
            engine_facts=obs["engine_facts"], n_free=obs["n_free"],
            seq_ids=obs["ids"], prompt_lens=obs["prompt_lens"],
            sampling=obs["sampling"],
            native_finished_outputs=obs["finished_outputs"],
            nonempty_steps=len(trace["steps"]),
            decisions=[dict(step=s["step"], **s.get("checked", {}))
                       for s in trace["steps"]],
            final_state=obs["final_state"])
        trace_path.write_text(json.dumps(trace, indent=1))
        summary_path.write_text(json.dumps(driver, indent=1))
    print("DRIVER RESULT:", "PASS" if driver["checks_passed"] else "FAIL",
          flush=True)
    return 0 if driver["checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
