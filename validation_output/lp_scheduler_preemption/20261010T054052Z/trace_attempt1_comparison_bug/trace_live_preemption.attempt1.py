"""Record the live CPU preemption workload as a decision/state trace.

Reuses the test harness from ``tests/test_lp_scheduler.py`` (real registered
``LPScheduler``, real mapper/solver/extraction/executor, central replay on
shared sequences, worker replay on separate copies, synthetic sampler, no-op
text conversion) with the same provisional inputs as
``LivePreemptionTest``. It records what happened; the assertions live in the
test. Usage, from the repository root with the environment active:

    python -B <this file> <output directory>
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "tests"))

import scipy  # noqa: E402

import test_lp_scheduler as t  # noqa: E402

OUT = Path(sys.argv[1])
CASE = t.LivePreemptionTest
TESTED_FILES = [
    "lpserve_plan_execution.py",
    "lpserve_state_mapping.py",
    "lp_relaxation_scheduler.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "tests/test_lpserve_plan_execution.py",
    "tests/test_lp_scheduler.py",
    "docs/lp_scheduler_design.md",
    str(Path(__file__).resolve().relative_to(REPO)),
]


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout


def _sha256(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def _seq_state(seq):
    if seq is None:
        return None
    return {
        "status": seq.get_status().name,
        "prompt_token_ids": list(seq.prompt_token_ids),
        "prompt_tokens_processed": seq.get_num_prompt_tokens_processed(),
        "prompt_processing_finished": seq.prompt_processing_finished,
        "output_token_ids": list(seq.output_token_ids),
        "logical_blocks": len(seq.logical_token_blocks),
    }


def _state(harness):
    s = harness.scheduler
    ids = sorted(set(harness.engine_seqs.seq_map) | set(harness.worker_seqs.seq_map))
    return {
        "waiting": [q.seq_id for q in s.waiting],
        "running": [q.seq_id for q in s.running],
        "central_tables": {str(k): v for k, v in harness.central_tables().items()},
        "worker_tables": {str(k): v for k, v in harness.worker_tables().items()},
        "central_free": s.block_manager.get_num_free_gpu_blocks(),
        "worker_free": harness.worker_seqs.block_manager.get_num_free_gpu_blocks(),
        "central_seqs": {str(k): _seq_state(harness.engine_seqs.get_seq(k)) for k in ids},
        "worker_seqs": {str(k): _seq_state(harness.worker_seqs.get_seq(k)) for k in ids},
    }


def _decision(spies):
    record = {"calls": spies.names()}
    if len(spies.calls) < 3:
        return record
    snapshot, result = spies.calls[1][4], spies.calls[2][2]
    problem = snapshot.lp_problem
    record["snapshot"] = {
        "snapshot_id": snapshot.snapshot_id,
        "m_free": problem.m_free, "w": problem.w,
        "b_max": problem.b_max, "c_max": problem.c_max, "s_max": problem.s_max,
        "legal_preemption_ids": sorted(problem.legal_preemption_ids),
        "requests": [
            {
                "request_id": r.request_id, "ownership": r.ownership,
                "status": r.status, "prompt_len": r.prompt_len,
                "prompt_tokens_processed": r.prompt_tokens_processed,
                "physical_block_count": r.physical_block_count,
                "prefill_fixed_charge": r.prefill_fixed_charge,
                "decode_charge": r.decode_charge,
                "preemption_eligible": r.preemption_eligible,
                "preemption_recovery": r.preemption_recovery,
            }
            for r in snapshot.requests
        ],
    }
    record["relaxed"] = {
        "decisions": [
            {"request_id": d.request_id, "x": d.x, "y": d.y,
             "prefill_indicator": d.prefill_indicator, "z": d.z}
            for d in result.relaxed.decisions
        ],
        "normalized_objective": result.relaxed.normalized_objective,
        "projection_count": result.relaxed.projection_count,
    }
    plan = result.plan
    record["plan"] = {
        "decisions": [
            {"request_id": d.request_id, "prefill_tokens": d.prefill_tokens,
             "decode": d.decode, "preempt": d.preempt}
            for d in plan.decisions
        ],
        "dominant_preemption_ids": list(plan.dominant_preemption_ids),
        "safety_preemption_ids": list(plan.safety_preemption_ids),
        "fractional_request_count": plan.fractional_request_count,
        "objective": plan.objective,
    }
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t.setUpModule()
    harness = t._Harness(num_gpu_blocks=CASE.POOL, **CASE.CONFIG)
    scheduler = harness.scheduler
    events = []

    def submit(seq_id, prompt, arrival):
        harness.add(seq_id, len(prompt), arrival, max_tokens=1)
        events.append({"event": "submit", "seq_id": seq_id,
                       "prompt_token_ids": prompt, "arrival_time": arrival,
                       "max_tokens": 1, "ignore_eos": True})

    worker_manager = harness.worker_seqs.block_manager
    real_free, real_allocate = worker_manager.free, worker_manager.allocate
    worker_calls = []

    def worker_free(seq):
        worker_calls.append(["free", seq.seq_id])
        real_free(seq)

    def worker_allocate(seq):
        worker_calls.append(["allocate", seq.seq_id])
        real_allocate(seq)

    worker_manager.free, worker_manager.allocate = worker_free, worker_allocate

    submit(0, CASE.PROMPT_A, 1.0)
    clock = [1.0, 2.0] + [3.0 + i for i in range(20)]
    step = 0
    while scheduler.has_unfinished_seqs():
        if step >= 21:
            raise RuntimeError("step limit reached without completion")
        if step == 1:
            submit(1, CASE.PROMPT_B, 2.0)
        now = clock[step]
        before = _state(harness)
        harness.clock.reset_mock()
        with t._Spies() as spies:
            outputs = harness.schedule(now)
        record = {"step": step, "now": now, "iteration_id": outputs.id,
                  "before": before, **_decision(spies)}
        record["outputs"] = {
            "preempted_seq_ids": list(outputs.preempted_seq_ids),
            "ignored_seq_ids": list(outputs.ignored_seq_ids),
            "scheduled": t._emitted(outputs),
        }
        record["after_execution"] = _state(harness)
        del worker_calls[:]
        harness.replay(outputs)
        record["worker_block_calls_during_replay"] = list(worker_calls)
        record["after_replay"] = _state(harness)
        events.append(record)
        step += 1

    with t._Spies() as spies:
        idle = harness.schedule(100.0)
    events.append({"event": "idle", "now": 100.0, "iteration_id": idle.id,
                   "calls": spies.names(), "has_no_output": idle.has_no_output(),
                   "after": _state(harness)})

    decisions = [e for e in events if "step" in e]
    boundary = decisions[1]
    relaxed = {d["request_id"]: d for d in boundary["relaxed"]["decisions"]}
    tol = 1e-7
    witnesses = {
        "boundary_relaxed_a_z": relaxed["0"]["z"],
        "boundary_relaxed_b": [relaxed["1"][k] for k in ("x", "y", "prefill_indicator", "z")],
        "boundary_objective": boundary["relaxed"]["normalized_objective"],
        "boundary_plan": boundary["plan"],
        "boundary_outputs": boundary["outputs"],
        "boundary_worker_block_calls": boundary["worker_block_calls_during_replay"],
        "scheduled_by_step": [e["outputs"]["scheduled"] for e in decisions],
        "preempted_by_step": [e["outputs"]["preempted_seq_ids"] for e in decisions],
    }
    a_after_boundary = boundary["after_replay"]["central_seqs"]["0"]
    checks = {
        "boundary_prediction": (
            abs(relaxed["0"]["z"] - 0.5) <= tol
            and all(abs(relaxed["0"][k]) <= tol for k in ("x", "y", "prefill_indicator"))
            and abs(relaxed["1"]["x"] - 4) <= tol
            and abs(relaxed["1"]["prefill_indicator"] - 1) <= tol
            and abs(relaxed["1"]["z"]) <= tol and abs(relaxed["1"]["y"]) <= tol
            and abs(boundary["relaxed"]["normalized_objective"] - 3.5) <= tol
        ),
        "boundary_extraction": (
            boundary["plan"]["dominant_preemption_ids"] == ["0"]
            and boundary["outputs"]["preempted_seq_ids"] == [0]
            and boundary["outputs"]["scheduled"] == [[1, 4]]
        ),
        "worker_free_before_allocate": boundary["worker_block_calls_during_replay"]
        == [["free", 0], ["allocate", 1]],
        "central_reset_for_recompute": (
            a_after_boundary["status"] == "WAITING"
            and a_after_boundary["prompt_tokens_processed"] == 0
        ),
        "all_finished_and_drained": (
            events[-1]["after"]["waiting"] == [] and events[-1]["after"]["running"] == []
            and events[-1]["after"]["central_tables"] == {}
            and events[-1]["after"]["worker_tables"] == {}
            and events[-1]["after"]["central_free"] == CASE.POOL
            and events[-1]["after"]["worker_free"] == CASE.POOL
        ),
        "final_idle_no_pipeline_calls": events[-1]["calls"] == [] and events[-1]["has_no_output"],
    }

    (OUT / "decision_trace.json").write_text(json.dumps(events, indent=1) + "\n")
    summary = {
        "provenance": {
            "head": _git("rev-parse", "HEAD").strip(),
            "branch": _git("branch", "--show-current").strip(),
            "git_status_short": _git("status", "--short", "--untracked-files=all").splitlines(),
            "tracked_diff_sha256": hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest(),
            "file_sha256": {p: _sha256(p) for p in TESTED_FILES},
        },
        "environment": {
            "host": socket.getfqdn(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unset"),
            "loaded_modules": os.environ.get("LOADEDMODULES", ""),
            "python": sys.executable, "python_version": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "inputs": {
            "block_size": t.BLOCK_SIZE, "num_gpu_blocks_central_and_worker": CASE.POOL,
            "max_model_len": t.MAX_MODEL_LEN, **CASE.CONFIG,
            "decode_memory_policy_id": t.DECODE_POLICY_ID,
            "numerical_policy": t.NUMERICAL_POLICY.__dict__,
            "sampled_token": t.SAMPLED_TOKEN,
            "requests": [e for e in events if e.get("event") == "submit"],
        },
        "witnesses": witnesses,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"checks": checks, "passed": summary["passed"],
                      "scheduled_by_step": witnesses["scheduled_by_step"],
                      "preempted_by_step": witnesses["preempted_by_step"]}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
