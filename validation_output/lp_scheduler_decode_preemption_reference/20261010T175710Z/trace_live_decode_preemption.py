"""Run the live CPU decode-preemption test once and retain its trace.

Usage (repository root, root on PYTHONPATH):
    python -B <this file> <output dir>

Writes decision_trace.json (the per-step records the test collects) and
summary.json (provenance, inputs, key witnesses, and the pass flag). No model
is initialized and no CUDA executes."""

import dataclasses
import enum
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "tests"))
import test_lp_scheduler as T  # noqa: E402

FILES = [
    "tests/test_lp_scheduler.py",
    "tests/test_check_lp_scheduler_decode_preemption_reference_gpu.py",
    "scripts/check_lp_scheduler_decode_preemption_reference_gpu.py",
    "sarathi/core/scheduler/lp_scheduler.py",
    "sarathi/core/scheduler/base_scheduler.py",
    "sarathi/core/sequence_manager/base_sequence_manager.py",
    "sarathi/core/sequence_manager/worker_sequence_manager.py",
    "sarathi/core/sequence_manager/engine_sequence_manager.py",
    "sarathi/core/datatypes/sequence.py",
    "sarathi/core/block_space_manager/base_block_space_manager.py",
    "sarathi/core/block_space_manager/vllm_block_space_manager.py",
    "lp_relaxation_scheduler.py", "lpserve_state_mapping.py",
    "lpserve_plan_execution.py",
]


def jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, (frozenset, set)):
        return sorted((jsonable(x) for x in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=REPO_ROOT).stdout


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out_dir):
    out = Path(out_dir)
    T.setUpModule()
    case = T.LiveDecodePreemptionTest(
        "test_decode_preemption_recomputes_expanded_context_to_completion")
    result = unittest.TestResult()
    case.run(result)
    T.tearDownModule()
    passed = result.wasSuccessful() and result.testsRun == 1
    trace = jsonable(getattr(case, "trace", None))
    (out / "decision_trace.json").write_text(json.dumps(trace, indent=1))

    boundary = trace[case.BOUNDARY_STEP]
    readmission = trace[case.READMISSION_STEP]
    summary = dict(
        passed=passed,
        failures=[f[1] for f in result.failures + result.errors],
        provenance=dict(
            git_head=git("rev-parse", "HEAD").strip(),
            git_status_short=git("status", "--short", "--untracked-files=all"),
            git_status_ignored_tests=git("status", "--short", "--ignored",
                                         "--", "tests"),
            git_diff_head_sha256=hashlib.sha256(
                git("diff", "HEAD").encode()).hexdigest(),
            file_sha256={f: sha256(REPO_ROOT / f) for f in FILES},
            trace_script_sha256=sha256(__file__),
        ),
        inputs=dict(
            pool=case.POOL, block_size=T.BLOCK_SIZE, config=case.CONFIG,
            utilities=jsonable(case.UTILITIES),
            numerical_policy=jsonable(T.NUMERICAL_POLICY),
            decode_policy_id=T.DECODE_POLICY_ID,
            prompts=dict(A=case.PROMPT_A, B=case.PROMPT_B),
            max_tokens=jsonable(case.MAX_TOKENS),
            synthetic_decode_tokens=jsonable(case.DECODE_TOKENS),
            synthetic_prefill_sample=case.PREFILL_SAMPLE,
        ),
        emitted=[dict(step=r["step"], **r["outputs"]) for r in trace[:-1]],
        boundary=dict(
            legal_preemption_ids=boundary["snapshot"]["lp_problem"]
            ["legal_preemption_ids"],
            m_free=boundary["snapshot"]["lp_problem"]["m_free"],
            w=boundary["snapshot"]["lp_problem"]["w"],
            requests=boundary["snapshot"]["requests"],
            relaxed=boundary["result"]["relaxed"],
            plan=boundary["result"]["plan"],
            central_ops=boundary["central_ops"],
            after_execution=boundary["after_execution"],
            replay_log=boundary["replay_log"],
            after_replay=boundary["after"],
        ),
        readmission=dict(
            mapped_a=[r for r in readmission["snapshot"]["requests"]
                      if r["raw_seq_id"] == 0],
            central_ops=readmission["central_ops"],
            after=readmission["after"],
        ),
        decode_events=jsonable(getattr(case, "decode_events", None)),
        idle=trace[-1],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("tests run:", result.testsRun, "failures:", len(result.failures),
          "errors:", len(result.errors))
    print("emitted:", json.dumps(summary["emitted"]))
    print("decode events (step, seq, token):", summary["decode_events"])
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
