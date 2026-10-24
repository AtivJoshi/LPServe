#!/bin/bash
# Context-192 correction of queue48_burst_pool12: CPU gate, corrected CPU
# rerun of that case only, and the runner failure-propagation check, before
# the GPU confirmation. Run from the repository root on the allocated node:
#   bash <run>/cpu_checks/run_checks.sh <run>
set -uo pipefail
RUN="$1"
OUT="$RUN/cpu_checks"
BASE=7cdf3868565ad608be361009b8e8d35860b46795
PRIOR=validation_output/scheduler_comparison/20261024T210104Z_queue_growth/cpu_comparison
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
CHANGED="scripts/compare_scheduler_gpu.py scripts/compare_coefficient_policies_cpu.py tests/test_compare_scheduler_gpu.py"
HELPERS="lpserve_state_mapping.py scripts/exact_integer_plan.py lp_relaxation_scheduler.py lpserve_plan_execution.py sarathi/core/scheduler/lp_scheduler.py sarathi/core/scheduler/adapted_vllm_scheduler.py sarathi/engine/base_llm_engine.py scripts/check_adapted_vllm_gpu.py scripts/check_lp_scheduler_gpu.py scripts/check_lp_scheduler_preemption_gpu.py"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-} slurm_nodelist=${SLURM_JOB_NODELIST:-} partition=${SLURM_JOB_PARTITION:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} (hidden from the checks; CPU only)"
  echo "virtual_env=${VIRTUAL_ENV:-}"
  echo "which_python=$(which python)"
  python --version 2>&1
  python -c 'import numpy, scipy; print("numpy", numpy.__version__); print("scipy", scipy.__version__)'
  module list 2>&1
} > "$OUT/environment.txt" 2>&1
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "base_reviewed=$BASE"
  echo "--- git status --short --untracked-files=all"
  git status --short --untracked-files=all
  echo "--- git diff HEAD --stat"
  git diff HEAD --stat
  echo "git_diff_head_sha256=$(git diff HEAD | sha256sum)"
} > "$OUT/git_state.txt"
git diff HEAD -- $CHANGED > "$OUT/working_tree.diff"
sha256sum $CHANGED $HELPERS > "$OUT/sha256sums.txt"
export CUDA_VISIBLE_DEVICES=""
run() {
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "exit_status=$?"; } > "$OUT/$name.txt" 2>&1
  tail -1 "$OUT/$name.txt" | sed "s/^/$name: /"
}
T=tests.test_compare_scheduler_gpu.QueueGrowthTest
run 01_py_compile python -B -m py_compile $CHANGED
run 02_help_gpu_driver env PYTHONPATH="$PWD" python -B -W error scripts/compare_scheduler_gpu.py --help
run 03_help_cpu_batch env PYTHONPATH="$PWD" python -B -W error scripts/compare_coefficient_policies_cpu.py --help
run 04_new_tests python -B -m unittest $T.test_only_the_constrained_context_changed $T.test_native_capacity_preflight_agrees_with_native_initialization $T.test_incompatible_pool_fails_before_engine_construction $T.test_queue_growth_case_selection $T.test_queue_growth_inputs -v
run 05_compare_driver python -B -m unittest tests.test_compare_scheduler_gpu -v
run 06_constrained_cpu_rerun env PYTHONPATH="$PWD" python -B scripts/compare_coefficient_policies_cpu.py --batch queue_growth --cases queue48_burst_pool12 --output-dir "$RUN/cpu_comparison"
run 07_compare_with_prior_cpu python -B - "$PRIOR" "$RUN/cpu_comparison" <<'PY'
import json, sys
prior, fresh = sys.argv[1:]
same = True
for m in ("adapted_vllm", "lp_reciprocal_square", "ilp_reciprocal_square"):
    a = json.load(open(f"{prior}/rollouts/queue48_burst_pool12/{m}/summary.json"))
    b = json.load(open(f"{fresh}/rollouts/queue48_burst_pool12/{m}/summary.json"))
    keys = ("decisions", "workload_outcome", "preemptions", "prefill_tokens_replayed",
            "discarded_prompt_tokens", "generated_token_events", "overgeneration")
    equal = a["emitted"] == b["emitted"]
    same &= equal
    print(m, "max_model_len", a["parameters"]["max_model_len"], "->",
          b["parameters"]["max_model_len"], "| emitted identical:", equal,
          "|", {k: (a[k], b[k]) for k in keys})
    ca = [dict(r.get("comparison") or {}, integer=r.get("integer")) for r in
          json.load(open(f"{prior}/rollouts/queue48_burst_pool12/{m}/comparisons.json"))]
    cb = [dict(r.get("comparison") or {}, integer=r.get("integer")) for r in
          json.load(open(f"{fresh}/rollouts/queue48_burst_pool12/{m}/comparisons.json"))]
    strip = lambda r: {k: v for k, v in r.items() if k != "problem_id"}
    print("  same-state records identical apart from problem IDs:",
          [strip(x) for x in ca] == [strip(x) for x in cb])
sys.exit(0 if same else 1)
PY
run 08_runner_failure_propagation bash "$RUN/cpu_checks/runner_propagation/check.sh" "$RUN"
run 09_whitespace bash -c "! grep -nP '[ \t]+\$' $CHANGED docs/handoffs/scheduler_comparison_handoff.md"
run 10_git_diff_check git diff --check
