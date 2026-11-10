#!/bin/bash
# Fixed multiplier comparison: CPU checks, the 10-rollout matrix (five
# methods on queue48_burst and queue48_burst_pool12), same-state integer
# comparisons, retained-control reproduction, and the report, in one batch.
# Run from the repository root on an already allocated Unity node (no srun,
# no GPU):
#   bash <run>/run_batch.sh <run>
# Every step runs even if an earlier one failed. Each step's real exit
# status goes to steps.tsv, and the script exits nonzero if any step failed.
set -uo pipefail
RUN="$1"
OUT="$RUN/checks"
mkdir -p "$OUT"
BASE=adf4b262980c8e1b4c729b12d71d02b913c83040
RETAINED=validation_output/scheduler_comparison/20261026T045354Z_running_order/cpu_comparison
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
export CUDA_VISIBLE_DEVICES=""
CHANGED="scripts/compare_coefficient_policies_cpu.py scripts/compare_scheduler_gpu.py tests/test_compare_scheduler_gpu.py"
HELPERS="lpserve_state_mapping.py scripts/exact_integer_plan.py lp_relaxation_scheduler.py lpserve_plan_execution.py sarathi/core/scheduler/lp_scheduler.py sarathi/core/scheduler/adapted_vllm_scheduler.py sarathi/engine/base_llm_engine.py scripts/check_adapted_vllm_gpu.py scripts/check_lp_scheduler_gpu.py scripts/check_lp_scheduler_preemption_gpu.py tests/test_lp_scheduler.py tests/test_lp_relaxation_scheduler.py tests/test_exact_integer_plan.py tests/test_lpserve_plan_execution.py tests/test_lpserve_state_mapping.py"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-} slurm_nodelist=${SLURM_JOB_NODELIST:-} partition=${SLURM_JOB_PARTITION:-}"
  echo "cuda_visible_devices='${CUDA_VISIBLE_DEVICES}' (hidden; CPU only)"
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
  echo "retained_controls=$RETAINED"
  echo "--- git status --short --untracked-files=all"
  git status --short --untracked-files=all
  echo "--- git diff HEAD --stat (tracked changes)"
  git diff HEAD --stat
  echo "git_diff_head_sha256=$(git diff HEAD | sha256sum)"
} > "$OUT/git_state.txt"
git diff HEAD -- $CHANGED > "$OUT/working_tree.diff"
sha256sum $CHANGED $HELPERS > "$OUT/sha256sums_start.txt"
: > "$OUT/steps.tsv"
run() {
  local name="$1"; shift
  { echo "\$ $*"; "$@"; } > "$OUT/$name.txt" 2>&1
  local status=$?
  echo "exit_status=$status" >> "$OUT/$name.txt"
  printf '%s\t%s\n' "$name" "$status" >> "$OUT/steps.tsv"
  echo "$name: exit_status=$status"
}
run 01_py_compile python -B -m py_compile $CHANGED
run 02_help_gpu_driver env PYTHONPATH="$PWD" python -B -W error scripts/compare_scheduler_gpu.py --help
run 03_help_cpu_batch env PYTHONPATH="$PWD" python -B -W error scripts/compare_coefficient_policies_cpu.py --help
run 04_multiplier_tests python -B -m unittest tests.test_compare_scheduler_gpu.MultiplierTest -v
run 05_compare_driver python -B -m unittest tests.test_compare_scheduler_gpu -v
run 06_state_mapping python -B -m unittest tests.test_lpserve_state_mapping -v
run 07_exact_integer_plan python -B -m unittest tests.test_exact_integer_plan -v
run 08_lp_relaxation python -B -m unittest tests.test_lp_relaxation_scheduler -v
run 09_plan_execution python -B -m unittest tests.test_lpserve_plan_execution -v
run 10_live_scheduler python -B -m unittest tests.test_lp_scheduler -v
run 11_whitespace bash -c "! grep -nP '[ \t]+\$' $CHANGED"
run 12_git_diff_check git diff --check
run 13_multiplier_batch env PYTHONPATH="$PWD" python -B scripts/compare_coefficient_policies_cpu.py --batch multiplier --retained-controls "$RETAINED" --output-dir "$RUN/cpu_comparison"
sha256sum $CHANGED $HELPERS > "$OUT/sha256sums_end.txt"
run 14_sources_unchanged_during_run cmp "$OUT/sha256sums_start.txt" "$OUT/sha256sums_end.txt"
failed=$(awk -F'\t' '$2 != 0' "$OUT/steps.tsv" | wc -l)
echo "failed_steps=$failed" | tee "$OUT/aggregate_status.txt"
awk -F'\t' '$2 != 0 {print "failed: " $1 " (exit " $2 ")"}' "$OUT/steps.tsv" | tee -a "$OUT/aggregate_status.txt"
if [ "$failed" -eq 0 ]; then exit 0; else exit 1; fi
