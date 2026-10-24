#!/bin/bash
# Reciprocal-square queue-growth study: CPU gate and CPU comparison batch
# before the GPU confirmation. Run from the repository root on the already
# allocated node:
#   bash <run>/cpu_checks/run_checks.sh <run>
set -uo pipefail
RUN="$1"
OUT="$RUN/cpu_checks"
BASE=fce4fd1b57527f3d198aecb39dadc807799359d9
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
CHANGED="lpserve_state_mapping.py scripts/compare_scheduler_gpu.py scripts/exact_integer_plan.py scripts/compare_coefficient_policies_cpu.py tests/test_compare_scheduler_gpu.py tests/test_lpserve_state_mapping.py"
HELPERS="lp_relaxation_scheduler.py lpserve_plan_execution.py sarathi/core/scheduler/lp_scheduler.py sarathi/core/scheduler/adapted_vllm_scheduler.py scripts/check_adapted_vllm_gpu.py scripts/check_lp_scheduler_gpu.py scripts/check_lp_scheduler_preemption_gpu.py tests/test_lp_scheduler.py tests/test_lp_relaxation_scheduler.py tests/test_exact_integer_plan.py"
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
  echo "--- git diff HEAD --stat (tracked changes)"
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
run 01_py_compile python -B -m py_compile $CHANGED
run 02_help_gpu_driver env PYTHONPATH="$PWD" python -B -W error scripts/compare_scheduler_gpu.py --help
run 03_help_cpu_batch env PYTHONPATH="$PWD" python -B -W error scripts/compare_coefficient_policies_cpu.py --help
run 04_compare_driver python -B -m unittest tests.test_compare_scheduler_gpu -v
run 05_integer_plan python -B -m unittest tests.test_exact_integer_plan -v
run 06_state_mapping python -B -m unittest tests.test_lpserve_state_mapping -v
run 07_live_scheduler python -B -m unittest tests.test_lp_scheduler -v
run 08_plan_execution python -B -m unittest tests.test_lpserve_plan_execution -v
# Shared comparison helpers changed: regenerate the earlier coordinated batch
# outside the repository and compare its report with the retained one. The
# only expected difference is the mixed_pressure diagnostic corrected at fce4fd1.
REGEN="$(mktemp -d "${TMPDIR:-/tmp}/coordinated_regen.XXXXXX")"
run 09_coordinated_batch_regenerated env PYTHONPATH="$PWD" python -B scripts/compare_coefficient_policies_cpu.py --output-dir "$REGEN/out"
run 10_coordinated_report_diff bash -c 'diff "$0" "$1"; n=$(diff "$0" "$1" | grep -c "^[<>]"); m=$(diff "$0" "$1" | grep "^[<>]" | grep -vc mixed_pressure); echo "changed_lines=$n other_than_mixed_pressure=$m"; test "$n" = 2 && test "$m" = 0' validation_output/scheduler_comparison/20261024T182347Z/cpu_comparison/comparison.txt "$REGEN/out/comparison.txt"
rm -rf "$REGEN"
run 11_whitespace bash -c "! grep -nP '[ \t]+\$' $CHANGED"
run 12_git_diff_check git diff --check
run 13_queue_growth_batch env PYTHONPATH="$PWD" python -B scripts/compare_coefficient_policies_cpu.py --batch queue_growth --output-dir "$RUN/cpu_comparison"
