#!/bin/bash
# Scheduler comparison driver, memory-pressure case: CPU-only gate before the
# GPU runs. Run from the repository root on the already allocated node:
#   bash <out>/cpu_checks/run_checks.sh <out>/cpu_checks
set -uo pipefail
OUT="$1"
BASE=d988ed1643689bfff0a4de05aa62a774bfa63190
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
CHANGED="scripts/compare_scheduler_gpu.py tests/test_compare_scheduler_gpu.py"
HELPERS="scripts/check_adapted_vllm_gpu.py scripts/check_lp_scheduler_gpu.py scripts/check_lp_scheduler_preemption_gpu.py tests/test_lp_scheduler.py"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-} slurm_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} (hidden from the checks; CPU only)"
  echo "virtual_env=${VIRTUAL_ENV:-}"
  echo "which_python=$(which python)"
  python --version 2>&1
  python -c 'import numpy, scipy, torch, ray; print("numpy", numpy.__version__); print("scipy", scipy.__version__); print("torch", torch.__version__); print("ray", ray.__version__)'
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
run 02_help env PYTHONPATH="$PWD" python -B -W error scripts/compare_scheduler_gpu.py --help
run 03_driver_cpu python -B -m unittest tests.test_compare_scheduler_gpu -v
run 04_whitespace bash -c "! grep -nP '[ \t]+\$' $CHANGED"
run 05_git_diff_check git diff --check
