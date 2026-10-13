#!/bin/bash
# Corrected connected CPU framework checks, in dependency order.
# Run from the repository root on an already allocated node; no srun/sbatch.
set -uo pipefail
OUT="$1"
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
CHANGED=$(git diff --name-only HEAD)
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-} slurm_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} (not used; CPU-only checks)"
  echo "virtual_env=${VIRTUAL_ENV:-}"
  echo "which_python=$(which python)"
  python --version 2>&1
  python -c 'import numpy, scipy; print("numpy", numpy.__version__); print("scipy", scipy.__version__)'
  module list 2>&1
} > "$OUT/environment.txt" 2>&1
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "--- git status --short"
  git status --short
} > "$OUT/git_state.txt"
git diff HEAD > "$OUT/working_tree.diff"
sha256sum $CHANGED > "$OUT/sha256sums.txt"
run() {  # name, command...
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "exit_status=$?"; } > "$OUT/$name.txt" 2>&1
  tail -1 "$OUT/$name.txt" | sed "s/^/$name: /"
}
PY_CHANGED=$(echo "$CHANGED" | grep '\.py$')
run 01_py_compile python -B -m py_compile $PY_CHANGED
run 02_import python -B -W error -c 'import sys, lp_relaxation_scheduler, lpserve_state_mapping, lpserve_plan_execution; import sarathi.core.scheduler.lp_scheduler, sarathi.engine.arg_utils, sarathi.engine.base_llm_engine; print("ok")'
run 03_smoke python -B lp_relaxation_scheduler.py
run 04_pure python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
run 05_mapping python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
run 06_executor python -B -m unittest discover -s tests -p 'test_lpserve_plan_execution.py' -v
run 07_live_scheduler_and_submission python -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v
run 08_engine_args python -B -m unittest discover -s tests -p 'test_lp_engine_args.py' -v
run 09_script_cpu_tests python -B -m unittest discover -s tests -p 'test_check_lp*.py' -v
run 10_run_lp_inference_cpu_tests python -B -m unittest discover -s tests -p 'test_run_lp_inference.py' -v
run 11_git_diff_check git diff --check
