#!/bin/bash
# Focused CPU checks for the corrected LP-relaxation planning layer.
# Run from the repository root on an already allocated node; no srun/sbatch.
set -uo pipefail
OUT="$1"
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-} slurm_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} (not used)"
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
git diff -- lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py > "$OUT/working_tree.diff"
sha256sum lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py > "$OUT/sha256sums.txt"
run() {  # name, command...
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "exit_status=$?"; } > "$OUT/$name.txt" 2>&1
  tail -1 "$OUT/$name.txt" | sed "s/^/$name: /"
}
run 01_py_compile python -B -m py_compile lp_relaxation_scheduler.py tests/test_lp_relaxation_scheduler.py
run 02_import python -B -W error -c 'import sys, lp_relaxation_scheduler; print("sarathi_imported", "sarathi" in sys.modules)'
run 03_smoke python -B lp_relaxation_scheduler.py
run 04_unittest python -B -m unittest discover -s tests -p 'test_lp_relaxation_scheduler.py' -v
run 05_git_diff_check git diff --check
