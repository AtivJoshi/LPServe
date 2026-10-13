#!/bin/bash
# Review-finding fixes: ordering helpers, explicit obsolete None, oversized
# arrival. Run from the repository root on an already allocated node.
set -uo pipefail
OUT="$1"
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
CHANGED="$(git diff --name-only HEAD) tests/test_check_lp_scheduler_ordering.py"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-} slurm_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} (not used; CPU-only checks)"
  echo "virtual_env=${VIRTUAL_ENV:-}"
  echo "which_python=$(which python)"
  python --version 2>&1
  python -c 'import numpy, scipy; print("numpy", numpy.__version__); print("scipy", scipy.__version__)'
} > "$OUT/environment.txt" 2>&1
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "--- git status --short (tests/ matches the repo's test* ignore rule, so the new test file is not listed)"
  git status --short
  echo "untracked ignored new file: tests/test_check_lp_scheduler_ordering.py"
} > "$OUT/git_state.txt"
git diff HEAD > "$OUT/working_tree.diff"
git diff --no-index /dev/null tests/test_check_lp_scheduler_ordering.py >> "$OUT/working_tree.diff"
sha256sum $CHANGED > "$OUT/sha256sums.txt"
run() {
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "exit_status=$?"; } > "$OUT/$name.txt" 2>&1
  tail -1 "$OUT/$name.txt" | sed "s/^/$name: /"
}
run 01_ordering_helpers python -B -m unittest discover -s tests -p 'test_check_lp_scheduler_ordering.py' -v
run 02_engine_args python -B -m unittest discover -s tests -p 'test_lp_engine_args.py' -v
run 03_live_scheduler_and_submission python -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v
run 04_existing_script_cpu_tests python -B -m unittest discover -s tests -p 'test_check_lp_scheduler_*_gpu.py' -v
run 05_py_compile python -B -m py_compile $(echo $CHANGED | tr ' ' '\n' | grep '\.py$')
run 06_import python -B -W error -c 'import sys; sys.path.insert(0, "scripts"); import sarathi.core.scheduler.lp_scheduler, sarathi.engine.arg_utils, check_lp_scheduler_contention_gpu, check_lp_scheduler_preemption_gpu, check_lp_scheduler_decode_preemption_reference_gpu; print("ok")'
run 07_git_diff_check git diff --check
PYTHONPATH=$PWD run 08_repro_after_fix python -B "$OUT/repro/repro_findings.py"
