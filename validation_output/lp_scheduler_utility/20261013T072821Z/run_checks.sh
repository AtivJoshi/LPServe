#!/bin/bash
# Rank-based utility policy: CPU-only checks. Run from the repository root on
# the already allocated node: bash <out>/run_checks.sh <out>
set -uo pipefail
OUT="$1"
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
CHANGED="lpserve_state_mapping.py tests/test_lpserve_state_mapping.py tests/test_lp_scheduler.py"
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
git diff HEAD -- $CHANGED > "$OUT/working_tree.diff"
sha256sum $CHANGED > "$OUT/sha256sums.txt"
run() {
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "exit_status=$?"; } > "$OUT/$name.txt" 2>&1
  tail -1 "$OUT/$name.txt" | sed "s/^/$name: /"
}
run 01_py_compile python -B -m py_compile $CHANGED
run 02_import python -B -W error -c 'import lpserve_state_mapping as m, sarathi.core.scheduler.lp_scheduler; print(m.rank_based_utility)'
run 03_focused_mapper python -B -m unittest tests.test_lpserve_state_mapping -k rank_based -v
run 04_focused_live python -B -m unittest tests.test_lp_scheduler.LiveRankBasedUtilityTest -v
run 05_full_mapper python -B -m unittest discover -s tests -p 'test_lpserve_state_mapping.py' -v
run 06_full_live_scheduler python -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v
run 07_git_diff_check git diff --check
PYTHONPATH=$PWD run 08_trace_dump python -B "$OUT/scripts/dump_trace.py" "$OUT"
PYTHONPATH=$PWD run 09_raw_id_control python -B "$OUT/scripts/raw_id_control.py"
PYTHONPATH=$PWD run 10_repro_resident_limit_one python -B "$OUT/scripts/repro_resident_limit_one.py"
