#!/bin/bash
# Adapted vLLM reference: CPU-only checks and bounded comparison. Run from the
# repository root on the already allocated node: bash <out>/run_checks.sh <out>
set -uo pipefail
OUT="$1"
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
MODIFIED="sarathi/config.py sarathi/core/scheduler/scheduler_registry.py sarathi/core/block_space_manager/block_space_manager_registry.py sarathi/engine/arg_utils.py"
NEW="sarathi/core/scheduler/adapted_vllm_scheduler.py tests/test_adapted_vllm_scheduler.py"
DOCS="docs/LP Scheduler Research Context.md"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-} slurm_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} (not used; CPU-only checks)"
  echo "virtual_env=${VIRTUAL_ENV:-}"
  echo "which_python=$(which python)"
  python --version 2>&1
  python -c 'import numpy, scipy, torch; print("numpy", numpy.__version__); print("scipy", scipy.__version__); print("torch", torch.__version__)'
  module list 2>&1
} > "$OUT/environment.txt" 2>&1
{
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "head=$(git rev-parse HEAD)"
  echo "--- git status --short"
  git status --short
} > "$OUT/git_state.txt"
{
  git diff HEAD -- $MODIFIED "$DOCS"
  for f in $NEW; do diff -u /dev/null "$f"; done
} > "$OUT/working_tree.diff"
sha256sum $MODIFIED $NEW "$DOCS" "$OUT/scripts/compare_policies.py" > "$OUT/sha256sums.txt"
run() {
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "exit_status=$?"; } > "$OUT/$name.txt" 2>&1
  tail -1 "$OUT/$name.txt" | sed "s/^/$name: /"
}
run 01_py_compile python -B -m py_compile $MODIFIED $NEW "$OUT/scripts/compare_policies.py"
run 02_import python -B -W error -c 'from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry; from sarathi.config import SchedulerType; print(SchedulerRegistry.get_key_from_str("adapted_vllm"), SchedulerRegistry._registry[SchedulerType.ADAPTED_VLLM])'
run 03_focused_reference python -B -m unittest tests.test_adapted_vllm_scheduler -v
PYTHONPATH=$PWD run 04_comparison python -B "$OUT/scripts/compare_policies.py" "$OUT"
run 05_regression_live_scheduler python -B -m unittest discover -s tests -p 'test_lp_scheduler.py' -v
run 06_regression_engine_args python -B -m unittest discover -s tests -p 'test_lp_engine_args.py' -v
run 07_regression_preemption_driver python -B -m unittest discover -s tests -p 'test_check_lp_scheduler_preemption_gpu.py' -v
run 08_git_diff_check git diff --check
