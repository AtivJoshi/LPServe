#!/bin/bash
# Adapted vLLM reference corrections and GPU-driver preparation: CPU-only
# checks. Run from the repository root on the already allocated node:
#   bash <out>/run_checks.sh <out>
set -uo pipefail
OUT="$1"
BASE=c6c89bc3c4f95114aecfd27bdbbc53e5e3374ec5
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
MODIFIED="sarathi/core/scheduler/adapted_vllm_scheduler.py sarathi/engine/arg_utils.py tests/test_adapted_vllm_scheduler.py"
NEW="scripts/check_adapted_vllm_gpu.py tests/test_check_adapted_vllm_gpu.py"
HELPERS="scripts/check_lp_scheduler_gpu.py scripts/check_lp_scheduler_preemption_gpu.py tests/test_lp_scheduler.py"
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
  echo "base_reviewed=$BASE"
  echo "--- git status --short --untracked-files=all"
  git status --short --untracked-files=all
} > "$OUT/git_state.txt"
{
  # Code only; documentation is not an input to any check.
  git diff HEAD -- sarathi/core/scheduler/adapted_vllm_scheduler.py sarathi/engine/arg_utils.py
  # tests/ matches .gitignore; diff the tracked test file explicitly.
  git diff HEAD -- tests/test_adapted_vllm_scheduler.py
  for f in $NEW; do diff -u /dev/null "$f"; done
} > "$OUT/working_tree.diff"
sha256sum $MODIFIED $NEW $HELPERS "$OUT/scripts/empty_prompt_at_base.py" > "$OUT/sha256sums.txt"
run() {
  local name="$1"; shift
  { echo "\$ $*"; "$@"; echo "exit_status=$?"; } > "$OUT/$name.txt" 2>&1
  tail -1 "$OUT/$name.txt" | sed "s/^/$name: /"
}
run 01_py_compile python -B -m py_compile $MODIFIED $NEW "$OUT/scripts/empty_prompt_at_base.py"
run 02_import python -B -W error -c 'import sys; sys.path.insert(0, "scripts"); import check_adapted_vllm_gpu as d; from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry; from sarathi.config import SchedulerType; print(SchedulerRegistry._registry[SchedulerType.ADAPTED_VLLM].__name__, sorted(d.CASES))'
run 03_focused_reference python -B -m unittest tests.test_adapted_vllm_scheduler -v
run 04_driver_cpu python -B -m unittest tests.test_check_adapted_vllm_gpu -v
run 05_empty_prompt_regression_at_base python -B "$OUT/scripts/empty_prompt_at_base.py" "$BASE"
run 06_regression_engine_args python -B -m unittest discover -s tests -p 'test_lp_engine_args.py' -v
run 07_regression_preemption_driver_helpers python -B -m unittest discover -s tests -p 'test_check_lp_scheduler_preemption_gpu.py' -v
run 08_git_diff_check git diff --check
