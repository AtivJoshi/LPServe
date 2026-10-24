#!/bin/bash
# Corrected mixed_pressure diagnostic (B has one generated token, so its
# decode needs one new block): focused CPU checks only. Run from the
# repository root on the already allocated node:
#   bash <run>/run_checks.sh <run>
set -uo pipefail
OUT="$1"
BASE=43aa8378bb466a80ee0a09ac55772fb5747964ba
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
CHANGED="scripts/compare_coefficient_policies_cpu.py tests/test_compare_scheduler_gpu.py"
DOCS="docs/handoffs/scheduler_comparison_handoff.md"
HELPERS="scripts/compare_scheduler_gpu.py scripts/exact_integer_plan.py lp_relaxation_scheduler.py lpserve_state_mapping.py lpserve_plan_execution.py sarathi/core/scheduler/lp_scheduler.py sarathi/core/scheduler/adapted_vllm_scheduler.py tests/test_lp_relaxation_scheduler.py tests/test_lp_scheduler.py"
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
  echo "git_diff_head_sha256=$(git diff HEAD -- $CHANGED | sha256sum)"
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
run 02_import env PYTHONPATH="$PWD" python -B -W error -c "import sys; sys.path[:0] = ['scripts', 'tests']; import compare_coefficient_policies_cpu, test_compare_scheduler_gpu"
run 03_regression_test python -B -m unittest tests.test_compare_scheduler_gpu.CoordinatedPolicyTest.test_mixed_pressure_uses_one_block_decode_demand -v
run 04_compare_driver_suite python -B -m unittest tests.test_compare_scheduler_gpu -v
run 05_diagnostic env PYTHONPATH="$PWD" python -B -c "
import json, sys
sys.path[:0] = ['scripts', 'tests']
import test_compare_scheduler_gpu as cpu
cpu.setUpModule()
import compare_coefficient_policies_cpu as batch
record = batch.driver.to_jsonable(batch.run_diagnostic(
    'mixed_pressure', batch.DIAGNOSTIC_STATES['mixed_pressure']))
open('$OUT/mixed_pressure_diagnostic.json', 'w').write(json.dumps(record, indent=1))
c = record['lp_vs_integer']
print('mapped', json.dumps(record['mapped_requests']))
print('lp relaxed', c['lp_relaxed_objective'], 'lp plan', c['lp_actions'], c['lp_plan_objective'])
print('integer', c['integer_actions'], c['integer_objective'], c['attribution'])
print('executed', json.dumps(record['executed']))
print('blocks', json.dumps(record['block_counts']))
print('reference', record['reference_vs_integer']['reference_actions'], record['reference_vs_integer']['classification'])
print('enumeration', record['enumeration'])
"
run 06_enumeration env PYTHONPATH="$PWD" python -B -c "
import json
r = json.load(open('$OUT/mixed_pressure_diagnostic.json'))
e, best = r['enumeration'], r['lp_vs_integer']['integer_objective']
print(e, 'integer optimum', best)
assert e['useful_plan_exists'] and abs(e['best'] - best) <= 1e-7, 'enumeration disagrees'
print('enumeration agrees with the integer optimum')
"
run 07_git_diff_check git diff --check -- $CHANGED $DOCS
