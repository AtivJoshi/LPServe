#!/bin/bash
# Offline replay of the four documented failure cases (Task 2): CPU checks,
# fixture construction from the retained traces, replay, and a second replay
# from the written fixtures alone. Run from the repository root on an already
# allocated Unity node (no srun, no GPU):
#   bash <run>/run_batch.sh <run>
# Every step runs even if an earlier one failed. Each step's real exit
# status goes to steps.tsv, and the script exits nonzero if any step failed.
set -uo pipefail
RUN="$1"
OUT="$RUN/checks"
mkdir -p "$OUT"
BASE=adf4b262980c8e1b4c729b12d71d02b913c83040
module load uri/main Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1
source env/bin/activate
export CUDA_VISIBLE_DEVICES=""
CHANGED="scripts/replay_failure_cases.py tests/test_replay_failure_cases.py"
HELPERS="lp_relaxation_scheduler.py scripts/exact_integer_plan.py tests/test_lp_relaxation_scheduler.py tests/test_exact_integer_plan.py"
E=validation_output/scheduler_comparison
RETAINED="$E/20261024T213405Z_queue_growth_context192/cpu_comparison/rollouts/queue48_burst_pool12/lp_reciprocal_square/decision_trace.json.gz
$E/20261024T213405Z_queue_growth_context192/cpu_comparison/rollouts/queue48_burst_pool12/lp_reciprocal_square/comparisons.json
$E/20261026T045354Z_running_order/cpu_comparison/rollouts/queue48_burst_pool12_limit8/lp_running_order/decision_trace.json.gz
$E/20261026T045354Z_running_order/cpu_comparison/rollouts/queue48_burst_pool12_limit8/lp_running_order/comparisons.json.gz
$E/20261026T045354Z_running_order/cpu_comparison/diagnostics.json
$E/20261026T045354Z_running_order/cpu_comparison/rollouts/queue48_burst_limit8/lp_reciprocal_square/decision_trace.json.gz
$E/20261026T045354Z_running_order/cpu_comparison/rollouts/queue48_burst_limit8/lp_reciprocal_square/comparisons.json.gz
$E/20261026T045354Z_running_order/cpu_comparison/rollouts/queue96_burst_pool12_limit8/ilp_running_order/decision_trace.json.gz
$E/20261024T182347Z/cpu_comparison/rollouts/two_request_limit1/lp_uniform/decision_trace.json
$E/20261024T182347Z/cpu_comparison/rollouts/two_request_limit1/lp_rank_based/decision_trace.json"
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
  echo "--- git status --short --untracked-files=all"
  git status --short --untracked-files=all
  echo "--- git diff HEAD --stat (tracked changes; the multiplier comparison, not used here)"
  git diff HEAD --stat
  echo "git_diff_head_sha256=$(git diff HEAD | sha256sum)"
} > "$OUT/git_state.txt"
sha256sum $CHANGED $HELPERS $RETAINED > "$OUT/sha256sums_start.txt"
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
run 02_help python -B -W error scripts/replay_failure_cases.py --help
run 03_replay_tests python -B -m unittest tests.test_replay_failure_cases -v
run 04_exact_integer_plan python -B -m unittest tests.test_exact_integer_plan -v
run 05_lp_relaxation python -B -m unittest tests.test_lp_relaxation_scheduler -v
run 06_whitespace bash -c "! grep -nP '[ \t]+\$' $CHANGED"
run 07_git_diff_check git diff --check
run 08_replay_from_evidence python -B scripts/replay_failure_cases.py --out "$RUN/replay"
run 09_replay_from_fixtures python -B scripts/replay_failure_cases.py --out "$RUN/replay_from_fixtures" --fixtures "$RUN/replay/fixtures"
run 10_reports_equal cmp "$RUN/replay/report.txt" "$RUN/replay_from_fixtures/report.txt"
sha256sum $CHANGED $HELPERS $RETAINED > "$OUT/sha256sums_end.txt"
run 11_sources_unchanged_during_run cmp "$OUT/sha256sums_start.txt" "$OUT/sha256sums_end.txt"
failed=$(awk -F'\t' '$2 != 0' "$OUT/steps.tsv" | wc -l)
echo "failed_steps=$failed" | tee "$OUT/aggregate_status.txt"
awk -F'\t' '$2 != 0 {print "failed: " $1 " (exit " $2 ")"}' "$OUT/steps.tsv" | tee -a "$OUT/aggregate_status.txt"
if [ "$failed" -eq 0 ]; then exit 0; else exit 1; fi
