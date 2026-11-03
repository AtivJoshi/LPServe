#!/bin/zsh
# CPU gates for the completion-bonus experiment; run from the repository root.
set -u
module purge 2>/dev/null; module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 2>/dev/null
source env/bin/activate
export OMP_NUM_THREADS=2 PYTHONPATH="$PWD:$PWD/scripts"
OUT="$1"
log() { echo "$@" | tee -a "$OUT/gates.txt"; }
run() { local name="$1"; shift; log "## $name: $*"; "$@" > "$OUT/$name.log" 2>&1; local rc=$?; log "exit=$rc $(grep -E '^(Ran|OK|FAILED)' "$OUT/$name.log" | tr '\n' ' ')"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-} head=$(git rev-parse HEAD) python=$(which python) $(python --version 2>&1)"
run 01_py_compile python -B -m py_compile lp_relaxation_scheduler.py lpserve_state_mapping.py scripts/exact_integer_plan.py scripts/soft_slo_pilot.py scripts/prompt_completion_study.py tests/test_soft_slo_pilot.py
run 02_diff_check git diff --check
run 03_whitespace_new zsh -c "! grep -nE '[[:space:]]+$' scripts/prompt_completion_study.py"
run 04_test_soft_slo_pilot python -B -m unittest tests.test_soft_slo_pilot -v
for m in test_lp_relaxation_scheduler test_lp_scheduler test_lpserve_plan_execution test_lpserve_state_mapping test_lp_engine_args test_exact_integer_plan test_compare_scheduler_gpu test_check_lp_scheduler_ordering; do
  run "05_$m" python -B -m unittest "tests.$m"
done
