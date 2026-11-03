#!/bin/zsh
# CPU gates for the soft-SLO pilot; run from the repository root.
set -u
module purge 2>/dev/null; module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 2>/dev/null
source env/bin/activate
export OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
OUT="$1"
log() { echo "$@" | tee -a "$OUT/gates.txt"; }
run() { local name="$1"; shift; log "## $name: $*"; "$@" > "$OUT/$name.log" 2>&1; local rc=$?; log "exit=$rc $(grep -E '^(Ran|OK|FAILED)' "$OUT/$name.log" | tr '\n' ' ')"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-} head=$(git rev-parse HEAD) python=$(which python) $(python --version 2>&1)"
run 01_py_compile python -B -m py_compile lpserve_state_mapping.py scripts/soft_slo_pilot.py tests/test_soft_slo_pilot.py
run 02_import python -B -c "import sys; sys.path.insert(0, 'scripts'); import soft_slo_pilot, lpserve_state_mapping as m; print([n for n in ('torch', 'ray') if n in sys.modules]); print(m.soft_slo_timed_utility, m.soft_slo_untimed_utility)"
run 03_diff_check git diff --check
run 04_whitespace_new_files zsh -c "! grep -nE '[[:space:]]+$' scripts/soft_slo_pilot.py tests/test_soft_slo_pilot.py"
run 06_test_soft_slo_pilot python -B -m unittest tests.test_soft_slo_pilot -v
for m in test_lp_relaxation_scheduler test_lp_scheduler test_lpserve_plan_execution test_lpserve_state_mapping test_lp_engine_args test_exact_integer_plan test_compare_scheduler_gpu; do
  run "07_$m" python -B -m unittest "tests.$m"
done
