#!/bin/bash
# Two bounded GPU confirmation runs of the constrained 48-request burst
# (queue48_burst_pool12), one fresh process each, sequentially on the already
# allocated GPU: reciprocal-square LP, then the adapted reference. Each trace
# is hashed, then gzip-compressed (-n: no name or timestamp). Afterwards, with
# CUDA hidden, every reached decision gets its offline CPU integer comparison
# from the retained problems, and the emitted decisions are checked against
# the CPU batch. Run from the repository root:
#   bash <run>/run_gpu.sh <run>
set -uo pipefail
OUT="$1"
module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 >/dev/null 2>&1
source env/bin/activate
MODEL_DIR=/home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
export PYTHONPATH="$PWD"
CASE=queue48_burst_pool12
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host=$(hostname)"
  env | grep -E '^(SLURM_JOB_ID|SLURM_JOB_NODELIST|SLURM_JOB_PARTITION|CUDA_VISIBLE_DEVICES|VIRTUAL_ENV|PYTHONPATH|LOADEDMODULES)=' | sort
  echo "which_python=$(which python)"; python --version 2>&1
  python -c 'import numpy, scipy, torch, ray; print("numpy", numpy.__version__); print("scipy", scipy.__version__); print("torch", torch.__version__); print("ray", ray.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used,memory.total --format=csv
} > "$OUT/shell_environment.txt" 2>&1
{
  echo "head=$(git rev-parse HEAD)"
  git status --short --untracked-files=all
  echo "git_diff_head_sha256=$(git diff HEAD | sha256sum)"
} > "$OUT/git_state_gpu.txt"
sha256sum lpserve_state_mapping.py scripts/compare_scheduler_gpu.py \
  scripts/exact_integer_plan.py scripts/compare_coefficient_policies_cpu.py \
  tests/test_compare_scheduler_gpu.py tests/test_lpserve_state_mapping.py \
  lp_relaxation_scheduler.py lpserve_plan_execution.py \
  sarathi/core/scheduler/lp_scheduler.py \
  sarathi/core/scheduler/adapted_vllm_scheduler.py \
  scripts/check_adapted_vllm_gpu.py scripts/check_lp_scheduler_gpu.py \
  scripts/check_lp_scheduler_preemption_gpu.py tests/test_lp_scheduler.py \
  tests/test_lp_relaxation_scheduler.py tests/test_exact_integer_plan.py \
  > "$OUT/sha256sums_gpu.txt"
RUNS=()
for POLICY in lp_reciprocal_square adapted_vllm; do
  RUN="$OUT/gpu_${CASE}_${POLICY}"
  RUNS+=("$RUN")
  mkdir -p "$RUN"
  CMD=(timeout 600s python -B scripts/compare_scheduler_gpu.py --case "$CASE"
       --policy "$POLICY" --model-path "$MODEL_DIR" --output-dir "$RUN")
  echo "${CMD[*]}" > "$RUN/command.txt"
  echo "=== $CASE $POLICY start $(date -u +%H:%M:%SZ)"
  "${CMD[@]}" 2>&1 | tee "$RUN/console.log" > /dev/null
  STATUSES=("${PIPESTATUS[@]}")
  echo "timeout_python_exit_status=${STATUSES[0]} tee_exit_status=${STATUSES[1]}" > "$RUN/exit_status.txt"
  echo "=== $CASE $POLICY end $(date -u +%H:%M:%SZ)"
  grep -cE "decision [0-9]+:" "$RUN/console.log" | sed 's/^/decisions_logged=/'
  grep -E "RESULT|CHECK FAILED" "$RUN/console.log"
  cat "$RUN/exit_status.txt"
  if [ -f "$RUN/decision_trace.json" ]; then
    sha256sum "$RUN/decision_trace.json" > "$RUN/decision_trace_json_sha256.txt"
    gzip -n "$RUN/decision_trace.json"
  fi
done
echo "=== offline CPU integer comparisons start $(date -u +%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES="" python -B scripts/compare_coefficient_policies_cpu.py \
  --batch retained_runs --runs "${RUNS[@]}" --cpu-batch "$OUT/cpu_comparison" \
  --output-dir "$OUT/gpu_offline_comparison" > "$OUT/gpu_offline_comparison_stdout.txt" 2>&1
echo "offline_comparison_exit_status=$?" | tee "$OUT/gpu_offline_comparison_exit_status.txt"
tail -1 "$OUT/gpu_offline_comparison_stdout.txt"
