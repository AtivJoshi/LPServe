#!/bin/bash
# Four bounded GPU confirmation runs, one fresh process each, sequentially on
# the already allocated GPU: coordinated LP and the adapted reference on the
# two-request case (resident limit 1) and the memory-pressure case. Run from
# the repository root:
#   bash <out>/run_gpu.sh <out>
set -uo pipefail
OUT="$1"
module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 >/dev/null 2>&1
source env/bin/activate
MODEL_DIR=/home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
export PYTHONPATH="$PWD"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host=$(hostname)"
  env | grep -E '^(SLURM_JOB_ID|SLURM_JOB_NODELIST|SLURM_JOB_PARTITION|CUDA_VISIBLE_DEVICES|VIRTUAL_ENV|PYTHONPATH|LOADEDMODULES)=' | sort
  echo "which_python=$(which python)"; python --version 2>&1
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used,memory.total --format=csv
} > "$OUT/shell_environment.txt" 2>&1
{
  echo "head=$(git rev-parse HEAD)"
  git status --short --untracked-files=all
  echo "git_diff_head_sha256=$(git diff HEAD | sha256sum)"
} > "$OUT/git_state_gpu.txt"
sha256sum lp_relaxation_scheduler.py lpserve_state_mapping.py \
  lpserve_plan_execution.py scripts/compare_scheduler_gpu.py \
  scripts/exact_integer_plan.py scripts/compare_coefficient_policies_cpu.py \
  tests/test_compare_scheduler_gpu.py tests/test_exact_integer_plan.py \
  tests/test_lpserve_state_mapping.py scripts/check_adapted_vllm_gpu.py \
  scripts/check_lp_scheduler_gpu.py scripts/check_lp_scheduler_preemption_gpu.py \
  > "$OUT/sha256sums_gpu.txt"
for SPEC in "two_request lp_coordinated 1" "two_request adapted_vllm 1" \
            "memory_pressure lp_coordinated 4" "memory_pressure adapted_vllm 4"; do
  read -r CASE POLICY LIMIT <<< "$SPEC"
  RUN="$OUT/gpu_${CASE}_${POLICY}_limit${LIMIT}"
  mkdir -p "$RUN"
  CMD=(timeout 600s python -B scripts/compare_scheduler_gpu.py --case "$CASE"
       --policy "$POLICY" --resident-limit "$LIMIT" --model-path "$MODEL_DIR"
       --output-dir "$RUN")
  echo "${CMD[*]}" > "$RUN/command.txt"
  echo "=== $CASE $POLICY limit $LIMIT start $(date -u +%H:%M:%SZ)"
  "${CMD[@]}" 2>&1 | tee "$RUN/console.log" > /dev/null
  STATUSES=("${PIPESTATUS[@]}")
  echo "timeout_python_exit_status=${STATUSES[0]} tee_exit_status=${STATUSES[1]}" > "$RUN/exit_status.txt"
  grep -E "decision [0-9]+:|RESULT|CHECK FAILED" "$RUN/console.log"
  cat "$RUN/exit_status.txt"
done
