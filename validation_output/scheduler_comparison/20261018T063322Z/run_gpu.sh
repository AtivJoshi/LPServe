#!/bin/bash
# Six bounded GPU comparison runs, one fresh process each, sequentially on
# the already allocated GPU. Run from the repository root:
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
sha256sum scripts/compare_scheduler_gpu.py tests/test_compare_scheduler_gpu.py \
  scripts/check_adapted_vllm_gpu.py scripts/check_lp_scheduler_gpu.py \
  scripts/check_lp_scheduler_preemption_gpu.py > "$OUT/sha256sums_gpu.txt"
for LIMIT in 2 1; do
  for POLICY in adapted_vllm lp_uniform lp_rank_based; do
    RUN="$OUT/${POLICY}_resident${LIMIT}"
    mkdir -p "$RUN"
    CMD=(timeout 600s python -B scripts/compare_scheduler_gpu.py --policy "$POLICY"
         --resident-limit "$LIMIT" --model-path "$MODEL_DIR" --output-dir "$RUN")
    echo "${CMD[*]}" > "$RUN/command.txt"
    echo "=== $POLICY limit $LIMIT start $(date -u +%H:%M:%SZ)"
    "${CMD[@]}" 2>&1 | tee "$RUN/console.log" > /dev/null
    STATUSES=("${PIPESTATUS[@]}")
    echo "timeout_python_exit_status=${STATUSES[0]} tee_exit_status=${STATUSES[1]}" > "$RUN/exit_status.txt"
    grep -E "decision [0-9]+:|RESULT|CHECK FAILED" "$RUN/console.log"
    cat "$RUN/exit_status.txt"
  done
done
