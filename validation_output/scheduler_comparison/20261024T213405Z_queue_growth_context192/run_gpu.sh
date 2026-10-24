#!/bin/bash
# Two bounded GPU confirmation runs of queue48_burst_pool12 at the approved
# max_model_len 192 (12 blocks), one fresh process each, sequentially on the
# already allocated GPU: reciprocal-square LP, then the adapted reference.
# Each trace is hashed and gzip-compressed (-n). Then, with CUDA hidden,
# every reached decision gets its offline CPU integer comparison and the
# emitted decisions are checked against the corrected CPU rollouts in
# <run>/cpu_comparison. Run from the repository root:
#   bash <run>/run_gpu.sh <run>
#
# Exit status: 0 only if both GPU processes, both tee pipelines, every
# artifact operation, and the offline comparison exited 0. Every failure is
# appended to <run>/runner_failures.txt and the remaining independent steps
# still run, so partial evidence is kept. A failed GPU process is never
# retried or reused. DRIVER and OFFLINE override the two Python entrypoints
# only for the CPU-only failure-propagation check.
set -uo pipefail
OUT="$1"
DRIVER="${DRIVER:-scripts/compare_scheduler_gpu.py}"
OFFLINE="${OFFLINE:-scripts/compare_coefficient_policies_cpu.py}"
CASE=queue48_burst_pool12
FAILURES=0
fail() {
  FAILURES=$((FAILURES + 1))
  echo "FAILED: $*" >> "$OUT/runner_failures.txt"
  echo "FAILED: $*"
}
: > "$OUT/runner_failures.txt" || exit 1
module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 >/dev/null 2>&1
source env/bin/activate || fail "activate env/bin/activate"
MODEL_DIR=/home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
export PYTHONPATH="$PWD"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host=$(hostname)"
  env | grep -E '^(SLURM_JOB_ID|SLURM_JOB_NODELIST|SLURM_JOB_PARTITION|CUDA_VISIBLE_DEVICES|VIRTUAL_ENV|PYTHONPATH|LOADEDMODULES)=' | sort
  echo "which_python=$(which python)"; python --version 2>&1
  python -c 'import numpy, scipy, torch, ray; print("numpy", numpy.__version__); print("scipy", scipy.__version__); print("torch", torch.__version__); print("ray", ray.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used,memory.total --format=csv
  echo "model_dir=$MODEL_DIR"; ls "$MODEL_DIR"
  echo "driver=$DRIVER offline=$OFFLINE"
} > "$OUT/shell_environment.txt" 2>&1
{
  echo "head=$(git rev-parse HEAD)"
  git status --short --untracked-files=all
  echo "git_diff_head_sha256=$(git diff HEAD | sha256sum)"
} > "$OUT/git_state_gpu.txt" || fail "git state record"
sha256sum scripts/compare_scheduler_gpu.py scripts/compare_coefficient_policies_cpu.py \
  tests/test_compare_scheduler_gpu.py lpserve_state_mapping.py \
  scripts/exact_integer_plan.py lp_relaxation_scheduler.py lpserve_plan_execution.py \
  sarathi/core/scheduler/lp_scheduler.py sarathi/core/scheduler/adapted_vllm_scheduler.py \
  sarathi/engine/base_llm_engine.py scripts/check_adapted_vllm_gpu.py \
  scripts/check_lp_scheduler_gpu.py scripts/check_lp_scheduler_preemption_gpu.py \
  > "$OUT/sha256sums_gpu.txt" || fail "source hashes"
RUNS=()
for POLICY in lp_reciprocal_square adapted_vllm; do
  RUN="$OUT/gpu_${CASE}_${POLICY}"
  RUNS+=("$RUN")
  mkdir -p "$RUN" || { fail "$POLICY: mkdir $RUN"; continue; }
  CMD=(timeout 600s python -B "$DRIVER" --case "$CASE" --policy "$POLICY"
       --model-path "$MODEL_DIR" --output-dir "$RUN")
  echo "${CMD[*]}" > "$RUN/command.txt" || fail "$POLICY: command record"
  echo "=== $CASE $POLICY start $(date -u +%H:%M:%SZ)"
  "${CMD[@]}" 2>&1 | tee "$RUN/console.log" > /dev/null
  STATUSES=("${PIPESTATUS[@]}")
  echo "timeout_python_exit_status=${STATUSES[0]} tee_exit_status=${STATUSES[1]}" \
    > "$RUN/exit_status.txt" || fail "$POLICY: exit status record"
  echo "=== $CASE $POLICY end $(date -u +%H:%M:%SZ) python=${STATUSES[0]} tee=${STATUSES[1]}"
  [ "${STATUSES[0]}" = 0 ] || fail "$POLICY: GPU process exited ${STATUSES[0]}"
  [ "${STATUSES[1]}" = 0 ] || fail "$POLICY: tee exited ${STATUSES[1]}"
  grep -E "RESULT|CHECK FAILED" "$RUN/console.log"
  if [ -f "$RUN/decision_trace.json" ]; then
    sha256sum "$RUN/decision_trace.json" > "$RUN/decision_trace_json_sha256.txt" \
      || fail "$POLICY: trace hash"
    gzip -n "$RUN/decision_trace.json" || fail "$POLICY: trace gzip"
  else
    fail "$POLICY: no decision_trace.json"
  fi
done
echo "=== offline CPU integer comparisons start $(date -u +%H:%M:%SZ)"
CUDA_VISIBLE_DEVICES="" python -B "$OFFLINE" --batch retained_runs \
  --runs "${RUNS[@]}" --cpu-batch "$OUT/cpu_comparison" \
  --output-dir "$OUT/gpu_offline_comparison" > "$OUT/gpu_offline_comparison_stdout.txt" 2>&1
OFFLINE_STATUS=$?
echo "offline_comparison_exit_status=$OFFLINE_STATUS" > "$OUT/gpu_offline_comparison_exit_status.txt" \
  || fail "offline status record"
[ "$OFFLINE_STATUS" = 0 ] || fail "offline comparison exited $OFFLINE_STATUS"
tail -1 "$OUT/gpu_offline_comparison_stdout.txt"
echo "runner_failures=$FAILURES" | tee "$OUT/runner_exit_status.txt"
[ "$FAILURES" -eq 0 ] && exit 0
exit 1
