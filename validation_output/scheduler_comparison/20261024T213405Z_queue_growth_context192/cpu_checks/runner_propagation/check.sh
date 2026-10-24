#!/bin/bash
# Runs the actual <run>/run_gpu.sh with fake child commands (no GPU, CUDA
# hidden) and checks its exit status per scenario.
set -uo pipefail
RUN="$1"; HERE="$RUN/cpu_checks/runner_propagation"
export CUDA_VISIBLE_DEVICES="" DRIVER="$HERE/fake_driver.py" OFFLINE="$HERE/fake_offline.py"
ok=0
scenario() {  # name, expected exit, env assignments...
  local name="$1" want="$2"; shift 2
  local out="$HERE/$name"; rm -rf "$out"; mkdir -p "$out"
  env "$@" bash "$RUN/run_gpu.sh" "$out" > "$out/runner_stdout.txt" 2>&1
  local got=$?
  echo "$name: runner exit $got (expected $want); failures: $(tr '\n' ';' < "$out/runner_failures.txt")"
  [ "$got" = "$want" ] || ok=1
}
scenario gpu_process_fails 1 FAKE_EXIT_lp_reciprocal_square=3
scenario offline_fails 1 FAKE_OFFLINE_EXIT=2
scenario all_succeed 0
exit $ok
