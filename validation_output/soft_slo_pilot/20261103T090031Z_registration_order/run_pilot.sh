#!/bin/zsh
# Registration-order pilot runs on the allocated node; run from the repository root.
set -u
E="$1"
M=/home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
module purge 2>/dev/null; module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 2>/dev/null
source env/bin/activate
module -t list > "$E/modules.txt" 2>&1
nvidia-smi > "$E/nvidia_smi_before.txt" 2>&1
export OMP_NUM_THREADS=2 PYTHONPATH="$PWD:$PWD/scripts"
run() { local name="$1"; shift; echo "## $name: python -B scripts/soft_slo_pilot.py $* --output-dir $E/$name" | tee -a "$E/runs.txt"; timeout 900 python -B scripts/soft_slo_pilot.py "$@" --output-dir "$E/$name" > "$E/$name.console.txt" 2>&1; echo "exit=$?" | tee -a "$E/runs.txt"; }
run cpu_ample_relaxed_first --cpu --case ample --registration-order relaxed_first
run cpu_ample_tight_first --cpu --case ample --registration-order tight_first
run gpu_smoke --case ample --smoke --model-path "$M"
run gpu_ample_relaxed_first --case ample --registration-order relaxed_first --model-path "$M"
run gpu_ample_tight_first --case ample --registration-order tight_first --model-path "$M"
nvidia-smi > "$E/nvidia_smi_after.txt" 2>&1
