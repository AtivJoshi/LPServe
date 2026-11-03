#!/bin/zsh
# Completion-bonus experiment runs on the allocated node; run from the repository root.
set -u
E="$1"
M=/home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6
REG=validation_output/soft_slo_pilot/20261103T090031Z_registration_order
module purge 2>/dev/null; module load uri/main Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 2>/dev/null
source env/bin/activate
module -t list > "$E/modules.txt" 2>&1
nvidia-smi > "$E/nvidia_smi_before.txt" 2>&1
export OMP_NUM_THREADS=2 PYTHONPATH="$PWD:$PWD/scripts"
echo "## offline: python -B scripts/prompt_completion_study.py --evidence $REG --output $E/offline/study.json" | tee -a "$E/runs.txt"
python -B scripts/prompt_completion_study.py --evidence "$REG" --output "$E/offline/study.json" > "$E/offline/study.console.txt" 2>&1; echo "exit=$?" | tee -a "$E/runs.txt"
run() { local name="$1"; shift; echo "## $name: python -B scripts/soft_slo_pilot.py $* --output-dir $E/$name" | tee -a "$E/runs.txt"; timeout 900 python -B scripts/soft_slo_pilot.py "$@" --output-dir "$E/$name" > "$E/$name.console.txt" 2>&1; echo "exit=$?" | tee -a "$E/runs.txt"; }
run cpu_ample_bonus --cpu --case ample --registration-order tight_first --comparison bonus
run gpu_smoke --case ample --smoke --model-path "$M"
run gpu_ample_bonus --case ample --registration-order tight_first --comparison bonus --model-path "$M"
run gpu_incumbents_bonus --case ample_incumbents --registration-order tight_first --comparison bonus --model-path "$M"
nvidia-smi > "$E/nvidia_smi_after.txt" 2>&1
