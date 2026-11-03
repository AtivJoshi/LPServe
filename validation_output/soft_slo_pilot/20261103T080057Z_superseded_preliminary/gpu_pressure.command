PYTHONPATH="$PWD:$PWD/scripts" timeout 900 python -B scripts/soft_slo_pilot.py --case pressure --model-path /home/atjoshi_umass_edu/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6 --output-dir validation_output/soft_slo_pilot/20261103T080057Z/gpu_pressure
exit=0
