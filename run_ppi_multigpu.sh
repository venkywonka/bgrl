#!/bin/bash

# Set the GPU indices you want to use
gpu1="1"
gpu2="2"

(
    echo "Running control experiment on GPU $gpu1..."
    CUDA_VISIBLE_DEVICES="$gpu1" python3 finetune_ppi.py --flagfile=config-eval/ppi.cfg --experiment=control --logdir=control_runs/ppi/
) &

(
    echo "Running SSL experiment on GPU $gpu2..."
    CUDA_VISIBLE_DEVICES="$gpu2" python3 finetune_ppi.py --flagfile=config-eval/ppi.cfg --experiment=ssl --logdir=finetune_runs/ppi/
) &

wait