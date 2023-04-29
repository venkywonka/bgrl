#!/bin/bash

declare -A datasets
datasets=(
    ["coauthor-cs"]="3"
    ["coauthor-physics"]="4"
    ["amazon-photos"]="1"
    ["amazon-computers"]="2"
)

for dataset in "${!datasets[@]}"; do
    gpu="${datasets[$dataset]}"

    (
        echo "Running control experiment for $dataset on GPU $gpu..."
        CUDA_VISIBLE_DEVICES="$gpu" python3 finetune_transductive.py --flagfile=config-eval/"$dataset".cfg --experiment=control --logdir=control_runs/"$dataset"/

        echo "Running SSL experiment for $dataset on GPU $gpu..."
        CUDA_VISIBLE_DEVICES="$gpu" python3 finetune_transductive.py --flagfile=config-eval/"$dataset".cfg --experiment=ssl --logdir=finetune_runs/"$dataset"/
    ) &
done

wait