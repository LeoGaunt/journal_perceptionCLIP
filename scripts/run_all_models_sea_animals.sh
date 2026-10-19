#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"

MODELS=(
    "OpenCLIP-ViT-B-16:8192"
    "OpenCLIP-ViT-L-14:4096"
    "OpenCLIP-ViT-H-14:4096"
    "ConvNeXt-B:2048"
    "SigLIP-ViT-B-16:8192"
    "SigLIP2-ViT-B-16:8192"
    "MetaCLIP-ViT-B-16:4096"
    "DataComp-ViT-B-16:4096"
)

for MODEL_SPEC in "${MODELS[@]}"; do
    MODEL="${MODEL_SPEC%%:*}"
    BATCH="${MODEL_SPEC##*:}"
    SAFE=$(echo "$MODEL" | tr '/@ ' '---')

    # Determine which results folder this model lives in
    case "$MODEL" in
        OpenCLIP-ViT-B-16) FOLDER="ViT-B_Baselines" ;;
        OpenCLIP-ViT-L-14|OpenCLIP-ViT-H-14) FOLDER="Scale" ;;
        ConvNeXt-B) FOLDER="Architecture" ;;
        SigLIP-ViT-B-16) FOLDER="Objective" ;;
        SigLIP2-ViT-B-16) FOLDER="Objective" ;;
        MetaCLIP-ViT-B-16|DataComp-ViT-B-16) FOLDER="Data" ;;
    esac

    python run_experiment.py \
        --dataset      Marine \
        --data_location ./datasets/data/marine \
        --model        "$MODEL" \
        --factors      in,size,amount,location,state \
        --simple_template simple_template \
        --main_template   marine_main_template \
        --factor_templates marine_factor_templates \
        --save_path    "./results/$FOLDER" \
        --save_name    "Marine_${SAFE}" \
        --batch_size   "$BATCH" \
        --n_bootstrap  2000
done