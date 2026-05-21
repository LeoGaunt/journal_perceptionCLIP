#!/bin/bash
# Run Fish dataset across all model families.
export PYTHONPATH="$PYTHONPATH:$PWD"

DATASET="Fish"
DATA="./datasets/data/fish"
FACTORS="in,size,amount,location,liveness"
MAIN="fish_main_template"
FACTOR_T="fish_factor_templates"
SIMPLE="fish_template"

for MODEL in \
    "ViT-B/16" \
    "ViT-L/14@336px" \
    "RN50x4" \
    "OpenCLIP-ViT-H-14" \
    "MetaCLIP-ViT-B-16" \
    "SigLIP-ViT-B-16"
do
    SAFE=$(echo "$MODEL" | tr '/@ ' '---')
    echo "======== $MODEL ========"
    python run_experiment.py \
        --dataset "$DATASET" \
        --data_location "$DATA" \
        --model "$MODEL" \
        --factors "$FACTORS" \
        --simple_template "$SIMPLE" \
        --main_template "$MAIN" \
        --factor_templates "$FACTOR_T" \
        --save_path "./results/fish" \
        --save_name "fish_${SAFE}" \
        --n_bootstrap 2000
    echo ""
done
