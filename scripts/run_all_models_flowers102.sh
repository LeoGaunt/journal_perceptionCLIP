#!/bin/bash
# Run Flowers-102 across all model families for the TPAMI cross-architecture study.
# Usage: bash scripts/run_all_models_flowers102.sh

export PYTHONPATH="$PYTHONPATH:$PWD"

DATASET="Flowers102"
DATA="./datasets/data/flowers102"
MAIN="flowers102_main_template"
FACTOR_T="flowers102_factor_templates"
SIMPLE="simple_template"

for MODEL in \
    "ViT-B/16:512" \
    "ViT-L/14@336px:256" \
    "RN50x4:256" \
    "OpenCLIP-ViT-H-14:128" \
    "MetaCLIP-ViT-B-16:512" \
    "SigLIP-ViT-B-16:512"
do
    MODEL_NAME="${MODEL%%:*}"
    BATCH="${MODEL##*:}"
    SAFE=$(echo "$MODEL_NAME" | tr '/@ ' '---')

    python run_experiment.py \
        --dataset "$DATASET" \
        --data_location "$DATA" \
        --model "$MODEL_NAME" \
        --factors "$FACTORS" \
        --simple_template "$SIMPLE" \
        --main_template "$MAIN" \
        --factor_templates "$FACTOR_T" \
        --save_path "./results/flowers102" \
        --save_name "flowers102_${SAFE}" \
        --batch_size "$BATCH" \
        --n_bootstrap 2000
done

echo "All Flowers102 runs complete."
