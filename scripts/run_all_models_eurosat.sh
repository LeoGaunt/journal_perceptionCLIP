#!/bin/bash
# Run EuroSAT across all model families for the TPAMI cross-architecture study.
# Usage: bash scripts/run_all_models_eurosat.sh

export PYTHONPATH="$PYTHONPATH:$PWD"

DATASET="EuroSAT"
DATA="./datasets/data"
FACTORS="condition,source"
MAIN="eurosat_main_template"
FACTOR_T="eurosat_factor_templates"
SIMPLE="simple_template"

for MODEL in \
    "ViT-B/16:512" \
    "ViT-L/14@336px:256" \
    "RN50x4:256" \
    "OpenCLIP-ViT-H-14:128" \
    "MetaCLIP-ViT-B-16:512" \
    "SigLIP2-ViT-B-16:512"
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
        --save_path "./results/eurosat" \
        --save_name "eurosat_${SAFE}" \
        --batch_size "$BATCH" \
        --n_bootstrap 2000
done

echo "All EuroSAT runs complete."
