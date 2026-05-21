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
    "ViT-B/16" \
    "ViT-L/14@336px" \
    "RN50x4" \
    "OpenCLIP-ViT-H-14" \
    "MetaCLIP-ViT-B-16" \
    "SigLIP-ViT-B-16"
do
    # Sanitise model name for use in filenames
    SAFE=$(echo "$MODEL" | tr '/@ ' '---')
    echo "========================================"
    echo "  Model: $MODEL"
    echo "========================================"

    python run_experiment.py \
        --dataset "$DATASET" \
        --data_location "$DATA" \
        --model "$MODEL" \
        --factors "$FACTORS" \
        --simple_template "$SIMPLE" \
        --main_template "$MAIN" \
        --factor_templates "$FACTOR_T" \
        --save_path "./results/eurosat" \
        --save_name "eurosat_${SAFE}" \
        --n_bootstrap 2000

    echo ""
done

echo "All EuroSAT runs complete."
