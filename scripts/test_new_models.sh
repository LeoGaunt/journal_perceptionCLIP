#!/bin/bash
# Quick check that all new models load without error.
# Runs EuroSAT Simple condition only (fast, no +Z) for each new model.
# Usage: bash scripts/test_new_models.sh
export PYTHONPATH="$PYTHONPATH:$PWD"

NEW_MODELS=(
    "ConvNeXt-B:512"
    "SigLIP-ViT-B-16:512"
    "DataComp-ViT-B-16:512"
)

for MODEL_SPEC in "${NEW_MODELS[@]}"; do
    MODEL="${MODEL_SPEC%%:*}"
    BATCH="${MODEL_SPEC##*:}"
    SAFE=$(echo "$MODEL" | tr '/@ ' '---')
    echo "--- Testing: $MODEL ---"
    python run_experiment.py \
        --dataset      EuroSAT \
        --data_location ./datasets/data \
        --model        "$MODEL" \
        --factors      condition,source \
        --simple_template simple_template \
        --main_template   eurosat_main_template \
        --factor_templates eurosat_factor_templates \
        --save_path    ./results/smoke_test \
        --save_name    "smoke_${SAFE}" \
        --batch_size   "$BATCH" \
        --n_bootstrap  100
    echo ""
done

echo "Smoke test complete. Check ./results/smoke_test/ for outputs."
