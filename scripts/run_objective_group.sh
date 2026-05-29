#!/bin/bash
# ============================================================
# Objective group — all ViT-B-16 scale.
# Only training loss / objective changes.
#   OpenCLIP-ViT-B-16  LAION-2B   softmax contrastive
#   SigLIP-ViT-B-16    WebLI      sigmoid pairwise (v1)
#   SigLIP2-ViT-B-16   WebLI      sigmoid pairwise (v2)
#
# Note: training data differs (LAION vs WebLI) — this is an
# unavoidable limitation acknowledged in the paper; no model
# exists with identical training data but different objective.
#
# Usage: bash scripts/run_objective_group.sh
# ============================================================
export PYTHONPATH="$PYTHONPATH:$PWD"

MODELS=(
    "OpenCLIP-ViT-B-16:512"
    "SigLIP-ViT-B-16:512"
    "SigLIP2-ViT-B-16:512"
)

DATASETS=(
    "EuroSAT|./datasets/data|condition,source|eurosat_main_template|eurosat_factor_templates"
    "Fish|./datasets/data/fish|in,size,amount,location,liveness|fish_main_template|fish_factor_templates"
    "Marine|./datasets/data/marine|in,size,amount,location,state|marine_main_template|marine_factor_templates"
    "UCM|./datasets/data/ucm|condition,source,land|ucm_main_template|ucm_factor_templates"
    "Flowers102|./datasets/data|background,illumination,petals,quality|flowers102_main_template|flowers102_factor_templates"
    "OxfordPets|./datasets/data|species,background,pose,interaction|oxfordpets_main_template|oxfordpets_factor_templates"
    "CUB200|./datasets/data|size,background,condition|cub200_main_template|cub200_factor_templates"
    "Dissertation|./datasets/data/dissertation|on,direction,amount,weather,decker|dissertation_main_template|dissertation_factor_templates"
)

for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET DATA FACTORS MAIN FACTOR_T <<< "$ENTRY"
    for MODEL_SPEC in "${MODELS[@]}"; do
        MODEL="${MODEL_SPEC%%:*}"
        BATCH="${MODEL_SPEC##*:}"
        SAFE=$(echo "$MODEL" | tr '/@ ' '---')
        echo "========================================"
        echo "  Objective | Dataset: $DATASET | Model: $MODEL"
        echo "========================================"
        python run_experiment.py \
            --dataset      "$DATASET" \
            --data_location "$DATA" \
            --model        "$MODEL" \
            --factors      "$FACTORS" \
            --simple_template simple_template \
            --main_template   "$MAIN" \
            --factor_templates "$FACTOR_T" \
            --save_path    "./results/objective/$DATASET" \
            --save_name    "${DATASET}_${SAFE}" \
            --batch_size   "$BATCH" \
            --n_bootstrap  2000
        echo ""
    done
done

echo "Objective group complete."
