#!/bin/bash
# Run Fish dataset across all model families.
export PYTHONPATH="$PYTHONPATH:$PWD"

DATASET="Fish"
DATA="./datasets/data/fish"
FACTORS="in,size,amount,location,liveness"
MAIN="fish_main_template"
FACTOR_T="fish_factor_templates"
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
        --save_path "./results/fish" \
        --save_name "fish_${SAFE}" \
        --batch_size "$BATCH" \
        --n_bootstrap 2000
done

echo "All Fish runs complete."
