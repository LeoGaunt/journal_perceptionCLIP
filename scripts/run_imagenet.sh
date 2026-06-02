#!/bin/bash
# ============================================================
# run_imagenet.sh  —  ImageNet experiments across all four
# comparison groups (Scale, Architecture, Data, Objective).
#
# Mirrors the structure of the per-dataset group scripts but
# runs ImageNet only, saving into the same results/ tree so
# results are directly comparable to other datasets.
#
# Usage
# -----
#   bash scripts/run_imagenet.sh              # all four groups
#   bash scripts/run_imagenet.sh --scale      # scale only
#   bash scripts/run_imagenet.sh --arch       # architecture only
#   bash scripts/run_imagenet.sh --data       # data only
#   bash scripts/run_imagenet.sh --objective  # objective only
#
# Prerequisites
# -------------
#   1. ImageNet val is set up:
#        datasets/data/imagenet/val/<synset_dirs>
#      Run `bash importers/imagenet.sh <user> <key>` if not done.
#
#   2. src/datasets/imagenet.py and src/templates/imagenet_template.py
#      are present (shipped alongside this script).
#
# Output
# ------
#   results/
#   ├── scale/ImageNet/
#   ├── architecture/ImageNet/
#   ├── data/ImageNet/
#   └── objective/ImageNet/
#
# Estimated runtime on A100  (50k val images, 1000 classes):
#   Each model × 3 conditions ≈ 8–15 min  →  total ≈ 3–5 hours
# ============================================================

set -e
export PYTHONPATH="$PYTHONPATH:$PWD"

# ── Shared ImageNet config ───────────────────────────────────────────────────
DATASET="ImageNet"
DATA_LOCATION="./datasets/data/imagenet"
FACTORS="orientation,background,quality,illumination,quantity,perspective,art,medium,condition,color_scheme,tool"
MAIN_TEMPLATE="imagenet_main_template"
FACTOR_TEMPLATE="imagenet_factor_templates"

# ── Argument parsing — default to all groups ────────────────────────────────
RUN_SCALE=false
RUN_ARCH=false
RUN_DATA=false
RUN_OBJ=false
RUN_ALL=true

for arg in "$@"; do
    case "$arg" in
        --scale)     RUN_SCALE=true; RUN_ALL=false ;;
        --arch)      RUN_ARCH=true;  RUN_ALL=false ;;
        --data)      RUN_DATA=true;  RUN_ALL=false ;;
        --objective) RUN_OBJ=true;   RUN_ALL=false ;;
    esac
done

if $RUN_ALL; then
    RUN_SCALE=true; RUN_ARCH=true; RUN_DATA=true; RUN_OBJ=true
fi

# ── Helper ───────────────────────────────────────────────────────────────────
run_experiment() {
    local GROUP="$1"
    local MODEL="$2"
    local BATCH="$3"
    local SAFE
    SAFE=$(echo "$MODEL" | tr '/@ ' '---')

    echo "========================================================"
    echo "  $GROUP | ImageNet | $MODEL"
    echo "========================================================"

    python run_experiment.py \
        --dataset          "$DATASET" \
        --data_location    "$DATA_LOCATION" \
        --model            "$MODEL" \
        --factors          "$FACTORS" \
        --simple_template  simple_template \
        --main_template    "$MAIN_TEMPLATE" \
        --factor_templates "$FACTOR_TEMPLATE" \
        --save_path        "./results/$(echo "$GROUP" | tr '[:upper:]' '[:lower:]')/ImageNet" \
        --save_name        "ImageNet_${SAFE}" \
        --batch_size       "$BATCH" \
        --n_bootstrap      2000

    echo ""
}

# ── Pre-flight check ─────────────────────────────────────────────────────────
VAL_DIR="$DATA_LOCATION/val"
if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: ImageNet val not found at $VAL_DIR"
    echo "       Run: bash importers/imagenet.sh <username> <access_key>"
    exit 1
fi
N_CLASSES=$(ls "$VAL_DIR" | wc -l | tr -d ' ')
if [ "$N_CLASSES" -lt 1000 ]; then
    echo "ERROR: Expected 1000 class folders in $VAL_DIR, found $N_CLASSES"
    echo "       The val directory may not be fully organised into synset subdirs."
    exit 1
fi
echo "✓ ImageNet val: $N_CLASSES class folders found"
echo ""

# ════════════════════════════════════════════════════════════
# SCALE GROUP
# ViT-B-16 (86M) | ViT-L-14 (307M) | ViT-H-14 (632M)
# All LAION-2B, softmax, transformer — only parameter count differs.
# ════════════════════════════════════════════════════════════
if $RUN_SCALE; then
    echo "════════════════════════════════════════════════════════"
    echo "  [1/4] SCALE GROUP"
    echo "════════════════════════════════════════════════════════"
    echo ""

    run_experiment "Scale" "OpenCLIP-ViT-B-16" 1024
    run_experiment "Scale" "OpenCLIP-ViT-L-14" 512
    run_experiment "Scale" "OpenCLIP-ViT-H-14" 512

    echo "✓ Scale group complete → ./results/scale/ImageNet/"
    echo ""
fi

# ════════════════════════════════════════════════════════════
# ARCHITECTURE GROUP
# ViT-B-16 (transformer) | ConvNeXt-B (modern CNN)
# All LAION-2B, softmax, ~88M params — only encoder arch differs.
# ════════════════════════════════════════════════════════════
if $RUN_ARCH; then
    echo "════════════════════════════════════════════════════════"
    echo "  [2/4] ARCHITECTURE GROUP"
    echo "════════════════════════════════════════════════════════"
    echo ""

    run_experiment "Architecture" "OpenCLIP-ViT-B-16" 1024
    run_experiment "Architecture" "ConvNeXt-B"        256

    echo "✓ Architecture group complete → ./results/architecture/ImageNet/"
    echo ""
fi

# ════════════════════════════════════════════════════════════
# DATA GROUP
# OpenCLIP-ViT-B-16  LAION-2B         (large-scale web data)
# MetaCLIP-ViT-B-16  MetaCLIP-400M    (curation matching WIT)
# DataComp-ViT-B-16  DataComp-XL      (quality-filtered curation)
# All ViT-B-16 arch, softmax — only training data differs.
# ════════════════════════════════════════════════════════════
if $RUN_DATA; then
    echo "════════════════════════════════════════════════════════"
    echo "  [3/4] DATA GROUP"
    echo "════════════════════════════════════════════════════════"
    echo ""

    run_experiment "Data" "OpenCLIP-ViT-B-16"  1024
    run_experiment "Data" "MetaCLIP-ViT-B-16"  1024
    run_experiment "Data" "DataComp-ViT-B-16"  1024

    echo "✓ Data group complete → ./results/data/ImageNet/"
    echo ""
fi

# ════════════════════════════════════════════════════════════
# OBJECTIVE GROUP
# OpenCLIP-ViT-B-16  LAION-2B   softmax contrastive
# SigLIP-ViT-B-16    WebLI      sigmoid pairwise (v1)
# SigLIP2-ViT-B-16   WebLI      sigmoid pairwise (v2)
# All ViT-B-16 scale — only training objective differs.
# (Training data differs LAION vs WebLI — unavoidable constraint,
#  acknowledged in paper; no model with same data but diff objective exists.)
# ════════════════════════════════════════════════════════════
if $RUN_OBJ; then
    echo "════════════════════════════════════════════════════════"
    echo "  [4/4] OBJECTIVE GROUP"
    echo "════════════════════════════════════════════════════════"
    echo ""

    run_experiment "Objective" "OpenCLIP-ViT-B-16" 1024
    run_experiment "Objective" "SigLIP-ViT-B-16"   1024
    run_experiment "Objective" "SigLIP2-ViT-B-16"  1024

    echo "✓ Objective group complete → ./results/objective/ImageNet/"
    echo ""
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  ImageNet experiments complete."
echo "  Results saved to:"
$RUN_SCALE && echo "    ./results/scale/ImageNet/"
$RUN_ARCH  && echo "    ./results/architecture/ImageNet/"
$RUN_DATA  && echo "    ./results/data/ImageNet/"
$RUN_OBJ   && echo "    ./results/objective/ImageNet/"
echo "════════════════════════════════════════════════════════"
