#!/bin/bash
set +e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/datasets/data/ucm"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading UCM dataset..."
curl -L \
    -o ucm.zip \
    https://www.kaggle.com/api/v1/datasets/download/abdulhasibuddin/uc-merced-land-use-dataset

echo "Extracting..."
unzip -q ucm.zip
rm ucm.zip

EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "*UCMerced*" | head -n 1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "Could not find extracted dataset folder"
    exit 1
fi

echo "Flattening dataset structure..."

shopt -s dotglob
mv "$EXTRACTED_DIR"/* "$DATASET_DIR"/ 2>/dev/null || true
rmdir "$EXTRACTED_DIR" 2>/dev/null || true

CLASS_ROOT=$(find . -type d -name "Images" | head -n 1)

if [ -z "$CLASS_ROOT" ]; then
    echo "Could not find Images directory."
    exit 1
fi

declare -A RENAMES=(
    ["agricultural"]="agricultural land"
    ["airplane"]="airplane(s)"
    ["baseballdiamond"]="baseball diamond"
    ["beach"]="beach"
    ["buildings"]="buildings"
    ["chaparral"]="chaparral"
    ["denseresidential"]="dense residential"
    ["forest"]="forest"
    ["freeway"]="freeway"
    ["golfcourse"]="golf course"
    ["harbor"]="harbor"
    ["intersection"]="intersection"
    ["mediumresidential"]="medium residential"
    ["mobilehomepark"]="mobile home park"
    ["overpass"]="overpass"
    ["parkinglot"]="parking lot"
    ["river"]="river"
    ["runway"]="runway"
    ["sparseresidential"]="sparse residential"
    ["storagetanks"]="storage tanks"
    ["tenniscourt"]="tennis court"
)

echo "Renaming class folders..."

for OLD in "${!RENAMES[@]}"; do
    SRC="$CLASS_ROOT/$OLD"
    DST="$CLASS_ROOT/${RENAMES[$OLD]}"

    if [ -d "$SRC" ]; then
        mv "$SRC" "$DST" 2>/dev/null || true
    fi
done

echo "Done."
echo "Final structure:"
ls "$CLASS_ROOT"