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

# Find the directory containing the class folders
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
    if [ -d "$CLASS_ROOT/$OLD" ]; then
        mv "$CLASS_ROOT/$OLD" "$CLASS_ROOT/${RENAMES[$OLD]}"
    fi
done

echo "Done."