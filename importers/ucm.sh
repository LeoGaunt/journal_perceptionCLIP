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

IMAGES_DIR=$(find . -type d -name "Images" | head -n 1)

if [ -z "$IMAGES_DIR" ]; then
    echo "Could not find Images directory"
    exit 1
fi

echo "Found Images folder: $IMAGES_DIR"
echo "Flattening dataset into $DATASET_DIR..."

shopt -s dotglob

for class_dir in "$IMAGES_DIR"/*/; do
    class_name=$(basename "$class_dir")

    echo "Moving $class_name"

    # move folder up to dataset root
    mv "$class_dir" "$DATASET_DIR/" 2>/dev/null || true
done

rm -rf "$IMAGES_DIR" 2>/dev/null || true

# remove top-level wrapper if it still exists
EXTRACTED_ROOT=$(find . -maxdepth 1 -type d -name "*UCMerced*" | head -n 1)
rm -rf "$EXTRACTED_ROOT" 2>/dev/null || true

echo "Done."
echo "Final structure:"
ls "$DATASET_DIR"