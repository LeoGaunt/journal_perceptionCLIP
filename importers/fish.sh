#!/bin/bash

set -e

cd ./datasets/data || { echo "Directory not found"; exit 1; }

mkdir -p fish

# Download dataset from Kaggle and unzip it
echo "Downloading dataset..."
curl -L -o fish.zip https://www.kaggle.com/api/v1/datasets/download/markdaniellampa/fish-dataset
echo "Unzipping dataset..."
unzip -q fish.zip -d fish_raw

# Get folder
cd fish_raw/FishImgDataset || { echo "Dataset structure not found"; exit 1; }

before_src=$(find train test val -mindepth 2 -maxdepth 2 -type f ! -name '.*' | wc -l)

# Loop through train, test, val
for split in train test val; do
    for class_dir in "$split"/*; do
        class_name=$(basename "$class_dir")
        mkdir -p ../../fish/"$class_name"
        for img in "$class_dir"/*; do
            cp "$img" ../../fish/"$class_name"/"${split}_$(basename "$img")"
        done
    done
done

after_dst=$(find ../../fish -type f | wc -l)
echo "source files:      $before_src"
echo "dest before:       $after_dst"

echo "Cleaning up..."
cd ../../
rm -rf fish_raw fish.zip

echo "✅ Dataset ready in ./datasets/data/fish"