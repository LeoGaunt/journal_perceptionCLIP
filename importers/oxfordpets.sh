#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/datasets/data/oxford-iiit-pet"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading Oxford-IIIT Pet dataset..."

curl -L -o images.tar.gz \
    https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz

curl -L -o annotations.tar.gz \
    https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

echo "Extracting images..."
tar -xzf images.tar.gz

echo "Extracting annotations..."
tar -xzf annotations.tar.gz

rm images.tar.gz
rm annotations.tar.gz

echo "Dataset installed to $DATASET_DIR"