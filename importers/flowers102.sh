#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/datasets/data/flowers-102"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# Download images, labels, and splits
curl -L -o 102flowers.tgz \
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

curl -L -o imagelabels.mat \
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

curl -L -o setid.mat \
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

# Extract images (creates jpg/ directory)
tar -xzf 102flowers.tgz

# Clean up archive
rm 102flowers.tgz