#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets/data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading CUB-200-2011..."

curl -L -o CUB_200_2011.tgz \
    https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

echo "Extracting..."
tar -xzf CUB_200_2011.tgz

rm CUB_200_2011.tgz

echo "Dataset installed to $DATA_DIR/CUB_200_2011"