#!/bin/bash
set -e

# Repository root (parent of importers/)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DATASET_DIR="$ROOT_DIR/datasets/data/eurosat"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

wget https://madm.dfki.de/files/sentinel/EuroSAT.zip --no-check-certificate
unzip EuroSAT.zip
rm EuroSAT.zip