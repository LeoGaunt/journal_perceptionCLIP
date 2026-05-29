#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets/data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Install gdown if needed
python3 -m pip install -q gdown

# Download from Google Drive
gdown "https://drive.google.com/uc?id=1GvlmhycXT_vdQKTIa8iBAxM9c3aUMlYs" -O dissertation.zip

# Extract and rename
unzip dissertation.zip
mv dataset dissertation
rm dissertation.zip