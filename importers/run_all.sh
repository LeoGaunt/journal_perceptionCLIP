#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================="
echo "Running all dataset importers"
echo "=============================="

bash "$DIR/eurosat.sh"
echo "✓ EuroSAT done"

bash "$DIR/ucm.sh"
echo "✓ UCM done"

bash "$DIR/flowers102.sh"
echo "✓ Flowers-102 done"

bash "$DIR/oxford_pets.sh"
echo "✓ Oxford-IIIT Pets done"

bash "$DIR/cub200.sh"
echo "✓ CUB-200-2011 done"

bash "$DIR/dissertation.sh"
echo "✓ Dissertation done"

bash "$DIR/fish.sh"
echo "✓ Fish done"

bash "$DIR/marine.sh"
echo "✓ Marine done"

echo "=============================="
echo "All datasets installed successfully"
echo "=============================="