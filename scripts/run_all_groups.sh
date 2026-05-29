#!/bin/bash
# ============================================================
# Master script — runs all four comparison groups in sequence.
# Expected total runtime on A100: ~12-18 hours.
#
# Run individual groups independently if you want to parallelise
# across multiple Colab sessions:
#   bash scripts/run_scale_group.sh
#   bash scripts/run_architecture_group.sh
#   bash scripts/run_objective_group.sh
#   bash scripts/run_data_group.sh
#
# Usage: bash scripts/run_all_groups.sh
# ============================================================
export PYTHONPATH="$PYTHONPATH:$PWD"

SCRIPT_DIR="$(dirname "$0")"

echo "================================================"
echo "  TPAMI Experiment Suite — All Comparison Groups"
echo "================================================"
echo ""

echo "[1/4] Scale group"
bash "$SCRIPT_DIR/run_scale_group.sh"
echo ""

echo "[2/4] Architecture group"
bash "$SCRIPT_DIR/run_architecture_group.sh"
echo ""

echo "[3/4] Objective group"
bash "$SCRIPT_DIR/run_objective_group.sh"
echo ""

echo "[4/4] Data group"
bash "$SCRIPT_DIR/run_data_group.sh"
echo ""

echo "================================================"
echo "  All groups complete."
echo "  Results saved to ./results/{scale,architecture,objective,data}/"
echo "================================================"
