#!/bin/bash
# Script to run mquad on simulation data
# Usage: bash run_mquad_simulation.sh [base_dir]

# Activate mquad conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mquad

# Default base directory
BASE_DIR="${1:-/Users/linxy29/Documents/Data/CIVET/simulation}"

# Number of threads
THREADS=20

echo "Running mquad analysis on simulation data"
echo "Using conda environment: mquad"
echo "Base directory: $BASE_DIR"
echo "Threads: $THREADS"
echo ""

# Counter for tracking
total=0
success=0
failed=0

# Function to run mquad on a single directory
run_mquad() {
    local sim_dir=$1
    local condition_name=$(basename "$sim_dir")
    local scenario_name=$(basename $(dirname "$sim_dir"))

    echo "=========================================="
    echo "Processing: $scenario_name / $condition_name"
    echo "=========================================="

    # Check if cellSNP folder exists
    if [ ! -d "$sim_dir/cellSNP" ]; then
        echo "ERROR: cellSNP folder not found in $sim_dir"
        ((failed++))
        return 1
    fi

    # Check if mquad_out already exists
    if [ -d "$sim_dir/mquad_out" ]; then
        echo "SKIP: mquad_out already exists"
        return 0
    fi

    # Run mquad using mtxData format
    echo "Running: mquad -m $sim_dir/cellSNP/cellSNP.tag.AD.mtx,$sim_dir/cellSNP/cellSNP.tag.DP.mtx -o $sim_dir/mquad_out -p $THREADS"

    if mquad -m "$sim_dir/cellSNP/cellSNP.tag.AD.mtx,$sim_dir/cellSNP/cellSNP.tag.DP.mtx" -o "$sim_dir/mquad_out" -p $THREADS; then
        echo "SUCCESS: mquad completed for $condition_name"
        ((success++))
    else
        echo "ERROR: mquad failed for $condition_name"
        ((failed++))
        return 1
    fi

    echo ""
}

# Process SCENARIO_4_CellCycle
echo "############################################"
echo "SCENARIO_4_CellCycle"
echo "############################################"
for condition in "$BASE_DIR/SCENARIO_4_CellCycle"/*; do
    if [ -d "$condition" ] && [ -d "$condition/cellSNP" ]; then
        ((total++))
        run_mquad "$condition"
    fi
done

# Process SCENARIO_5_Metabolic
echo "############################################"
echo "SCENARIO_5_Metabolic"
echo "############################################"
for condition in "$BASE_DIR/SCENARIO_5_Metabolic"/*; do
    if [ -d "$condition" ] && [ -d "$condition/cellSNP" ]; then
        ((total++))
        run_mquad "$condition"
    fi
done

# Process SCENARIO_6_VaryingDepth
echo "############################################"
echo "SCENARIO_6_VaryingDepth"
echo "############################################"
for condition in "$BASE_DIR/SCENARIO_6_VaryingDepth"/*; do
    if [ -d "$condition" ] && [ -d "$condition/cellSNP" ]; then
        ((total++))
        run_mquad "$condition"
    fi
done

# Print summary
echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Total conditions processed: $total"
echo "Successful: $success"
echo "Failed: $failed"
echo "Skipped: $((total - success - failed))"
echo "=============================================="
