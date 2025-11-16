#!/usr/bin/env Rscript

# Script to regenerate CIVET results for SCENARIO_6_CellCycle only
# This script will regenerate all 4 modes for Scenario 6

# Load required packages
library(tidyverse)
library(Seurat)

# Source your CIVET function and the main processing functions
source("/Users/linxy29/Documents/Code/CIVET/civet_function.R")
source("/Users/linxy29/Documents/Code/civet_analysis/civet_simulation_selection.R")

cat(strrep("=", 80), "\n")
cat("REGENERATING CIVET RESULTS FOR SCENARIO_6_CellCycle\n")
cat(strrep("=", 80), "\n\n")

# Define the scenario directory
scenario_6_dir <- "/Users/linxy29/Documents/Data/CIVET/simulation/SCENARIO_6_CellCycle"

if (!dir.exists(scenario_6_dir)) {
  stop("SCENARIO_6_CellCycle directory not found at: ", scenario_6_dir)
}

# Find all proliferation subdirectories
subruns <- list.dirs(path = scenario_6_dir, full.names = TRUE, recursive = FALSE)

# Filter to actual sub-run directories (which contain "metadata" and "cellSNP" folders)
# Exclude copy directories (those ending with "_cp" or containing "copy")
subruns <- subruns[
  sapply(subruns, function(x) {
    file.exists(file.path(x, "metadata")) &&
    file.exists(file.path(x, "cellSNP")) &&
    !grepl("_cp$|copy", basename(x), ignore.case = TRUE)
  })
]

cat("Found", length(subruns), "simulation runs to process\n\n")

# ============================================================================
# STEP 1: Generate full model results (civet_res/) with base_model = "full"
# ============================================================================
cat(strrep("=", 80), "\n")
cat("STEP 1: Generating FULL MODEL results (civet_res/) with base_model = 'full'\n")
cat(strrep("=", 80), "\n\n")

for (sr in subruns) {
  cat("\n", strrep("-", 80), "\n")
  cat("Processing:", basename(sr), "\n")
  cat(strrep("-", 80), "\n")
  run_supervised_glm_for_subrun(sr, use_extra_covariates = TRUE)
}

# ============================================================================
# STEP 2: Generate generation-only baseline (civet_res_generation_only/)
# ============================================================================
cat("\n\n", strrep("=", 80), "\n")
cat("STEP 2: Generating GENERATION-ONLY baseline (civet_res_generation_only/)\n")
cat(strrep("=", 80), "\n\n")

for (sr in subruns) {
  cat("\n", strrep("-", 80), "\n")
  cat("Processing:", basename(sr), "\n")
  cat(strrep("-", 80), "\n")
  run_supervised_glm_for_subrun(sr, use_extra_covariates = FALSE)
}

# ============================================================================
# STEP 3: Generate cell cycle only results (civet_res_cell_cycle_only/)
# ============================================================================
cat("\n\n", strrep("=", 80), "\n")
cat("STEP 3: Generating CELL CYCLE ONLY results (civet_res_cell_cycle_only/)\n")
cat(strrep("=", 80), "\n\n")

for (sr in subruns) {
  cat("\n", strrep("-", 80), "\n")
  cat("Processing:", basename(sr), "\n")
  cat(strrep("-", 80), "\n")
  run_supervised_glm_scenario6_alternative(sr, mode = "cell_cycle_only")
}

# ============================================================================
# STEP 4: Generate permuted generation control (civet_res_permuted/)
# ============================================================================
cat("\n\n", strrep("=", 80), "\n")
cat("STEP 4: Generating PERMUTED GENERATION control (civet_res_permuted/)\n")
cat(strrep("=", 80), "\n\n")

for (sr in subruns) {
  cat("\n", strrep("-", 80), "\n")
  cat("Processing:", basename(sr), "\n")
  cat(strrep("-", 80), "\n")
  run_supervised_glm_scenario6_alternative(sr, mode = "permuted_generation")
}

# ============================================================================
# SUMMARY
# ============================================================================
cat("\n\n", strrep("=", 80), "\n")
cat("REGENERATION COMPLETE!\n")
cat(strrep("=", 80), "\n\n")

cat("Successfully regenerated CIVET results for SCENARIO_6_CellCycle:\n")
cat("  ✓ civet_res/                     - Full model (generation + cell_cycle) with base_model='full'\n")
cat("  ✓ civet_res_generation_only/     - Generation only (baseline)\n")
cat("  ✓ civet_res_cell_cycle_only/     - Cell cycle only\n")
cat("  ✓ civet_res_permuted/            - Permuted generation (negative control)\n\n")

cat("Processed", length(subruns), "simulation runs\n")
cat(strrep("=", 80), "\n")
