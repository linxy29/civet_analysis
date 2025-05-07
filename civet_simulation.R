#!/usr/bin/env Rscript

# Load required packages
library(tidyverse)
library(Seurat)    # for ReadMtx, if you prefer
# Or library(Matrix) if you want readMM, etc.

# Source your CIVET function
source("./civet_function_v2.R")

################################################################################
# 1) Function to read mtx safely
################################################################################
read_mtx_safe <- function(mtx, mutations, barcodes, feature.column = 1) {
  # A simple wrapper around Seurat::ReadMtx (or your read code).
  # Modify if your actual environment differs.
  if (!file.exists(mtx)) {
    stop("Matrix file does not exist: ", mtx)
  }
  if (!file.exists(barcodes)) {
    stop("Barcode file does not exist: ", barcodes)
  }
  # If you have a features file, pass it here; otherwise pass NULL.
  # For demonstration, let's assume no separate 'features' file is needed.
  mat <- ReadMtx(
    mtx = mtx,
    features = barcodes,     # or your variants file if needed
    cells = mutations,
    feature.column = feature.column
  )
  return(mat)
}

################################################################################
# 2) Function to run supervised_glm on a single sub-run folder
################################################################################
run_supervised_glm_for_subrun <- function(subrun_dir) {
  # subrun_dir example:
  #   SCENARIO_1_Mutation_Rate/mutation_rate_1_20250403_152908
  
  # Paths we expect
  metadata_csv <- file.path(subrun_dir, "metadata", "simulation_cell_metadata.csv")
  ad_mtx_path  <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.AD.mtx")
  dp_mtx_path  <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.DP.mtx")
  barcodes_txt <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.barcodes.txt")
  mutations_txt <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.mutations.txt")
  
  # Check if mandatory files exist. If not, skip.
  if (!all(file.exists(metadata_csv, ad_mtx_path, dp_mtx_path, barcodes_txt, mutations_txt))) {
    # check which file is missing
    missing_files <- c(
      metadata_csv,
      ad_mtx_path,
      dp_mtx_path,
      barcodes_txt,
      mutations_txt
    )[!file.exists(c(metadata_csv, ad_mtx_path, dp_mtx_path, barcodes_txt, mutations_txt))]
    # Print a message and skip this subrun
    message("Missing files in ", subrun_dir, ": ", paste(missing_files, collapse = ", "))
    message("Skipping ", subrun_dir, " because required files not found.")
    return(NULL)
  }

  
  # Read metadata to build clone_mat
  metadata <- read_csv(metadata_csv)
  max_gene <- max(metadata$generation)
  
  # Example: pick the 3 columns from "generation" or "stage" columns
  # If your columns are exactly 'Tumor_KC_Basal', 'Tumor_KC_Cyc', 'Tumor_KC_Diff', do:
  clone_mat <- metadata %>%
    rename(cell_id = 1) %>%     # rename first column to cell_id if needed
    column_to_rownames("cell_id") %>%
    dplyr::select(generation)
  
  # Read AD/DP
  AD_mtx <- read_mtx_safe(ad_mtx_path, mutations_txt, barcodes_txt)
  DP_mtx <- read_mtx_safe(dp_mtx_path, mutations_txt, barcodes_txt)
  
  # Subset the AD/DP to the common barcodes in clone_mat
  common_barcodes <- intersect(rownames(clone_mat), colnames(AD_mtx))
  subset_AD  <- AD_mtx[, common_barcodes, drop = FALSE]
  subset_DP  <- DP_mtx[, common_barcodes, drop = FALSE]
  subset_clones <- clone_mat[common_barcodes, , drop = FALSE]
  
  # Run the supervised_glm
  cat("Running supervised_glm on", subrun_dir, "with", ncol(subset_AD), "cells...\n")
  res <- supervised_glm(
    AD_mat          = subset_AD,
    DP_mat          = subset_DP,
    clone_mat       = subset_clones,
    minDP           = 5,
    use_random_effect = FALSE
  )
  
  # Combine and write out results
  resDF <- purrr::imap_dfr(
    res,
    ~ as.data.frame(.x) %>%
      tibble::rownames_to_column("variant") %>%
      mutate(value = .y)
  )
  
  # Make an output directory "civet_res" inside subrun_dir
  outdir <- file.path(subrun_dir, "civet_res")
  if (!dir.exists(outdir)) dir.create(outdir)
  
  out_rds  <- file.path(outdir, "civet_results.rds")
  out_csv  <- file.path(outdir, "civet_results.csv")
  
  saveRDS(res, file = out_rds)
  write.csv(resDF, file = out_csv, row.names = FALSE)
  
  cat("Finished supervised_glm for", subrun_dir, "\n")
  
  return(invisible(TRUE))
}

################################################################################
# 3) Main driver: find SCENARIO_ folders, then sub-run directories
################################################################################
main <- function() {
  # List all SCENARIO_ directories in the current working directory
  scenario_dirs <- list.dirs(path = "/home/linxy29/data/CIVET/simulation/", full.names = TRUE, recursive = FALSE)
  scenario_dirs <- scenario_dirs[grepl("^.*/SCENARIO_", scenario_dirs)]
  
  # For each SCENARIO_..., find sub-run directories
  for (scen in scenario_dirs) {
    # sub-run directories are the immediate children of scen
    # e.g. "mutation_rate_1_20250403_152908", "mutation_rate_2_..."
    subruns <- list.dirs(path = scen, full.names = TRUE, recursive = FALSE)
    
    # We only want actual sub-run directories (which contain "metadata" or "cellSNP" folder)
    # Filter them:
    subruns <- subruns[
      sapply(subruns, function(x) {
        file.exists(file.path(x, "metadata")) && file.exists(file.path(x, "cellSNP"))
      })
    ]
    
    # Run the supervised_glm steps for each sub-run
    for (sr in subruns) {
      run_supervised_glm_for_subrun(sr)
    }
  }
  
  cat("All SCENARIO_ runs completed.\n")
}

################################################################################
# 4) Execute main
################################################################################
if (interactive()) {
  # If running in an interactive session, just call main()
  main()
} else {
  # If running as a script via Rscript, call main
  main()
}
