#!/usr/bin/env Rscript

# Load required packages
library(tidyverse)
library(Seurat)    # for ReadMtx, if you prefer
# Or library(Matrix) if you want readMM, etc.

# Source your CIVET function
source("/Users/linxy29/Documents/Code/CIVET/civet_function.R")

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
    features = mutations,     # or your variants file if needed
    cells = barcodes,
    feature.column = feature.column
  )
  return(mat)
}

################################################################################
# 2) Function to run supervised_glm on a single sub-run folder
################################################################################
run_supervised_glm_for_subrun <- function(subrun_dir, use_extra_covariates = TRUE) {
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

  # Detect scenario from directory path
  scenario_name <- basename(dirname(subrun_dir))

  # Select covariates based on scenario and use_extra_covariates flag
  if (use_extra_covariates && grepl("SCENARIO_6", scenario_name, ignore.case = TRUE)) {
    # Scenario 6: Cell Cycle - use generation + cell_cycle_phase
    cat("Detected SCENARIO_6 (Cell Cycle) - using generation + cell_cycle_potential\n")

    # Convert cell_cycle_phase to differentiation potential
    # Higher values = higher proliferative/stem-like potential
    # S (replicating) = 4, G2/M (dividing) = 3, G1 (entering cycle) = 2, G0 (quiescent) = 1
    metadata <- metadata %>%
      mutate(cell_cycle_potential = case_when(
        cell_cycle_phase == "S"  ~ 4,  # Highest potential (actively replicating DNA)
        cell_cycle_phase == "G2" ~ 3,  # High potential (preparing to divide)
        cell_cycle_phase == "M"  ~ 3,  # High potential (actively dividing)
        cell_cycle_phase == "G1" ~ 2,  # Moderate potential (entering cell cycle)
        cell_cycle_phase == "G0" ~ 1,  # Lowest potential (quiescent/differentiated)
        TRUE ~ 2  # default to G1
      ))

    clone_mat <- metadata %>%
      rename(cell_id = 1) %>%
      column_to_rownames("cell_id") %>%
      dplyr::select(generation, cell_cycle_potential)

  } else if (use_extra_covariates && grepl("SCENARIO_5", scenario_name, ignore.case = TRUE)) {
    # Scenario 5: Metabolic - use generation + metabolic_state
    cat("Detected SCENARIO_5 (Metabolic) - using generation + metabolic_potential\n")

    # Convert metabolic_state to differentiation potential
    # Higher values = higher stem-like potential
    # Glycolytic (stem) = 3, OXPHOS_low (progenitor) = 2, OXPHOS_high (terminal) = 1
    metadata <- metadata %>%
      mutate(metabolic_potential = case_when(
        metabolic_state == "Glycolytic"   ~ 3,  # Highest potential (stem cells)
        metabolic_state == "OXPHOS_low"   ~ 2,  # Moderate potential (progenitors)
        metabolic_state == "OXPHOS_high"  ~ 1,  # Lowest potential (terminally differentiated)
        TRUE ~ 2  # default to OXPHOS_low
      ))

    clone_mat <- metadata %>%
      rename(cell_id = 1) %>%
      column_to_rownames("cell_id") %>%
      dplyr::select(generation, metabolic_potential)

  } else {
    # Scenarios 1, 2, 3, 4 OR scenarios 5, 6 without extra covariates: use generation only
    if (!use_extra_covariates && (grepl("SCENARIO_5|SCENARIO_6", scenario_name, ignore.case = TRUE))) {
      cat("Using generation ONLY (baseline model for comparison)\n")
    } else {
      cat("Using generation only\n")
    }
    clone_mat <- metadata %>%
      rename(cell_id = 1) %>%
      column_to_rownames("cell_id") %>%
      dplyr::select(generation)
  }
  
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

  # Set base_model parameter: use "full" for Scenario 6 with extra covariates
  if (use_extra_covariates && grepl("SCENARIO_6", scenario_name, ignore.case = TRUE)) {
    cat("Using base_model = 'full' for Scenario 6 full model\n")
    res <- civet(
      AD_mat          = subset_AD,
      DP_mat          = subset_DP,
      clone_mat       = subset_clones,
      minDP           = 5,
      use_random_effect = FALSE,
      base_model      = "full"
    )
  } else {
    res <- civet(
      AD_mat          = subset_AD,
      DP_mat          = subset_DP,
      clone_mat       = subset_clones,
      minDP           = 5,
      use_random_effect = FALSE
    )
  }
  
  # Combine and write out results
  resDF <- purrr::imap_dfr(
    res,
    ~ as.data.frame(.x) %>%
      tibble::rownames_to_column("variant") %>%
      mutate(value = .y)
  )
  
  # Make an output directory inside subrun_dir
  # Use different directory names based on whether extra covariates are used
  if (!use_extra_covariates && (grepl("SCENARIO_5|SCENARIO_6", scenario_name, ignore.case = TRUE))) {
    outdir <- file.path(subrun_dir, "civet_res_generation_only")
  } else {
    outdir <- file.path(subrun_dir, "civet_res")
  }
  if (!dir.exists(outdir)) dir.create(outdir)

  out_rds  <- file.path(outdir, "civet_results.rds")
  out_csv  <- file.path(outdir, "civet_results.csv")

  saveRDS(res, file = out_rds)
  write.csv(resDF, file = out_csv, row.names = FALSE)

  cat("Finished supervised_glm for", subrun_dir, "\n")

  return(invisible(TRUE))
}

################################################################################
# 2b) Function for Scenario 6 alternative analyses
################################################################################
run_supervised_glm_scenario6_alternative <- function(subrun_dir, mode = "cell_cycle_only") {
  # mode: "cell_cycle_only" OR "permuted_generation"

  # Paths we expect
  metadata_csv <- file.path(subrun_dir, "metadata", "simulation_cell_metadata.csv")
  ad_mtx_path  <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.AD.mtx")
  dp_mtx_path  <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.DP.mtx")
  barcodes_txt <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.barcodes.txt")
  mutations_txt <- file.path(subrun_dir, "cellSNP", "cellSNP.tag.mutations.txt")

  # Check if mandatory files exist
  if (!all(file.exists(metadata_csv, ad_mtx_path, dp_mtx_path, barcodes_txt, mutations_txt))) {
    missing_files <- c(
      metadata_csv,
      ad_mtx_path,
      dp_mtx_path,
      barcodes_txt,
      mutations_txt
    )[!file.exists(c(metadata_csv, ad_mtx_path, dp_mtx_path, barcodes_txt, mutations_txt))]
    message("Missing files in ", subrun_dir, ": ", paste(missing_files, collapse = ", "))
    message("Skipping ", subrun_dir, " because required files not found.")
    return(NULL)
  }

  # Read metadata
  metadata <- read_csv(metadata_csv)

  if (mode == "cell_cycle_only") {
    # Use ONLY cell_cycle_potential (no generation)
    cat("Running with CELL CYCLE POTENTIAL ONLY (no generation)\n")

    # Convert cell_cycle_phase to differentiation potential
    metadata <- metadata %>%
      mutate(cell_cycle_potential = case_when(
        cell_cycle_phase == "S"  ~ 4,
        cell_cycle_phase == "G2" ~ 3,
        cell_cycle_phase == "M"  ~ 3,
        cell_cycle_phase == "G1" ~ 2,
        cell_cycle_phase == "G0" ~ 1,
        TRUE ~ 2
      ))

    clone_mat <- metadata %>%
      rename(cell_id = 1) %>%
      column_to_rownames("cell_id") %>%
      dplyr::select(cell_cycle_potential)

    outdir_name <- "civet_res_cell_cycle_only"

  } else if (mode == "permuted_generation") {
    # PERMUTE generation labels (negative control)
    cat("Running with PERMUTED GENERATION labels (negative control)\n")

    set.seed(42)  # Set seed for reproducibility
    metadata$generation_permuted <- sample(metadata$generation)

    clone_mat <- metadata %>%
      rename(cell_id = 1) %>%
      column_to_rownames("cell_id") %>%
      dplyr::select(generation_permuted) %>%
      rename(generation = generation_permuted)

    outdir_name <- "civet_res_permuted"

  } else {
    stop("mode must be 'cell_cycle_only' or 'permuted_generation'")
  }

  # Read AD/DP
  AD_mtx <- read_mtx_safe(ad_mtx_path, mutations_txt, barcodes_txt)
  DP_mtx <- read_mtx_safe(dp_mtx_path, mutations_txt, barcodes_txt)

  # Subset to common barcodes
  common_barcodes <- intersect(rownames(clone_mat), colnames(AD_mtx))
  subset_AD  <- AD_mtx[, common_barcodes, drop = FALSE]
  subset_DP  <- DP_mtx[, common_barcodes, drop = FALSE]
  subset_clones <- clone_mat[common_barcodes, , drop = FALSE]

  # Run CIVET
  cat("Running supervised_glm on", subrun_dir, "with", ncol(subset_AD), "cells...\n")
  res <- civet(
    AD_mat          = subset_AD,
    DP_mat          = subset_DP,
    clone_mat       = subset_clones,
    minDP           = 5,
    use_random_effect = FALSE
  )

  # Combine results
  resDF <- purrr::imap_dfr(
    res,
    ~ as.data.frame(.x) %>%
      tibble::rownames_to_column("variant") %>%
      mutate(value = .y)
  )

  # Save to specific directory
  outdir <- file.path(subrun_dir, outdir_name)
  if (!dir.exists(outdir)) dir.create(outdir)

  out_rds  <- file.path(outdir, "civet_results.rds")
  out_csv  <- file.path(outdir, "civet_results.csv")

  saveRDS(res, file = out_rds)
  write.csv(resDF, file = out_csv, row.names = FALSE)

  cat("Finished supervised_glm for", subrun_dir, "(mode:", mode, ")\n")

  return(invisible(TRUE))
}

################################################################################
# 3) Main driver: find SCENARIO_ folders, then sub-run directories
################################################################################
main <- function() {
  # List all SCENARIO_ directories in the current working directory
  scenario_dirs <- list.dirs(path = "/Users/linxy29/Documents/Data/CIVET/simulation", full.names = TRUE, recursive = FALSE)
  scenario_dirs <- scenario_dirs[grepl("^.*/SCENARIO_", scenario_dirs)]

  # Filter to only process scenarios 4, 5, 6
  # NOTE: Scenario 4 = Varying Depth, Scenario 5 = Metabolic, Scenario 6 = Cell Cycle
  scenario_dirs <- scenario_dirs[grepl("SCENARIO_4|SCENARIO_5|SCENARIO_6", scenario_dirs)]

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
      run_supervised_glm_for_subrun(sr, use_extra_covariates = TRUE)
    }
  }

  cat("\n=== Completed analysis with extra covariates ===\n\n")

  # Additional run: scenarios 5 and 6 with generation ONLY (for comparison)
  cat("=== Running scenarios 5 and 6 with generation ONLY (baseline models) ===\n\n")

  scenario_dirs_baseline <- list.dirs(path = "/Users/linxy29/Documents/Data/CIVET/simulation", full.names = TRUE, recursive = FALSE)
  scenario_dirs_baseline <- scenario_dirs_baseline[grepl("SCENARIO_5|SCENARIO_6", scenario_dirs_baseline)]

  for (scen in scenario_dirs_baseline) {
    subruns <- list.dirs(path = scen, full.names = TRUE, recursive = FALSE)

    # Filter to actual sub-run directories
    subruns <- subruns[
      sapply(subruns, function(x) {
        file.exists(file.path(x, "metadata")) && file.exists(file.path(x, "cellSNP"))
      })
    ]

    # Run with generation only
    for (sr in subruns) {
      run_supervised_glm_for_subrun(sr, use_extra_covariates = FALSE)
    }
  }

  cat("\n=== Completed baseline models ===\n\n")

  # Additional runs for Scenario 6: cell cycle only & permuted generation
  cat("=== Running Scenario 6 with CELL CYCLE ONLY (no generation) ===\n\n")

  scenario_6_dirs <- list.dirs(path = "/Users/linxy29/Documents/Data/CIVET/simulation", full.names = TRUE, recursive = FALSE)
  scenario_6_dirs <- scenario_6_dirs[grepl("SCENARIO_6", scenario_6_dirs)]

  for (scen in scenario_6_dirs) {
    subruns <- list.dirs(path = scen, full.names = TRUE, recursive = FALSE)

    # Filter to actual sub-run directories
    subruns <- subruns[
      sapply(subruns, function(x) {
        file.exists(file.path(x, "metadata")) && file.exists(file.path(x, "cellSNP"))
      })
    ]

    # Run with cell cycle only
    for (sr in subruns) {
      run_supervised_glm_scenario6_alternative(sr, mode = "cell_cycle_only")
    }
  }

  cat("\n=== Running Scenario 6 with PERMUTED GENERATION (negative control) ===\n\n")

  for (scen in scenario_6_dirs) {
    subruns <- list.dirs(path = scen, full.names = TRUE, recursive = FALSE)

    # Filter to actual sub-run directories
    subruns <- subruns[
      sapply(subruns, function(x) {
        file.exists(file.path(x, "metadata")) && file.exists(file.path(x, "cellSNP"))
      })
    ]

    # Run with permuted generation
    for (sr in subruns) {
      run_supervised_glm_scenario6_alternative(sr, mode = "permuted_generation")
    }
  }

  cat("\n=== All CIVET runs completed ===\n")
  cat("- Scenarios 4, 5, 6 with full covariates: civet_res/\n")
  cat("- Scenarios 5, 6 with generation only: civet_res_generation_only/\n")
  cat("- Scenario 6 with cell cycle only: civet_res_cell_cycle_only/\n")
  cat("- Scenario 6 with permuted generation (negative control): civet_res_permuted/\n")
}

################################################################################
# 4) Execute main only when run directly (not when sourced)
################################################################################
# Only execute main() if this script is run directly via Rscript
# When sourced by other scripts, main() will not execute automatically
if (!interactive() && sys.nframe() == 0) {
  # Running as a script via Rscript, call main
  main()
}
# If you want to run main() when sourcing this file, explicitly call main() after sourcing

