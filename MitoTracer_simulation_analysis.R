#!/usr/bin/env Rscript

# This script runs MitoTracer on simulation data
# It converts cellSNP output to MitoTracer input format and runs the analysis

library(MitoTracer)
library(data.table)
library(Matrix)

# ============================================
# CONFIGURATION: Set subsample size here
# ============================================
# Set to NULL to use all cells, or specify a number (e.g., 200)
N_CELLS_SUBSAMPLE <- 200

# Function to convert cellSNP data to MitoTracer format
convert_cellsnp_to_mitotracer <- function(cellsnp_dir, n_cells_subsample = N_CELLS_SUBSAMPLE) {
  cat("Converting cellSNP data to MitoTracer format...\n")

  # Load cell barcodes
  barcodes_file <- file.path(cellsnp_dir, "cellSNP.tag.barcodes.txt")
  cell_barcodes <- readLines(barcodes_file)

  # Subsample cells if requested
  n_total_cells <- length(cell_barcodes)
  if (!is.null(n_cells_subsample) && n_cells_subsample < n_total_cells) {
    cat("Subsampling", n_cells_subsample, "cells from", n_total_cells, "total cells...\n")
    set.seed(42)  # For reproducibility
    selected_cell_indices <- sort(sample(1:n_total_cells, n_cells_subsample))
  } else {
    cat("Using all", n_total_cells, "cells...\n")
    selected_cell_indices <- 1:n_total_cells
  }

  # Load mutation names from VCF
  vcf_file <- file.path(cellsnp_dir, "cellSNP.tag.vcf")
  vcf_data <- read.table(vcf_file, comment.char = "#", stringsAsFactors = FALSE)
  colnames(vcf_data) <- c("CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO")

  # Load AD (Allele Depth) matrix
  ad_file <- file.path(cellsnp_dir, "cellSNP.tag.AD.mtx")
  ad_matrix <- readMM(ad_file)

  # Load DP (Depth) matrix
  dp_file <- file.path(cellsnp_dir, "cellSNP.tag.DP.mtx")
  dp_matrix <- readMM(dp_file)

  # Subset matrices to selected cells
  ad_matrix <- ad_matrix[, selected_cell_indices]
  dp_matrix <- dp_matrix[, selected_cell_indices]
  cell_barcodes <- cell_barcodes[selected_cell_indices]

  cat("Processing", nrow(ad_matrix), "variants and", ncol(ad_matrix), "cells...\n")

  # Pre-extract numeric positions from VCF IDs
  variant_positions <- as.integer(sub(".*m(\\d+)$", "\\1", vcf_data$ID))

  # Convert sparse matrices to triplet format for efficient iteration
  # This avoids iterating over zero entries
  ad_triplet <- summary(ad_matrix)
  dp_triplet <- summary(dp_matrix)

  # Merge AD and DP data by matching row (variant) and column (cell) indices
  # This only processes non-zero coverage entries
  cat("Merging AD and DP data...\n")
  merged_data <- merge(
    ad_triplet,
    dp_triplet,
    by = c("i", "j"),
    suffixes = c("_ad", "_dp")
  )

  # Filter out zero coverage entries
  merged_data <- merged_data[merged_data$x_dp > 0, ]

  cat("Creating data frame from", nrow(merged_data), "non-zero entries...\n")

  # Create data frame directly - much faster than loop
  mt_data <- data.frame(
    ID = vcf_data$ID[merged_data$i],
    pos = variant_positions[merged_data$i],
    s_reads = merged_data$x_ad,
    avg_BQ = 33,
    t_reads = merged_data$x_dp,
    AF = merged_data$x_ad / merged_data$x_dp,
    sample = cell_barcodes[merged_data$j],
    stringsAsFactors = FALSE
  )

  cat("Conversion complete. Created data frame with", nrow(mt_data), "records.\n")
  return(mt_data)
}

# Function to process simulation data with MitoTracer
process_simulation_with_mitotracer <- function(sim_dir) {
  cat("\n===========================================\n")
  cat("Processing simulation data from:", sim_dir, "\n")
  cat("===========================================\n\n")

  # Extract scenario and condition information
  parts <- unlist(strsplit(sim_dir, "/"))
  scenario <- "unknown"
  for (part in parts) {
    if (grepl("^SCENARIO", part)) {
      scenario <- part
      break
    }
  }
  condition <- parts[length(parts)]

  cat("Scenario:", scenario, "\n")
  cat("Condition:", condition, "\n\n")

  # Set up paths
  cellsnp_dir <- file.path(sim_dir, "cellSNP")
  output_dir <- file.path(sim_dir, "MitoTracer_selection_summary")

  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Convert cellSNP to MitoTracer format and create mt_data directly
  cat("\nCreating MitoTracer data structure...\n")
  mt_data <- convert_cellsnp_to_mitotracer(cellsnp_dir)

  # Remove sequencing errors and extract VAF matrix
  cat("\nDetecting errors and extracting variants...\n")
  mt_data_variants <- ErrorMutmatrix_detect(mt_data)

  # Filter detected variants
  mt_data_detected <- mt_data_variants[mt_data_variants$detection == "detected", ]

  cat("Detected variants after error filtering:", nrow(mt_data_detected), "\n")

  # Create mutation matrix
  cat("\nCreating mutation matrix...\n")
  mt_matrix <- MTmutMatrix_refined(mt_data_detected)

  cat("Mutation matrix dimensions:", nrow(mt_matrix), "variants x", ncol(mt_matrix), "samples\n")

  # Calculate distance (use all variants if <= 1000, otherwise sample)
  n_variants <- nrow(mt_matrix)
  if (n_variants > 1000) {
    cat("\nToo many variants (", n_variants, "), using first 1000 for distance calculation...\n")
    mt_matrix_subset <- mt_matrix[1:1000, ]
  } else {
    cat("\nCalculating distances for", n_variants, "variants...\n")
    mt_matrix_subset <- mt_matrix
  }

  mt_distance <- MT.feature.distance(mt_matrix_subset, iteration = 1000)

  # Select informative variants
  cat("\nSelecting informative variants...\n")
  # dis_cutoff = 0.05, sample_type = 2 (paired samples)
  # MT.feature.selection returns a vector of informative variant names
  informative_variant_names <- MT.feature.selection(
    mt_distance,
    mt_matrix,
    dis_cutoff = 0.05,
    sample_type = 2
  )

  cat("\nInformative variants identified:", length(informative_variant_names), "\n")

  # Load all mutations for categorization
  mutations_file <- file.path(cellsnp_dir, "cellSNP.tag.mutations.txt")
  all_mutations <- readLines(mutations_file)

  # Since we now use VCF IDs directly, informative_variant_names should match all_mutations
  # No mapping needed - the IDs are already in the correct format
  informative_mapped <- informative_variant_names

  # Categorize mutations
  baseline_mutations <- grep("baseline", all_mutations, value = TRUE, ignore.case = TRUE)
  false_mutations <- grep("false", all_mutations, value = TRUE, ignore.case = TRUE)
  rest_mutations <- setdiff(all_mutations, c(baseline_mutations, false_mutations))

  # Map distance values to all mutations
  # mt_distance is a named vector, we need to match it to all_mutations
  # Vectorized approach - much faster than loop
  distance_values <- mt_distance[all_mutations]
  # mt_distance[all_mutations] returns NA for non-matching names automatically

  # Create mutation data frame
  mutation_data <- data.frame(
    scenario = scenario,
    condition = condition,
    mutation_name = all_mutations,
    detected = all_mutations %in% informative_mapped,
    distance = distance_values,
    baseline_mutation = all_mutations %in% baseline_mutations,
    false_mutation = all_mutations %in% false_mutations,
    rest_mutation = all_mutations %in% rest_mutations,
    stringsAsFactors = FALSE
  )

  # Save results
  cat("\nSaving results...\n")

  # Save informative variants
  writeLines(informative_mapped, file.path(output_dir, "informative_variants.txt"))

  # Save mutation data
  write.csv(mutation_data, file.path(output_dir, "mutation_data.csv"), row.names = FALSE)

  # Save mutation matrix
  write.csv(mt_matrix, file.path(output_dir, "mutation_matrix.csv"))

  # Save distance data
  distance_df <- data.frame(
    mutation = names(mt_distance),
    distance = as.vector(mt_distance),
    stringsAsFactors = FALSE
  )
  write.csv(distance_df, file.path(output_dir, "distance_data.csv"), row.names = FALSE)

  # Print summary
  cat("\n===========================================\n")
  cat("Summary:\n")
  cat("===========================================\n")
  cat("Total mutations:", nrow(mutation_data), "\n")
  cat("Informative mutations (MitoTracer):", sum(mutation_data$detected), "\n")
  cat("  - Baseline mutations detected:", sum(mutation_data$detected & mutation_data$baseline_mutation), "/",
      sum(mutation_data$baseline_mutation), "\n")
  cat("  - False mutations detected:", sum(mutation_data$detected & mutation_data$false_mutation), "/",
      sum(mutation_data$false_mutation), "\n")
  cat("  - Other mutations detected:", sum(mutation_data$detected & mutation_data$rest_mutation), "/",
      sum(mutation_data$rest_mutation), "\n")
  cat("\nResults saved to:", output_dir, "\n")
  cat("===========================================\n\n")

  return(list(
    informative_variants = informative_mapped,
    mutation_data = mutation_data
  ))
}

# Function to find simulation folders
find_simulation_folders <- function(base_dir = ".") {
  simulation_folders <- c()

  # First check if base_dir itself contains a cellSNP folder
  if (dir.exists(file.path(base_dir, "cellSNP"))) {
    simulation_folders <- c(simulation_folders, base_dir)
    cat("Found cellSNP folder in base directory:", base_dir, "\n")
  }

  # Check if base_dir is itself a SCENARIO directory
  base_name <- basename(base_dir)
  if (grepl("^SCENARIO", base_name)) {
    subdirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
    for (subdir in subdirs) {
      if (dir.exists(file.path(subdir, "cellSNP"))) {
        simulation_folders <- c(simulation_folders, subdir)
      }
    }
  } else {
    # Find all directories starting with SCENARIO
    all_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
    scenario_dirs <- all_dirs[grepl("^.*/SCENARIO", all_dirs)]

    for (scenario_dir in scenario_dirs) {
      # Find all subdirectories
      subdirs <- list.dirs(scenario_dir, full.names = TRUE, recursive = FALSE)

      for (subdir in subdirs) {
        # Check if this is a simulation directory (has cellSNP folder)
        if (dir.exists(file.path(subdir, "cellSNP"))) {
          simulation_folders <- c(simulation_folders, subdir)
        }
      }
    }
  }

  return(simulation_folders)
}

# Main function
main <- function() {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) == 0) {
    base_dir <- "/Users/linxy29/Documents/Data/CIVET/simulation"
    cat("No base directory specified, using default:", base_dir, "\n")
  } else {
    base_dir <- args[1]
    cat("Using base directory:", base_dir, "\n")
  }

  # Check if base_dir exists, if not use current directory
  if (!dir.exists(base_dir)) {
    cat("Base directory does not exist, using current directory\n")
    base_dir <- "."
  }

  # Find all simulation folders
  sim_folders <- find_simulation_folders(base_dir)

  if (length(sim_folders) == 0) {
    cat("No simulation folders found under SCENARIO* directories.\n")
    return()
  }

  cat("Found", length(sim_folders), "simulation folders to process:\n")
  for (i in seq_along(sim_folders)) {
    cat("  ", i, ".", sim_folders[i], "\n")
  }
  cat("\n")

  # Process each simulation folder
  all_results <- list()
  all_mutation_data <- list()

  for (sim_dir in sim_folders) {
    # Check if output directory already exists, skip if it does
    output_dir <- file.path(sim_dir, "MitoTracer_selection_summary")
    if (dir.exists(output_dir)) {
      cat("\nSkipping", sim_dir, "- MitoTracer_selection_summary already exists\n")
      next
    }

    tryCatch({
      result <- process_simulation_with_mitotracer(sim_dir)
      # Only add to results if not skipped (NULL return)
      if (!is.null(result)) {
        all_results[[sim_dir]] <- result
        all_mutation_data[[length(all_mutation_data) + 1]] <- result$mutation_data
      }
    }, error = function(e) {
      cat("Error processing", sim_dir, ":", conditionMessage(e), "\n")
    })
  }

  # Combine all mutation data
  if (length(all_mutation_data) > 0) {
    combined_mutation_df <- do.call(rbind, all_mutation_data)

    # Save combined mutation data
    combined_output_path <- file.path(base_dir, "MitoTracer_mutation_combine.csv")
    write.csv(combined_mutation_df, combined_output_path, row.names = FALSE)
    cat("\nCombined mutation data saved to:", combined_output_path, "\n")
  }

  cat("\nAll processing complete!\n")
}

# Run main function
if (!interactive()) {
  main()
}

#sim_dir <- "/Users/linxy29/Documents/Data/CIVET/simulation/SCENARIO_1_Mutation_Rate/mutation_rate_1_20250507_120608"
#result <- process_simulation_with_mitotracer(sim_dir)

main()
