#!/usr/bin/env Rscript

# This script runs MitoTracer on simulation data
# It converts cellSNP output to MitoTracer input format and runs the analysis

library(MitoTracer)
library(data.table)
library(Matrix)

# Function to convert cellSNP data to MitoTracer format
convert_cellsnp_to_mitotracer <- function(cellsnp_dir, output_dir) {
  cat("Converting cellSNP data to MitoTracer format...\n")

  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Load cell barcodes
  barcodes_file <- file.path(cellsnp_dir, "cellSNP.tag.barcodes.txt")
  cell_barcodes <- readLines(barcodes_file)

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

  # Convert sparse matrices to dense for easier manipulation
  ad_dense <- as.matrix(ad_matrix)
  dp_dense <- as.matrix(dp_matrix)

  cat("Processing", nrow(ad_dense), "variants and", ncol(ad_dense), "cells...\n")

  # Create MitoTracer-style files for each sample (cell)
  # MitoTracer expects files with columns: ID, pos, s_reads, avg_BQ, t_reads, AF
  for (j in 1:ncol(ad_dense)) {
    sample_data <- data.frame()

    for (i in 1:nrow(ad_dense)) {
      coverage <- dp_dense[i, j]
      alt_depth <- ad_dense[i, j]

      # Skip if coverage is 0
      if (coverage == 0) next

      # Use the ID from VCF file directly
      mt_id <- vcf_data$ID[i]

      # Extract numeric part from ID (e.g., baseline_m5192 -> 5192)
      numeric_part <- sub(".*m(\\d+)$", "\\1", mt_id)
      variant_pos <- as.integer(numeric_part)

      # Calculate allele frequency
      af <- alt_depth / coverage

      # Assume average base quality of 33 (typical for good quality data)
      avg_bq <- 33

      sample_data <- rbind(sample_data, data.frame(
        ID = mt_id,
        pos = variant_pos,
        s_reads = alt_depth,
        avg_BQ = avg_bq,
        t_reads = coverage,
        AF = af,
        stringsAsFactors = FALSE
      ))
    }

    # Write sample file
    sample_name <- cell_barcodes[j]
    sample_file <- file.path(output_dir, paste0(sample_name, ".txt"))
    write.table(sample_data, sample_file, sep = "\t", row.names = FALSE, quote = FALSE)

    if (j %% 100 == 0) {
      cat("Processed", j, "cells...\n")
    }
  }

  cat("Conversion complete. Wrote", ncol(ad_dense), "sample files.\n")
  return(output_dir)
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
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Convert cellSNP to MitoTracer format
  mt_input_dir <- file.path(output_dir, "mitotracer_input")
  convert_cellsnp_to_mitotracer(cellsnp_dir, mt_input_dir)

  # Load data using MitoTracer
  cat("\nLoading data with MitoTracer...\n")
  mt_data <- read.MT.variant.files.bulkATACseq(mt_input_dir)

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

  mt_distance <- MT.feature.distance(mt_matrix_subset, iteration = 2000)

  # Select informative variants
  cat("\nSelecting informative variants...\n")
  # dis_cutoff = 0, sample_type = 2 (paired samples), size = 5 (top 5 features)
  mt_informative <- MT.feature.selection(
    mt_distance,
    mt_matrix,
    dis_cutoff = 0,
    sample_type = 2,
    size = 5
  )

  # Get informative variant names
  informative_variant_names <- rownames(mt_informative)
  cat("\nInformative variants identified:", length(informative_variant_names), "\n")

  # Load all mutations for categorization
  mutations_file <- file.path(cellsnp_dir, "cellSNP.tag.mutations.txt")
  all_mutations <- readLines(mutations_file)

  # Load VCF to create mapping from position to mutation name
  vcf_file <- file.path(cellsnp_dir, "cellSNP.tag.vcf")
  vcf_data <- read.table(vcf_file, comment.char = "#", stringsAsFactors = FALSE)
  colnames(vcf_data) <- c("CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO")

  # Create mapping from MT_ID format to original mutation name
  # Extract position from MT_ID (e.g., MT_5192_N-A -> 5192)
  pos_to_mutation <- setNames(vcf_data$ID,
                               sapply(vcf_data$ID, function(id) {
                                 paste0("MT_", sub(".*m(\\d+)$", "\\1", id), "_",
                                        vcf_data$REF[vcf_data$ID == id], "-",
                                        vcf_data$ALT[vcf_data$ID == id])
                               }))

  # Map informative variants back to original mutation names
  informative_mapped <- character(0)
  for (var_name in informative_variant_names) {
    # Extract position from MT_pos_ref-alt format
    pos_match <- regmatches(var_name, regexpr("MT_\\d+", var_name))
    if (length(pos_match) > 0) {
      # Find matching mutation in original list
      for (mut in all_mutations) {
        numeric_part <- sub(".*m(\\d+)$", "\\1", mut)
        if (grepl(numeric_part, pos_match)) {
          informative_mapped <- c(informative_mapped, mut)
          break
        }
      }
    }
  }

  # Categorize mutations
  baseline_mutations <- grep("baseline", all_mutations, value = TRUE, ignore.case = TRUE)
  false_mutations <- grep("false", all_mutations, value = TRUE, ignore.case = TRUE)
  rest_mutations <- setdiff(all_mutations, c(baseline_mutations, false_mutations))

  # Create mutation data frame
  mutation_data <- data.frame(
    scenario = scenario,
    condition = condition,
    mutation_name = all_mutations,
    detected = all_mutations %in% informative_mapped,
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
    tryCatch({
      result <- process_simulation_with_mitotracer(sim_dir)
      all_results[[sim_dir]] <- result
      all_mutation_data[[length(all_mutation_data) + 1]] <- result$mutation_data
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
