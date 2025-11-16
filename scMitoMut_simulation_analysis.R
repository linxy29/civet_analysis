#!/usr/bin/env Rscript

# This script runs scMitoMut on simulation data
# It converts cellSNP output to scMitoMut input format and runs the analysis

library(scMitoMut)
library(data.table)
library(Matrix)
library(rhdf5)

# Function to convert cellSNP data to scMitoMut TSV format
convert_cellsnp_to_scmitomut <- function(cellsnp_dir, output_file) {
  cat("Converting cellSNP data to scMitoMut format...\n")

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

  # Note: Matrix Market format is (variants x cells)
  # We need to extract forward and reverse depths
  # For simulation, we'll split the depth approximately
  # In real data, this would come from BAM files

  # Convert sparse matrices to dense for easier manipulation
  ad_dense <- as.matrix(ad_matrix)
  dp_dense <- as.matrix(dp_matrix)

  # Create data frame in scMitoMut format
  # For each variant and cell combination
  scmitomut_data <- data.frame()

  cat("Processing", nrow(ad_dense), "variants and", ncol(ad_dense), "cells...\n")

  # Process in batches to save memory
  batch_size <- 100
  n_variants <- nrow(ad_dense)

  for (batch_start in seq(1, n_variants, by = batch_size)) {
    batch_end <- min(batch_start + batch_size - 1, n_variants)

    batch_data <- list()

    for (i in batch_start:batch_end) {
      # Extract numeric part from ID (e.g., baseline_m5192 -> 5192, m10004 -> 10004)
      mutation_id <- vcf_data$ID[i]
      numeric_part <- sub(".*m(\\d+)$", "\\1", mutation_id)
      variant_pos <- as.integer(numeric_part)

      # Use N for reference and A for alternate (not simulated)
      ref_allele <- "N"
      alt_allele <- "A"

      for (j in 1:ncol(ad_dense)) {
        coverage <- dp_dense[i, j]
        alt_depth <- ad_dense[i, j]

        # Skip if coverage is 0
        if (coverage == 0) next

        # fwd_depth and rev_depth are split from alternative allele count (AD)
        # In real data, this would come from strand-specific information
        fwd_depth <- floor(alt_depth / 2)
        rev_depth <- alt_depth - fwd_depth

        batch_data[[length(batch_data) + 1]] <- data.frame(
          loc = variant_pos,
          cell_barcode = cell_barcodes[j],
          fwd_depth = fwd_depth,
          rev_depth = rev_depth,
          alt = alt_allele,
          coverage = coverage,
          ref = ref_allele,
          stringsAsFactors = FALSE
        )
      }
    }

    if (length(batch_data) > 0) {
      batch_df <- do.call(rbind, batch_data)
      scmitomut_data <- rbind(scmitomut_data, batch_df)
    }

    if (batch_end %% 500 == 0) {
      cat("Processed", batch_end, "variants...\n")
    }
  }

  # Write to TSV file (gzipped)
  cat("Writing to", output_file, "...\n")
  fwrite(scmitomut_data, output_file, sep = "\t", compress = "gzip")

  cat("Conversion complete. Wrote", nrow(scmitomut_data), "records.\n")
  return(output_file)
}

# Function to process simulation data with scMitoMut
process_simulation_with_scmitomut <- function(sim_dir) {
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
  output_dir <- file.path(sim_dir, "scMitoMut_selection_summary")
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Convert cellSNP to scMitoMut format
  tsv_file <- file.path(output_dir, "scMitoMut_input.tsv.gz")
  convert_cellsnp_to_scmitomut(cellsnp_dir, tsv_file)

  # Parse the TSV file to HDF5
  cat("\nParsing TSV to HDF5 format...\n")
  h5_file <- file.path(output_dir, "scMitoMut_data.h5")
  h5_file <- parse_table(tsv_file, h5_file = h5_file)

  # Open HDF5 file
  cat("Opening HDF5 file...\n")
  x <- open_h5_file(h5_file)

  # Get all cells from the HDF5 file directly
  # Read cell list from HDF5 file
  all_cells <- h5read(x$h5f, "cell_list")
  cat("Total cells in dataset:", length(all_cells), "\n")

  # Subset cells (using all cells)
  x <- subset_cell(x, all_cells)

  # Run scMitoMut model fitting
  cat("\nRunning scMitoMut model fitting...\n")
  run_model_fit(x, mc.cores = 1)

  # Filter locations
  cat("\nFiltering locations...\n")
  x <- filter_loc(
    mtmutObj = x,
    min_cell = 2,
    model = "bb",
    p_threshold = 0.01,
    p_adj_method = "fdr"
  )

  # Get informative variants
  informative_variants <- x$loc_pass
  cat("\nInformative variants identified:", length(informative_variants), "\n")

  # Load mutation names from VCF for mapping
  vcf_file <- file.path(cellsnp_dir, "cellSNP.tag.vcf")
  vcf_data <- read.table(vcf_file, comment.char = "#", stringsAsFactors = FALSE)
  colnames(vcf_data) <- c("CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO")

  # Create mapping from numeric position to mutation name
  # Extract numeric part from each ID
  numeric_positions <- sapply(vcf_data$ID, function(id) {
    as.integer(sub(".*m(\\d+)$", "\\1", id))
  })
  pos_to_name <- setNames(vcf_data$ID, as.character(numeric_positions))

  # Map scMitoMut positions back to mutation names
  # Extract numeric from informative variants (chrM.15027 -> 15027)
  informative_numeric <- sapply(informative_variants, function(loc) {
    as.character(as.integer(sub(".*\\.(\\d+)$", "\\1", loc)))
  })

  informative_variant_names <- pos_to_name[informative_numeric]
  informative_variant_names <- informative_variant_names[!is.na(informative_variant_names)]

  # Load all mutations for categorization
  mutations_file <- file.path(cellsnp_dir, "cellSNP.tag.mutations.txt")
  all_mutations <- readLines(mutations_file)

  # Categorize mutations
  baseline_mutations <- grep("baseline", all_mutations, value = TRUE, ignore.case = TRUE)
  false_mutations <- grep("false", all_mutations, value = TRUE, ignore.case = TRUE)
  rest_mutations <- setdiff(all_mutations, c(baseline_mutations, false_mutations))

  # Extract both raw and FDR-adjusted p-values from scMitoMut results
  # P-values are stored at /pval/{loc}/bb_pval for each location
  p_values_raw <- rep(NA, length(all_mutations))
  p_values_adj <- rep(NA, length(all_mutations))

  tryCatch({
    cat("\nExtracting p-values from HDF5 file...\n")

    # Get list of all locations that were analyzed
    loc_list <- h5read(x$h5f, "loc_list")
    cat("Total locations analyzed:", length(loc_list), "\n")

    # For each location, get both raw and adjusted p-values (minimum across all cells)
    loc_pvalues_raw <- list()
    loc_pvalues_adj <- list()

    for (loc in loc_list) {
      # Path to p-values for this location: /pval/{loc}/bb_pval
      pval_path <- paste0("/pval/", loc, "/bb_pval")

      # Check if this path exists
      if (H5Lexists(x$h5f, pval_path)) {
        # Read raw p-values for all cells at this location
        pval_raw <- h5read(x$h5f, pval_path)

        # Apply FDR adjustment (as done in get_pval function)
        pval_adj <- p.adjust(pval_raw, method = "fdr")

        # Take minimum p-value across cells (most significant)
        min_pval_raw <- min(pval_raw, na.rm = TRUE)
        min_pval_adj <- min(pval_adj, na.rm = TRUE)

        loc_pvalues_raw[[as.character(loc)]] <- min_pval_raw
        loc_pvalues_adj[[as.character(loc)]] <- min_pval_adj
      }
    }

    cat("Retrieved p-values for", length(loc_pvalues_adj), "locations\n")

    # Debug: Show what we have
    if (length(loc_pvalues_adj) > 0) {
      cat("\nDebug - First location with p-value:\n")
      first_loc <- names(loc_pvalues_adj)[1]
      cat("  Location:", first_loc, "\n")
      cat("  Raw p-value:", loc_pvalues_raw[[first_loc]], "\n")
      cat("  Adj p-value:", loc_pvalues_adj[[first_loc]], "\n")
      cat("  Numeric extracted:", as.integer(sub(".*\\.(\\d+)$", "\\1", first_loc)), "\n")
    }

    # Map back to mutations using numeric positions
    mapped_count <- 0
    for (i in seq_along(all_mutations)) {
      # Extract numeric part from mutation ID
      mut_id <- all_mutations[i]
      numeric_pos <- as.character(as.integer(sub(".*m(\\d+)$", "\\1", mut_id)))

      # Find matching location in loc_pvalues
      # Extract numeric from locations (chrM.15027 -> 15027)
      for (loc in names(loc_pvalues_adj)) {
        loc_numeric <- as.character(as.integer(sub(".*\\.(\\d+)$", "\\1", loc)))
        if (loc_numeric == numeric_pos) {
          p_values_raw[i] <- loc_pvalues_raw[[loc]]
          p_values_adj[i] <- loc_pvalues_adj[[loc]]
          mapped_count <- mapped_count + 1
          break
        }
      }
    }

    cat("Successfully mapped", mapped_count, "mutations to p-values\n")
    cat("Raw p-values extracted:", sum(!is.na(p_values_raw)), "\n")
    cat("Adj p-values extracted:", sum(!is.na(p_values_adj)), "mutations\n")
  }, error = function(e) {
    cat("Warning: Error extracting p-values:", conditionMessage(e), "\n")
    cat("Error details:", conditionMessage(e), "\n")
  })

  # Create mutation data frame
  mutation_data <- data.frame(
    scenario = scenario,
    condition = condition,
    mutation_name = all_mutations,
    detected = all_mutations %in% informative_variant_names,
    pvalue = p_values_raw,
    fdr_adjusted_pvalue = p_values_adj,
    baseline_mutation = all_mutations %in% baseline_mutations,
    false_mutation = all_mutations %in% false_mutations,
    rest_mutation = all_mutations %in% rest_mutations,
    stringsAsFactors = FALSE
  )

  # Save results
  cat("\nSaving results...\n")

  # Save informative variants
  writeLines(informative_variant_names, file.path(output_dir, "informative_variants.txt"))

  # Save mutation data
  write.csv(mutation_data, file.path(output_dir, "mutation_data.csv"), row.names = FALSE)

  # Print summary
  cat("\n===========================================\n")
  cat("Summary:\n")
  cat("===========================================\n")
  cat("Total mutations:", nrow(mutation_data), "\n")
  cat("Informative mutations (scMitoMut):", sum(mutation_data$detected), "\n")
  cat("  - Baseline mutations detected:", sum(mutation_data$detected & mutation_data$baseline_mutation), "/",
      sum(mutation_data$baseline_mutation), "\n")
  cat("  - False mutations detected:", sum(mutation_data$detected & mutation_data$false_mutation), "/",
      sum(mutation_data$false_mutation), "\n")
  cat("  - Other mutations detected:", sum(mutation_data$detected & mutation_data$rest_mutation), "/",
      sum(mutation_data$rest_mutation), "\n")
  cat("\nResults saved to:", output_dir, "\n")
  cat("===========================================\n\n")

  return(list(
    informative_variants = informative_variant_names,
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
  # In this case, check immediate subdirectories for cellSNP folders
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
    cat(" ", i, ".", sim_folders[i], "\n")
  }
  cat("\n")

  # Process each simulation folder
  all_results <- list()
  all_mutation_data <- list()

  for (sim_dir in sim_folders) {
    # Check if mutation_data.csv already exists, skip if it does
    output_dir <- file.path(sim_dir, "scMitoMut_selection_summary")
    mutation_data_file <- file.path(output_dir, "mutation_data.csv")

    if (file.exists(mutation_data_file)) {
      cat("\nSkipping", sim_dir, "- mutation_data.csv already exists\n")
      next
    }

    tryCatch({
      result <- process_simulation_with_scmitomut(sim_dir)
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
    combined_output_path <- file.path(base_dir, "scMitoMut_mutation_combine.csv")
    write.csv(combined_mutation_df, combined_output_path, row.names = FALSE)
    cat("\nCombined mutation data saved to:", combined_output_path, "\n")
  }

  cat("\nAll processing complete!\n")
}

# Run main function
if (!interactive()) {
  main()
}

main()