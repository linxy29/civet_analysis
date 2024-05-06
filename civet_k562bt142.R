# This file failed to run, need further check
# Xinyi Lin, 202404
# Goal: apply the variant selection method in k562bt142 sample
# Useage: Rscript /home/linxy29/code/R/MAESTER-2021/Pre-processing/Assemble_fastq_modified.R . "k562bt142_10x" "774" ./cellranger/outs/filtered_feature_bc_matrix/barcodes_copy.tsv 16 12

options(max.print = 500)
options(stringsAsFactors = FALSE)
options(scipen = 999)

### Load libraries
suppressMessages(library(tidyverse))
suppressMessages(library(Seurat))

rm(list=ls())

source("./supervised_selection_function_twogroup.R")

### Arguments to be provided when executing script
cellsnp_path <- commandArgs(trailingOnly=TRUE)[1] # One or multiple directories containing fastq files, not searched recursively
SampleName <- commandArgs(trailingOnly=TRUE)[2] # Sample name that will be used for output files

### My own
#cellsnp_path <- "/home/linxy29/data/maester/maester_paper/k562_bt142/maester/trimmed_starsolo_chrM_cellSNP0"
#SampleName <- "k562_bt142_10x"

# Define paths to input files for better readability and maintenance
base_path <- "/home/linxy29/data/maester/maester_paper/k562_bt142/maester/trimmed_starsolo_chrM_cellSNP0"
ad_mtx_path <- file.path(base_path, "cellSNP.tag.AD.mtx")
dp_mtx_path <- file.path(base_path, "cellSNP.tag.DP.mtx")
features_path <- file.path(base_path, "cellSNP.variants.tsv")
cells_path <- file.path(base_path, "cellSNP.samples.tsv")

# Function to read matrix with error checking
read_mtx_safe <- function(mtx, features, cells, feature.column = 1) {
  if (!file.exists(mtx) || !file.exists(features) || !file.exists(cells)) {
    stop("One or more input files do not exist.")
  }
  
  # Assuming ReadMtx is a predefined function or part of a package not shown here
  ReadMtx(mtx = mtx, features = features, cells = cells, feature.column = feature.column)
}

# Read allele depth (AD) and total depth (DP) matrices
AD_mtx <- read_mtx_safe(ad_mtx_path, features_path, cells_path)
DP_mtx <- read_mtx_safe(dp_mtx_path, features_path, cells_path)

# Calculate allele frequencies, handling division by zero and NA values
af_mtx <- AD_mtx / DP_mtx
af_mtx[is.na(af_mtx)] <- 0  # Replace NA values resulting from division by zero with 0

# Output dimensions of matrices for verification
cat("Dimensions of AD_mtx:", dim(AD_mtx), "\n")
cat("Dimensions of DP_mtx:", dim(DP_mtx), "\n")
cat("Dimensions of af_mtx:", dim(af_mtx), "\n")

# Display a subset of the allele frequency matrix
af_mtx[1:10, 1:10]

cell_wClone = read_csv("/home/linxy29/data/maester/maester_paper/k562_bt142/cell_label.csv")

common_barcode = intersect(cell_wClone$barcode, colnames(AD_mtx))
subset_AD = AD_mtx[,common_barcode]
subset_DP = DP_mtx[,common_barcode]
subset_cell_wClone = cell_wClone %>% 
  filter(barcode %in% common_barcode)
subset_AD[1:5, 1:5]
subset_cell_wClone %>% head()
dim(subset_AD)
dim(subset_DP)
dim(subset_cell_wClone)

colnames(subset_cell_wClone) = c("cellID", "cell_label")
res = FindVariants(AD_mat = subset_AD, DP_mat = subset_DP, clone_mat = subset_cell_wClone, ident.1 = "K562")
#res = supervised_glm(AD_mat = subset_AD, DP_mat = subset_DP, clone_mat = subset_cell_wClone, use_random_effect = FALSE)
saveRDS(res, file = "/home/linxy29/data/maester/maester_paper/k562_bt142/variant_selection/fixed_effect/temp_res.rds")
write.csv(res, file = "/home/linxy29/data/maester/maester_paper/k562_bt142/variant_selection/fixed_effect/temp_res.csv")