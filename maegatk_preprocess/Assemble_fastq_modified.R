# Xinyi Lin, 202206
# Goal: append cell barcode and unique molecular identifier, _CB_UMI, from Read 1 to each Read 2 identifier (modified)
# Useage: Rscript /home/linxy29/code/civet/maegatk_preprocess/Assemble_fastq_modified.R . "maester_fastq_assemble" "mb1_S1_L001" ../scRNA_cellranger/outs/filtered_feature_bc_matrix/barcodes_copy.tsv 16 12

options(max.print = 500)
options(stringsAsFactors = FALSE)
options(scipen = 999)


### Load libraries
suppressMessages(library(ShortRead))

rm(list=ls())

### Functions
cutf <- function(x, f=1, d="/", ...) sapply(strsplit(x, d), function(i) i[f], ...)

### Arguments to be provided when executing script
Folder <- commandArgs(trailingOnly=TRUE)[1] # One or multiple directories containing fastq files, not searched recursively
SampleName <- commandArgs(trailingOnly=TRUE)[2] # Sample name that will be used for output files
R1Pattern <- commandArgs(trailingOnly=TRUE)[3]
CellBarcodes <- commandArgs(trailingOnly=TRUE)[4] # Allowlist of cell barcodes to filter by. Could be cells from the CellRanger filtered_feature_bc_matrix, or that passed scRNA-seq QC, or all whitelisted 10x cell barcodes.
CBlength <- as.numeric( commandArgs(trailingOnly=TRUE)[5] ) # Length of the cell barcode (16 for 10x 3' v3)
UMIlength <- as.numeric( commandArgs(trailingOnly=TRUE)[6] ) # Length of the UMI (12 for 10x 3' v3)


### My own
#Folder <- "/home/linxy29/data/maester/mb/mb1/maester_fastq"
#SampleName <- "k562bt142_test"
#R1Pattern <- "S1_L001"
#CellBarcodes <- "/home/linxy29/data/maester/maester_paper/k562_bt142/RNA/cellranger/outs/filtered_feature_bc_matrix/barcodes_copy.tsv"
#CBlength <- 16
#UMIlength <- 12

### Find R1 fastq files (R2 is found by substitution later)
#R1.ch <- list.files(Folder, pattern = paste0(".*", R1Pattern, "_1.fastq*"), full.names = T)
R1.ch <- list.files(Folder, pattern = paste0(".*", R1Pattern, "_R1_*"), full.names = T)
message(Sys.time(), "\nLoading ", length(R1.ch)*2, " fastq files:")
message(cat(c(R1.ch, sub("_1", "_2", R1.ch)), sep = "\n"))
if(length(R1.ch) == 0) stop("Did not find fastq files.")


### Filter for cell barcodes in the allowlist. Remove -1 from the end (if added by CellRanger count).
cells.split <- unlist( strsplit(CellBarcodes, ",") )
cells.df <- do.call(rbind, lapply(cells.split, function(x) read.table(x)))
cells.ch <- cutf(cells.df$V1, d = "-", f = 1)
if(length(cells.ch) == 0) stop("No cells found.")
message("Found ", length(cells.ch), " cells.\n")


### Process fastq files
message("Read R1 and R2 sequences, filter by ", length(cells.ch), " cell barcodes, write assembled fastq...")
report.ls <- list()  # empty list to store number of reads

# For each R1 fastq
for(f1 in R1.ch) {
  #f1 <- R1.ch[1]
  # Identify R2 fastq
  f2 <- sub("_R1", "_R2", f1)
  
  # Load file in 1E7 read increments
  message("file ", match(f1, R1.ch), "/", length(R1.ch), ": ", basename(f1), " ", appendLF = FALSE)
  strm1 <- FastqStreamer(f1, n=1E7)  # 1M reads by default
  strm2 <- FastqStreamer(f2, n=1E7)
  
  start <- Sys.time()
  message("\nStart time: ", start, "\n")
  # For every 10 million reads...  
  repeat{
    message("*", appendLF = FALSE)
    fq1 <- yield(strm1)
    fq2 <- yield(strm2)
    if(length(fq1) == 0 | length(fq2) == 0) break
    
    # Match to expected cell barcodes
    fq1.m <- ifelse(is.element(as.vector(subseq(sread(fq1), 1, CBlength)), cells.ch), yes = T, no = F)
    report.ls[[basename(f1)]] <- c(length(fq1.m), sum(fq1.m))
    
    # Filter unmatched reads from the ShortRead objects
    fq1.f <- fq1[fq1.m]
    fq2.f <- fq2[fq1.m]
    
    # Extract cell barcode and umi from Read1
    fq1.f.cell <- as.vector(subseq(sread(fq1.f), 1, CBlength))
    fq1.f.umi <- as.vector(subseq(sread(fq1.f), CBlength+1, CBlength+UMIlength))
    
    # Add cell barcode and umi to id of Read2
    #fq2.f@id <- BStringSet(paste(as.vector(fq2.f@id), fq1.f.cell, fq1.f.umi))
    #fq2.f@id <- BStringSet(paste0(sub(" .:N:0:", "_", as.vector(fq2.f@id)), "_", fq1.f.cell, "_", fq1.f.umi))
    # Modify each identifier
    new_ids <- mapply(function(id, cell, umi) {
      parts <- strsplit(id, " ")[[1]]
      # Append cell barcode and UMI to the first part and reconstruct the identifier
      parts[1] <- paste0(parts[1], "_", cell, "_", umi)
      paste(parts, collapse=" ")
    }, as.vector(fq2.f@id), fq1.f.cell, fq1.f.umi)
    
    # Step 3: Assign these new IDs back to the fq2.f object
    fq2.f@id <- BStringSet(as.vector(new_ids))
    
    # Check if all the ids of Read1 and Read2 files match up
    #if(! all(cutf(as.vector(fq1.f@id), d = " ", f = 1) == cutf(as.vector(fq2.f@id), d = " ", f = 1))) stop("Read ID mismatch")
    if(! all(cutf(as.vector(fq1.f@id), d = " ", f = 1) == cutf(as.vector(fq2.f@id), d = "_", f = 1))) stop("Read ID mismatch")
    
    # Save fastq file
    message("\nWriting file ", paste0(Folder, "/", SampleName, ".fastq.gz"), "......\n")
    writeFastq(fq2.f, file = paste0(Folder, "/", SampleName, ".fastq.gz"), mode = "a")
    
    # Clean up memory
    invisible(gc())
  }
  
  end <- Sys.time()
  message("\nEnd time: ", end, "\n")
  message("\nDuration time: ", round(end - start, 3), " hours. \n")
  close(strm1)
  close(strm2)
  message(" done")
  invisible(gc())
}


### Generate and save report. NOTE: report.ls only contains data from the last iteration of the repeat{} loop above.
report.mat <- do.call(rbind, report.ls)
report.mat <- rbind(report.mat, colSums(report.mat))
rownames(report.mat)[nrow(report.mat)] <- "total"
report.mat <- cbind(report.mat, report.mat[,2] / report.mat[,1])
colnames(report.mat) <- c("all", "filtered", "fraction")
write.table(report.mat, file = paste0(Folder, "/", SampleName, ".stats.txt"), sep = "\t", quote = F)

invisible(gc())

message("\nMaintained ", round(report.mat["total", "fraction"]*100, 2), "% of reads.\n")
sessionInfo()
message("\nFinished!")
