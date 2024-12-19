library(tidyverse)
library(Seurat)

load_matrices <- function(
  base_path = NULL,
  ad_mtx_path = file.path(base_path, "cellSNP.tag.AD.mtx"),
  dp_mtx_path = file.path(base_path, "cellSNP.tag.DP.mtx"),
  features_path = file.path(base_path, "cellSNP.variants.tsv"),
  cells_path = file.path(base_path, "cellSNP.samples.tsv")
) {
  # Check and generate 'cellSNP.variants.tsv' if it does not exist
  if (!file.exists(features_path)) {
    vcf_path <- file.path(base_path, "cellSNP.base.vcf.gz")
    if (!file.exists(vcf_path)) {
      stop(paste("VCF file does not exist:", vcf_path))
    }
    
    # Shell command to generate the variants file
    cmd <- sprintf("zcat %s | grep -v '^#' | awk '{print $2$4\">\"$5}' > %s", vcf_path, features_path)
    message("Generating features file using command: ", cmd)
    system(cmd, intern = FALSE)
    
    # Verify the file was created
    if (!file.exists(features_path)) {
      stop("Failed to generate 'cellSNP.variants.tsv'. Please check the shell command.")
    }
  }
  
  # Helper function to read matrix with error handling
  read_mtx_safe <- function(mtx, features, cells, feature.column = 1) {
    if (!file.exists(mtx)) stop(paste("Matrix file does not exist:", mtx))
    if (!file.exists(features)) stop(paste("Features file does not exist:", features))
    if (!file.exists(cells)) stop(paste("Cells file does not exist:", cells))
    
    # Assuming ReadMtx is a predefined function in your environment
    ReadMtx(mtx = mtx, features = features, cells = cells, feature.column = feature.column)
  }
  
  # Read AD and DP matrices
  AD_mtx <- read_mtx_safe(ad_mtx_path, features_path, cells_path)
  DP_mtx <- read_mtx_safe(dp_mtx_path, features_path, cells_path)
  
  # Validate that AD and DP matrices have the same dimensions
  if (!all(dim(AD_mtx) == dim(DP_mtx))) {
    stop("AD and DP matrices have different dimensions.")
  }
  
  # Calculate allele frequency matrix (AF)
  AF_mtx <- AD_mtx / DP_mtx
  AF_mtx[is.na(AF_mtx)] <- 0  # Replace NA values (e.g., from division by zero) with 0
  
  # Return matrices as a list
  return(list(AD = AD_mtx, DP = DP_mtx, AF = AF_mtx))
}

new.wald.test = function (Sigma, b, Terms = NULL, L = NULL, H0 = NULL, df = NULL, verbose = FALSE) 
{
  if (is.null(Terms) & is.null(L)) 
    stop("One of the arguments Terms or L must be used.")
  if (!is.null(Terms) & !is.null(L)) 
    stop("Only one of the arguments Terms or L must be used.")
  if (is.null(Terms)) {
    w <- nrow(L)
    Terms <- seq(length(b))[colSums(L) > 0]
  } else w <- length(Terms)
  if (is.null(H0)) 
    H0 <- rep(0, w)
  if (w != length(H0)) 
    stop("Vectors of tested coefficients and of null hypothesis have different lengths\n")
  if (is.null(L)) {
    L <- matrix(rep(0, length(b) * w), ncol = length(b))
    for (i in 1:w) L[i, Terms[i]] <- 1
  }
  dimnames(L) <- list(paste("L", as.character(seq(NROW(L))), 
                            sep = ""), names(b))
  f <- L %*% b
  V <- Sigma
  #mat <- qr.solve(L %*% V %*% t(L), tol = 1e-10)
  mat <- solve(L %*% V %*% t(L))
  stat <- t(f - H0) %*% mat %*% (f - H0)
  p <- 1 - pchisq(stat, df = w)
  if (is.null(df)) 
    res <- list(chi2 = c(chi2 = stat, df = w, P = p))
  else {
    fstat <- stat/nrow(L)
    df1 <- nrow(L)
    df2 <- df
    res <- list(chi2 = c(chi2 = stat, df = w, P = p), Ftest = c(Fstat = fstat, 
                                                                df1 = df1, df2 = df2, P = 1 - pf(fstat, df1, df2)))
  }
  structure(list(Sigma = Sigma, b = b, Terms = Terms, H0 = H0, 
                 L = L, result = res, verbose = verbose, df = df), class = "wald.test")
}

CheckDf <- function(df) {
  # Check if the number of rows is less than 3
  if (nrow(df) < 3) {
    return(TRUE)
  }
  
  # Ensure the required columns are present
  required_columns <- c("n1", "ref_count")
  if (!all(required_columns %in% colnames(df))) {
    stop(paste("The dataframe must contain the following columns:", paste(required_columns, collapse = ", ")))
  }
  
  # Check for non-numeric columns with only one unique value
  non_numeric_single_value <- sapply(df, function(column) {
    !is.numeric(column) && length(unique(column)) == 1
  })
  
  if (any(non_numeric_single_value)) {
    return(TRUE)
  }
  
  # Check if the 'n1' or 'ref_count' columns sum to zero
  if (sum(df$n1) == 0 || sum(df$ref_count) == 0) {
    return(TRUE)
  }
  
  return(FALSE)  # If none of the conditions are met, proceed
}

supervised_glm <- function(AD_mat = NULL, DP_mat = NULL, clone_mat = NULL, minDP = 20, use_random_effect = FALSE) {
  # Ensure all matrices have proper row and column names
  if (is.null(rownames(AD_mat))) 
    rownames(AD_mat) <- paste0("variant_", seq_len(nrow(AD_mat)))
  if (is.null(rownames(DP_mat))) 
    rownames(DP_mat) <- paste0("variant_", seq_len(nrow(DP_mat)))
  if (is.null(colnames(clone_mat))) 
    colnames(clone_mat) <- paste0("factor_", seq_len(ncol(clone_mat)))
  
  # Check consistency across matrices
  if (!all(colnames(AD_mat) == colnames(DP_mat)) || !all(colnames(AD_mat) == rownames(clone_mat))) {
    stop("Cells in AD, DP, and clone matrices are not identical.")
  }
  if (!all(rownames(AD_mat) == rownames(DP_mat))) {
    stop("Variants in AD and DP matrices are not identical.")
  }
  
  # Initialize result matrices with NA values
  n_variants <- nrow(AD_mat)
  n_clones <- ncol(clone_mat)
  
  result_matrices <- lapply(c("LR_vals", "LRT_pvals", "Wald_pvals", "ANOVA_pvals", "LRT_fdr"), 
                            function(name) matrix(NA, n_variants, n_clones, 
                                                  dimnames = list(rownames(AD_mat), colnames(clone_mat))))
  names(result_matrices) <- c("LR_vals", "LRT_pvals", "Wald_pvals", "ANOVA_pvals", "LRT_fdr")
  
  # Iterate through clones
  for (c in seq_len(n_clones)) {
    message(sprintf("Processing covariate number: %d", c))
    sub_LR_val <- rep(NA, n_variants)
    
    # Iterate through variants
    for (v in seq_len(n_variants)) {
      # Identify valid indices for DP above the threshold
      valid_indices <- DP_mat[v, ] > minDP
      if (sum(valid_indices) == 0) next
      
      # Prepare data subset
      total <- DP_mat[v, valid_indices]
      n1 <- AD_mat[v, valid_indices]
      clone_use <- clone_mat[valid_indices, , drop = FALSE]
      df_use <- data.frame(n1 = n1, total = total, ref_count = total - n1)
      df_use <- cbind(df_use, clone_use)
      df_tmp <- df_use[!is.na(clone_use[, c]), ]
      
      # Check dataframe conditions
      if (CheckDf(df_tmp)) next
      
      # Fit null model
      formula_fm0 <- as.formula("cbind(n1, ref_count) ~ 1")
      fm0 <- tryCatch(aod::betabin(formula_fm0, ~1, data = df_tmp, warnings = FALSE),
                      error = function(e) NULL)
      if (is.null(fm0)) next
      
      # Fit alternative model
      if (!use_random_effect) {
        formula_fm1 <- as.formula(paste0("cbind(n1, ref_count) ~ 1 + ", colnames(clone_mat)[c]))
        fm1 <- tryCatch(aod::betabin(formula_fm1, ~1, data = df_tmp, warnings = FALSE),
                        error = function(e) NULL)
        if (is.null(fm1)) next
        
        # Wald test
        wald_res <- tryCatch(new.wald.test(b = aod::coef(fm1), Sigma = aod::vcov(fm1), Terms = 2:length(aod::coef(fm1))),
                             error = function(e) NULL)
        if (!is.null(wald_res)) {
          result_matrices$Wald_pvals[v, c] <- wald_res$result$chi2[3]
        }
      } else {
        random_effect <- as.formula(paste0("~", colnames(clone_mat)[c]))
        fm1 <- tryCatch(aod::betabin(formula_fm0, random_effect, data = df_tmp, warnings = FALSE),
                        error = function(e) NULL)
        if (is.null(fm1)) next
        
        # ANOVA test
        anova_res <- tryCatch(anova(fm0, fm1), error = function(e) NULL)
        if (!is.null(anova_res)) {
          result_matrices$ANOVA_pvals[v, c] <- anova_res@anova.table$`P(> Chi2)`[2]
        }
      }
      
      # Likelihood Ratio Test
      if (!is.null(fm1)) {
        sub_LR_val[v] <- fm0@dev - fm1@dev
        result_matrices$LRT_pvals[v, c] <- tryCatch(
          pchisq(sub_LR_val[v], df = fm1@nbpar - fm0@nbpar, lower.tail = FALSE, log.p = FALSE),
          error = function(e) NA
        )
      }
    }
    
    # Store likelihood ratio values
    result_matrices$LR_vals[, c] <- sub_LR_val
  }
  
  # Adjust p-values for multiple testing
  result_matrices$LRT_fdr[] <- p.adjust(result_matrices$LRT_pvals, method = "fdr")
  
  # Return result matrices
  return(result_matrices)
}
