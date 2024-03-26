#load("sample_AD_DP_clone_mat.RData")

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

# Helper function for FindVariants. Convert idents to cells
IdentsToCells <- function(clone_mat, ident.1, ident.2 = NULL) {
  if (is.null(x = ident.1)) {
    stop("Please provide ident.1")
  } else if (is.null(x = ident.2)) {
    message("ident.2 is not provided. Using all the rest cells as reference.")
  }
  if (length(ident.1) > 1 && any(!ident.1 %in% clone_mat$cell_label)) {
    bad.cells <- setdiff(ident.1, unique(clone_mat$cell_label))
    if (length(x = bad.cells) > 0) {
      stop(paste0("The following cell labels provided to ident.1 are not present in the clone matrix: ", paste(bad.cells, collapse = ", ")))
    }
  } else {
    cells.1 <- clone_mat[which(clone_mat$cell_label %in% ident.1), ]$cellID
  }
  # if NULL for ident.2, use all other cells
  if (length(ident.2) > 1 && any(!ident.2 %in% clone_mat$cell_label)) {
    bad.cells <- setdiff(ident.2, unique(clone_mat$cell_label))
    if (length(x = bad.cells) > 0) {
      stop(paste0("The following cell labels provided to ident.2 are not present in the clone matrix: ", paste(bad.cells, collapse = ", ")))
    }
  } else {
    if (is.null(x = ident.2)) {
      cells.2 <- setdiff(x = clone_mat$cellID, y = cells.1)
    } else {
      cells.2 <- clone_mat[which(clone_mat$cell_label %in% ident.2), ]$cellID
    }
  }
  return(list(cells.1 = cells.1, cells.2 = cells.2))
}

# FindVariants helper function for cell grouping error checking
ValidateCellGroups <- function(AD_mat, DP_mat, cells.1, cells.2, min.cells.group) {
  if (!all(colnames(AD_mat) == colnames(DP_mat))) {
    stop("Cells in AD and DP matrics are not identical")
  }
  if (!all(rownames(AD_mat) == rownames(DP_mat))) {
    stop("Variants in AD and DP matrics are not identical")
  }
  if (!any(colnames(AD_mat) %in% cells.1)) {
    stop("Cell group 1 is empty - no cells with identity class ", cells.1)
  } else if (!any(colnames(AD_mat) %in% cells.2)) {
    stop("Cell group 2 is empty - no cells with identity class ", cells.2)
    return(NULL)
  } else if (length(match(cells.1, colnames(AD_mat))) < min.cells.group) {
    stop("Cell group 1 has fewer than ", min.cells.group, " cells")
  } else if (length(match(cells.2, colnames(AD_mat))) < min.cells.group) {
    stop("Cell group 2 has fewer than ", min.cells.group, " cells")
  } else if (any(!cells.1 %in% colnames(AD_mat))) {
    bad.cells <- setdiff(cells.1, colnames(AD_mat))
    stop(
      "The following cell names provided to cells.1 are not present: ",
      paste(bad.cells, collapse = ", ")
    )
  } else if (any(!cells.2 %in% colnames(AD_mat))) {
    bad.cells <- setdiff(cells.2, colnames(AD_mat))
    stop(
      "The following cell names provided to cells.2 are not present: ",
      paste(bad.cells, collapse = ", ")
    )
  }
}

## helper function to add matrix to each row 
add_row_to_matrix <- function(matrix, row) {
  if (is.null(matrix) || nrow(matrix) == 0) {
    # If the matrix is empty, convert the row to a matrix and return
    return(matrix(row, nrow = 1))
  } else {
    # If the matrix is not empty, use rbind to add the new row
    return(rbind(matrix, row))
  }
}

## perform test based on betabin GLM
performStatisticalTests <- function(df_use, use_random_effect, df) { ## df here is degree of freedom
  library(aod)
  if(sum(df_use$n1 != 0) < 5) {
    return(list(vals = rep(NA, 2), df = df))
  }
  formula_fm0 <- as.formula("cbind(n1, ref_count) ~ 1")
  fm0 <- aod::betabin(formula_fm0, ~1, data = df_use, warnings = FALSE)
  if (!use_random_effect) {
    formula_fm1 <- as.formula("cbind(n1, ref_count) ~ 1 + cell_label")
    fm1 <- aod::betabin(formula_fm1, ~1, data = df_use, warnings = FALSE)
    ## wald test
    if (is.na(fm1@varparam[1,1])){
      nonLR_pval = NA
    } else {
      wald_res <- new.wald.test(b = coef(fm1), Sigma = vcov(fm1), Terms = 2)
      nonLR_pval <- wald_res$result$chi2[3]
    }
  } else {
    random_effect <- as.formula("~ cell_label")
    fm1 <- aod::betabin(formula_fm0, random_effect, data = df_use, warnings = FALSE)
    ## ANOVA test
    anova_res <- anova(fm0, fm1)
    nonLR_pval <- anova_res@anova.table$`P(> Chi2)`[2]
  }
  # likelihood ratio test
  LR_val <- fm0@dev - fm1@dev
  df <- fm1@nbpar - fm0@nbpar
  return(list(vals = c(nonLR_pval, LR_val), df = df))
}

FindVariants <- function(
  AD_mat, DP_mat, clone_mat,
  ident.1 = NULL,
  ident.2 = NULL,
  minDP = 20,
  min.cells.group = 3,
  max.cells.per.ident = Inf,
  random.seed = 123,
  use_random_effect = FALSE
) {
  ## check clone_mat, check whether there are NA in cell_label
  if (any(is.na(clone_mat$cell_label))) {
    message("NA values found in clone_mat$cell_label. Omitting rows with NA...\n")
    clone_mat <- na.omit(clone_mat)
  } 
  ## extract cellID
  cells = IdentsToCells(
    clone_mat = clone_mat, 
    ident.1 = ident.1, 
    ident.2 = ident.2)
  ## check whether variant name exist
  if (is.null(rownames(AD_mat))) 
    rownames(AD_mat) <- paste0("variant_", seq(nrow(AD_mat)))
  if (is.null(rownames(DP_mat))) 
    rownames(DP_mat) <- paste0("variant_", seq(nrow(DP_mat)))
  ## validation
  ValidateCellGroups(
    AD_mat = AD_mat,
    DP_mat = DP_mat,
    cells.1 = cells$cells.1,
    cells.2 = cells$cells.2,
    min.cells.group = min.cells.group
  )
  ## subsample cell groups if they are too large
  if (max.cells.per.ident < Inf) {
    set.seed(seed = random.seed)
    if (length(x = cells.1) > max.cells.per.ident) {
      cells.1 <- sample(x = cells.1, size = max.cells.per.ident)
    }
    if (length(x = cells.2) > max.cells.per.ident) {
      cells.2 <- sample(x = cells.2, size = max.cells.per.ident)
    }
  }
  ## Initialize matrices for results
  V <- nrow(AD_mat)  # Number of variants
  res_mat = matrix(nrow = 0, ncol = 0)
  df = NA
  ## Iterate through variants, calculate p.value
  for (v in seq_len(V)) {
    #message(paste("Processing the ", str(v), " variants..."))
    total = c(DP_mat[v, cells$cells.1], DP_mat[v, cells$cells.2])
    n1 = c(AD_mat[v, cells$cells.1], AD_mat[v, cells$cells.2])
    cell_label = c(rep("traget", length(cells$cells.1)), rep("reference", length(cells$cells.2)))
    df_tmp = data.frame(total = total, n1 = n1, cell_label = cell_label)
    df_use = df_tmp[which(df_tmp$total > minDP),]   ## this step might also remove some clones
    df_use$ref_count <- df_use$total - df_use$n1
    if(length(unique(df_use$cell_label)) == 1) {
      res_mat = add_row_to_matrix(res_mat, rep(NA,2))
    } else {
      res_test = performStatisticalTests(df_use = df_use, use_random_effect = use_random_effect, df = df)
      res_mat = add_row_to_matrix(res_mat, res_test$vals)
    }
  }
  if (!use_random_effect) {
    colnames(res_mat) = c("Wald_pval", "LR_val")
  } else {
    colnames(res_mat) = c("ANOVA_pval", "LR_val")
  }
  rownames(res_mat) = rownames(AD_mat)
  res_mat = na.omit(res_mat)
  LRT_pval = pchisq(res_mat[,2], df = res_test$df, lower.tail = FALSE, log.p = FALSE)
  LRT_fdr = p.adjust(LRT_pval, method = "fdr")
  res_mat = cbind(res_mat, LRT_pval)
  res_mat = cbind(res_mat, LRT_fdr)
  
  return(res_mat)
}

#load("sample_multi_AD_DP_clone_mat.RData")

## The FindAllVariants here is apply find FindVariants for each group using 1vsAll mode.
FindAllVariants <- function(
  AD_mat, DP_mat, clone_mat,
  minDP = 20,
  min.cells.group = 3,
  max.cells.per.ident = Inf,
  random.seed = 123,
  use_random_effect = FALSE
) {
  groups = unique(clone_mat$cell_label)
  resL = list()
  for (group in groups) {
    message(paste0("Processing ", group, " ......"))
    sub_res = tryCatch({
      FindVariants(AD_mat = AD_mat, DP_mat = DP_mat, clone_mat = clone_mat, 
                   ident.1 = group, minDP = minDP, 
                   min.cells.group = min.cells.group, 
                   max.cells.per.ident = max.cells.per.ident, 
                   random.seed = random.seed, 
                   use_random_effect = use_random_effect)
    }, error = function(e) {
      cat(e$message, ", skipping to the next iteration.\n")
      NULL # Return NULL or any indicator of failure)
    })
    if (is.null(sub_res)){
      next
    } else {
      sub_res = data.frame(sub_res)
      sub_res$group = group
      sub_res$variant = rownames(sub_res)
      resL[[group]] = sub_res
    }
  }
  res = do.call(rbind, resL)
  return(res)
}

#a = FindAllVariants(AD_mat = sample_AD, DP_mat = sample_DP, clone_mat = sample_clone_mat)

