library(tidyverse)
library(Seurat)

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

supervised_glm = function(AD_mat = NULL, DP_mat = NULL, clone_mat = NULL, minDP = 20, use_random_effect = FALSE){
  if (is.null(rownames(AD_mat))) 
    rownames(AD_mat) <- paste0("variant_", seq(nrow(AD_mat)))
  if (is.null(rownames(DP_mat))) 
    rownames(DP_mat) <- paste0("variant_", seq(nrow(DP_mat)))
  if (is.null(colnames(clone_mat))) colnames(clone_mat) <- paste0("factor_", seq(ncol(clone_mat)))
  
  LR_vals <- matrix(NA, nrow(AD_mat), ncol(clone_mat))
  LRT_pvals <- matrix(NA, nrow(AD_mat), ncol(clone_mat))
  Wald_pvals <- matrix(NA, nrow(AD_mat), ncol(clone_mat))
  ANOVA_pvals <- matrix(NA, nrow(AD_mat), ncol(clone_mat))
  LRT_fdr <- matrix(NA, nrow(AD_mat), ncol(clone_mat))
  
  rownames(LR_vals) <- rownames(LRT_pvals) <- rownames(Wald_pvals) <- rownames(ANOVA_pvals) <- rownames(LRT_fdr) <- rownames(AD_mat)
  colnames(LR_vals) <- colnames(LRT_pvals) <- colnames(Wald_pvals) <- colnames(ANOVA_pvals) <- colnames(LRT_fdr) <- colnames(clone_mat)
  
  V <- nrow(AD_mat) ## number of variants
  for (c in seq_len(ncol(clone_mat))) {  ## iterate through different clone
    sub_LR_val = rep(NA, V)
    for (v in seq_len(nrow(AD_mat))) {
      #for (v in 3848:nrow(AD_mat)) {  
      if(sum(DP_mat[v,]>minDP) == 0) {
        next
      }
      total = DP_mat[v,DP_mat[v,]>minDP]
      n1 = AD_mat[v,DP_mat[v,]>minDP]
      clone_use = clone_mat[DP_mat[v,] > minDP,,drop = FALSE]
      #if (!is.matrix(clone_use))
      df_use = data.frame(n1 = n1, total = total)
      df_use$ref_count <- df_use$total - df_use$n1
      df_use <- cbind(df_use, clone_use)
      df_tmp <- df_use[!is.na(clone_use[,c]), ]
      if(nrow(df_tmp[df_tmp$n1>0,]) <= length(unique(clone_use[,c]))*2) {
        next
      }
      if(length(unique(df_tmp[,colnames(clone_mat)[c]])) == 1){
        next
        if (sum(df_tmp$n1) > 0) {
          Wald_pvals[v,c] = 0.013
          sub_LR_val[v] = 0.013
        }
      }
      formula_fm0 <- as.formula("cbind(n1, ref_count) ~ 1")
      fm0 <- aod::betabin(formula_fm0, ~1, data = df_tmp, warnings = FALSE)
      if (use_random_effect == FALSE) {
        formula_fm1 <- as.formula(paste0("cbind(n1, ref_count)", "~ 1+", colnames(clone_mat)[c], sep = ""))
        fm1 <- aod::betabin(formula_fm1, ~1, data = df_tmp, warnings = FALSE)
        wald_res = new.wald.test(b = aod::coef(fm1), Sigma = aod::vcov(fm1), Terms = 2:length(aod::coef(fm1)))
        Wald_pvals[v,c] = wald_res$result$chi2[3]
      } else {
        random_effect <- as.formula(paste0("~", colnames(clone_mat)[c]))
        fm1 <- aod::betabin(formula_fm0, random_effect, data = df_tmp, warnings = FALSE)
        anova_res = anova(fm0, fm1)
        ANOVA_pvals[v,c] = anova_res@anova.table$`P(> Chi2)`[2]
      }
      # likelihood ratio test
      sub_LR_val[v] <- fm0@dev - fm1@dev
    }
    LR_vals[, c] <- sub_LR_val
    LRT_pvals[, c] <- pchisq(sub_LR_val, df = fm1@nbpar-fm0@nbpar, lower.tail = FALSE, log.p = FALSE)
  }
  LRT_fdr[, ] <- p.adjust(LRT_pvals, method = "fdr")
  res <- list(Wald_pvals = Wald_pvals, ANOVA_pvals =  ANOVA_pvals, LR_vals = LR_vals, LRT_pvals = LRT_pvals, LRT_fdr = LRT_fdr)
  return(res)
}
