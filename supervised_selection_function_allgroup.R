# FindAllVariants helper function for cell grouping error checking
ValidateAllCellGroups <- function(AD_mat, DP_mat, clone_mat, min.cells.group) {
  if (!all(colnames(AD_mat) == colnames(DP_mat))) {
    stop("Cells in AD and DP matrics are not identical")
  }
  if (!all(rownames(AD_mat) == rownames(DP_mat))) {
    stop("Variants in AD and DP matrics are not identical")
  }
  mis_label = c()
  for (label in unique(clone_mat$cell_label)) {
    cells = clone_mat[which(clone_mat$cell_label == label),]$cellID
    if (length(match(cells, colnames(AD_mat))) < min.cells.group) {
      mis_label = c(label, mis_label)
    }
  }
  if (length(mis_label) != 0) {
    message("The following label(s) have less than required cells, they will be excluded for further testing: ",
            paste(mis_label, collapse = ", "))
  }
  return(unique(clone_mat$cell_label)[!unique(clone_mat$cell_label) %in% mis_label])
}

# example
# library(dplyr)
# 
# # Sample data
# ad_mat <- data.frame(
#   Cell1 = c(10, 20, 30),
#   Cell2 = c(15, 25, 35),
#   Cell3 = c(10, 20, 30),
#   Cell4 = c(15, 25, 35)
# )
# rownames(ad_mat) <- c("Variant1", "Variant2", "Variant3")
# 
# dp_mat <- data.frame(
#   Cell1 = c(100, 200, 300),
#   Cell2 = c(150, 250, 350),
#   Cell3 = c(100, 200, 300),
#   Cell4 = c(150, 250, 350)
# )
# rownames(dp_mat) <- c("Variant1", "Variant2", "Variant3")
# 
# clone_mat <- data.frame(
#   cellID = c("Cell1", "Cell2", "Cell3", "Cell4"),
#   cell_label = c("Label1", "Label1", "Label2", "Label3")
# )
# 
# min.cells.group <- 2
# 
# # Validate cell groups
# valid_labels <- ValidateAllCellGroups(ad_mat, dp_mat, clone_mat, min.cells.group)
# print("Valid labels:")
# print(valid_labels)


performMultiLevelStatisticalTests <- function(df_use, use_random_effect) {
  library(aod)
  df_use$cell_label = as.factor(df_use$cell_label)
  formula_fm0 <- as.formula("cbind(n1, ref_count) ~ 1")
  fm0 <- aod::betabin(formula_fm0, ~1, data = df_use, warnings = FALSE)
  if (!use_random_effect) {
    formula_fm1 <- as.formula("cbind(n1, ref_count) ~ 1 + cell_label")
    # Use tryCatch to handle errors
    fm1 <- tryCatch({
      # Attempt to fit the model
      aod::betabin(formula_fm1, ~1, data = df_use, warnings = FALSE)
    }, error = function(e) {
      # If an error occurs, return NA
      return(NA)
    })
    } else {
    random_effect <- as.formula("~ cell_label")
    fm1 <- tryCatch({
      # Attempt to fit the model
      aod::betabin(formula_fm0, random_effect, data = df_use, warnings = FALSE)
    }, error = function(e) {
      # If an error occurs, return NA
      return(NA)
    })
    }
  if (is.na(fm1)) {
    pvals = rep(NA, length(levels(df_use$cell_label)))
  } else {
    pvals = summary(fm1)@Coef[, "Pr(> |z|)"]
  }
  names(pvals) = levels(df_use$cell_label)
  return(pvals)
}

add_row_to_matrix <- function(mat, row) {
  # Ensure that 'row' is a named vector
  if (is.null(names(row))) {
    stop("The 'row' vector must have names.")
  }
  
  # Prepare a new row with NA values
  new_row <- setNames(rep(NA, ncol(mat)), colnames(mat))
  
  # Find matching names and assign directly to new_row based on those names
  intersecting_names <- intersect(names(row), names(new_row))
  if (length(intersecting_names) > 0) {
    new_row[intersecting_names] <- row[intersecting_names]
    updated_mat <- rbind(mat, new_row)
  } else {
    warning("No matching column names found. Row not added.")
    updated_mat <- mat # Return the original matrix unchanged
  }
  return(updated_mat)
}


#load("sample_multi_AD_DP_clone_mat.RData")

# The FindAllVariants here calculate pvals for all groups simultanously
FindAllVariants <- function(
  AD_mat, DP_mat, clone_mat,
  minDP = 20,
  min.cells.group = 2,
  #max.cells.per.ident = Inf,
  #random.seed = 123,
  use_random_effect = FALSE
) {
  ## check clone_mat, check whether there are NA in cell_label
  if (any(is.na(clone_mat$cell_label))) {
    message("NA values found in clone_mat$cell_label. Omitting rows with NA...\n")
    clone_mat <- na.omit(sample_clone_mat)
  } 
  ## check whether variant name exist
  if (is.null(rownames(AD_mat))) 
    rownames(AD_mat) <- paste0("variant_", seq(nrow(AD_mat)))
  if (is.null(rownames(DP_mat))) 
    rownames(DP_mat) <- paste0("variant_", seq(nrow(DP_mat)))
  ## validation
  labels = ValidateAllCellGroups(
    AD_mat = AD_mat,
    DP_mat = DP_mat,
    clone_mat = clone_mat,
    min.cells.group = min.cells.group
  )
  clone_mat = clone_mat[which(clone_mat$cell_label %in% labels),]
  ## Initialize matrices for results
  labels = unique(clone_mat$cell_label)
  V <- nrow(AD_mat)  # Number of variants
  res_mat = matrix(nrow = 0, ncol = length(labels))
  colnames(res_mat) = labels
  ## Iterate through variants, calculate p.value
  for (v in seq_len(V)) {
    message(paste("Processing the ", str(v), " variants..."))
    total = DP_mat[v, clone_mat$cellID]
    n1 = AD_mat[v, clone_mat$cellID]
    cell_label = clone_mat$cell_label
    df_tmp = data.frame(total = total, n1 = n1, cell_label = cell_label)
    df_use = df_tmp[which(df_tmp$total > minDP),]
    df_use$ref_count <- df_use$total - df_use$n1
    if(length(unique(df_use$cell_label)) <= 1) {
      vec_NA = rep(NA,length(labels))
      names(vec_NA) = labels
      res_mat = add_row_to_matrix(res_mat, vec_NA)
    } else {
      res_test = performMultiLevelStatisticalTests(df_use = df_use, use_random_effect = use_random_effect)
      res_mat = add_row_to_matrix(res_mat, res_test)
    }
  }
  rownames(res_mat) = rownames(AD_mat)
  return(res_mat)
}

#res = FindAllVariants(sample_AD, sample_DP, sample_clone_mat)