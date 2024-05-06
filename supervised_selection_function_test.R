# Assuming the ValidateCellGroups and IdentsToCells functions are defined as given

# Helper function to simulate matrix creation for testing
createTestMatrix <- function(cells, variants) {
  matrix(runif(length(cells) * length(variants)), nrow = length(variants), ncol = length(cells),
         dimnames = list(variants, cells))
}

testValidateCellGroupsValidInput <- function() {
  # Correct setup for AD_mat and DP_mat to include cell names matching cells.1 and cells.2
  AD_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  DP_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  cells.1 <- "Cell1"
  cells.2 <- "Cell2"
  min.cells.group <- 1
  
  # Correcting the error in the test: Ensure cells.1 and cells.2 are defined as vectors
  cells.1 <- c("Cell1")
  cells.2 <- c("Cell2")
  
  tryCatch({
    ValidateCellGroups(AD_mat, DP_mat, cells.1, cells.2, min.cells.group)
    cat("Test 1 Passed: ValidateCellGroups with valid input\n")
  }, error = function(e) {
    cat("Test 1 Failed:", e$message, "\n")
  })
}

# Additional Tests for ValidateCellGroups

# Test 2: Non-identical Cell Names in Matrices
testValidateCellGroupsNonIdenticalCells <- function() {
  AD_mat <- createTestMatrix(c("Cell1", "Cell3"), c("Var1", "Var2"))
  DP_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  cells.1 <- c("Cell1")
  cells.2 <- c("Cell3")
  min.cells.group <- 1
  
  tryCatch({
    ValidateCellGroups(AD_mat, DP_mat, cells.1, cells.2, min.cells.group)
    cat("Test 2 Failed: Expected an error for non-identical cell names\n")
  }, error = function(e) {
    cat("Test 2 Passed: ", e$message, "\n")
  })
}

# Test 3: Non-identical Variant Names in Matrices
testValidateCellGroupsNonIdenticalVariants <- function() {
  AD_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var3"))
  DP_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  cells.1 <- c("Cell1")
  cells.2 <- c("Cell2")
  min.cells.group <- 1
  
  tryCatch({
    ValidateCellGroups(AD_mat, DP_mat, cells.1, cells.2, min.cells.group)
    cat("Test 3 Failed: Expected an error for non-identical variant names\n")
  }, error = function(e) {
    cat("Test 3 Passed: ", e$message, "\n")
  })
}

# Test for bad.cells in cells.1
testValidateCellGroupsBadCells1 <- function() {
  AD_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  DP_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  cells.1 <- c("Cell1", "CellX") # CellX does not exist
  cells.2 <- c("Cell2")
  min.cells.group <- 1
  
  tryCatch({
    ValidateCellGroups(AD_mat, DP_mat, cells.1, cells.2, min.cells.group)
    cat("Test for bad.cells in cells.1 Failed: Expected an error\n")
  }, error = function(e) {
    cat("Test for bad.cells in cells.1 Passed: ", e$message, "\n")
  })
}

# Test for bad.cells in cells.2
testValidateCellGroupsBadCells2 <- function() {
  AD_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  DP_mat <- createTestMatrix(c("Cell1", "Cell2"), c("Var1", "Var2"))
  cells.1 <- c("Cell1")
  cells.2 <- c("Cell2", "CellY") # CellY does not exist
  min.cells.group <- 1
  
  tryCatch({
    ValidateCellGroups(AD_mat, DP_mat, cells.1, cells.2, min.cells.group)
    cat("Test for bad.cells in cells.2 Failed: Expected an error\n")
  }, error = function(e) {
    cat("Test for bad.cells in cells.2 Passed: ", e$message, "\n")
  })
}

# Test 1 for IdentsToCells: Valid input for both identifiers
testIdentsToCellsValidInput <- function() {
  clone_mat <- data.frame(cellID = c("Cell1", "Cell2", "Cell3"), cell_label = c("ident1_cell1", "ident2_cell1", "ident3_cell1"))
  ident.1 <- "ident1_cell1"
  ident.2 <- "ident2_cell1"
  
  tryCatch({
    result <- IdentsToCells(clone_mat, ident.1, ident.2)
    if (!is.null(result$cells.1) && !is.null(result$cells.2)) {
      cat("Test IdentsToCells Valid Input Passed\n")
    } else {
      cat("Test IdentsToCells Valid Input Failed: Unexpected result\n")
    }
  }, error = function(e) {
    cat("Test IdentsToCells Valid Input Failed:", e$message, "\n")
  })
}

# Additional Tests for IdentsToCells

# Test 2: ident.1 Not Provided
testIdentsToCellsIdent1NotProvided <- function() {
  clone_mat <- data.frame(cellID = c("Cell1", "Cell2"), cell_label = c("ident1_cell1", "ident2_cell1"))
  ident.1 <- NULL
  ident.2 <- "ident2_cell1"
  
  tryCatch({
    IdentsToCells(clone_mat, ident.1, ident.2)
    cat("Test IdentsToCells ident.1 Not Provided Failed: Expected an error\n")
  }, error = function(e) {
    cat("Test IdentsToCells ident.1 Not Provided Passed: ", e$message, "\n")
  })
}

# Test 3: ident.2 Not Provided
testIdentsToCellsIdent2NotProvided <- function() {
  clone_mat <- data.frame(cellID = c("Cell1", "Cell2", "Cell3"), cell_label = c("ident1_cell1", "ident2_cell1", "ident3_cell1"))
  ident.1 <- "ident1_cell1"
  ident.2 <- NULL
  
  tryCatch({
    result <- IdentsToCells(clone_mat, ident.1, ident.2)
    # Expect cells.2 to contain all other cells not in ident.1
    if (!is.null(result$cells.2) && setdiff(clone_mat$cellID, result$cells.1) == result$cells.2) {
      cat("Test IdentsToCells ident.2 Not Provided Passed\n")
    } else {
      cat("Test IdentsToCells ident.2 Not Provided Failed: cells.2 did not contain expected values\n")
    }
  }, error = function(e) {
    cat("Test IdentsToCells ident.2 Not Provided Failed: ", e$message, "\n")
  })
}

# Test for bad.cells in ident.1
testIdentsToCellsBadIdent1 <- function() {
  clone_mat <- data.frame(cell_label = c("ident1_cell1", "ident2_cell1"), cellID = c("Cell1", "Cell2"))
  ident.1 <- c("ident1_cell1", "identX") # identX does not exist
  ident.2 <- "ident2_cell1"
  
  tryCatch({
    IdentsToCells(clone_mat, ident.1, ident.2)
    cat("Test for bad.cells in ident.1 Failed: Expected an error\n")
  }, error = function(e) {
    cat("Test for bad.cells in ident.1 Passed: ", e$message, "\n")
  })
}

# Test for bad.cells in ident.2
testIdentsToCellsBadIdent2 <- function() {
  clone_mat <- data.frame(cell_label = c("ident1_cell1", "ident2_cell1"), cellID = c("Cell1", "Cell2"))
  ident.1 <- "ident1_cell1"
  ident.2 <- c("ident2_cell1", "identY") # identY does not exist
  
  tryCatch({
    IdentsToCells(clone_mat, ident.1, ident.2)
    cat("Test for bad.cells in ident.2 Failed: Expected an error\n")
  }, error = function(e) {
    cat("Test for bad.cells in ident.2 Passed: ", e$message, "\n")
  })
}

# Test performStatisticalTests with mock data
test_performStatisticalTests <- function() {
  library(aod)
  
  # Mock data setup
  df_use <- data.frame(n1 = c(rep(0, 4), rep(1, 5)), ref_count = c(rep(5, 9)), cell_label = gl(3, 3))
  df = NA
  # Without random effect
  results_without_random_effect <- performStatisticalTests(df_use, FALSE, df)
  if (!is.na(results_without_random_effect[1])) {
    cat("Test passed: performStatisticalTests without random effect produces expected results.\n")
  } else {
    cat("Test failed: performStatisticalTests without random effect did not produce expected results.\n")
  }
  
  # With random effect
  results_with_random_effect <- performStatisticalTests(df_use, TRUE, df)
  if (!is.na(results_with_random_effect[1])) {
    cat("Test passed: performStatisticalTests with random effect produces expected results.\n")
  } else {
    cat("Test failed: performStatisticalTests with random effect did not produce expected results.\n")
  }
  
  # Test with insufficient non-zero n1 values
  df_use_insufficient <- df_use
  df_use_insufficient$n1 <- rep(0, nrow(df_use_insufficient))  # All zeros
  results_insufficient <- performStatisticalTests(df_use_insufficient, FALSE, df)
  if (all(is.na(results_insufficient[[1]]))) {
    cat("Test passed: performStatisticalTests correctly handles insufficient non-zero n1 values.\n")
  } else {
    cat("Test failed: performStatisticalTests failed to handle insufficient non-zero n1 values.\n")
  }
}

testValidateAllCellGroups <- function() {
  # Mock data setup for matching AD_mat and DP_mat
  AD_mat <- matrix(1:4, 2, 2, dimnames = list(c("V1", "V2"), c("C1", "C2")))
  DP_mat <- matrix(5:8, 2, 2, dimnames = list(c("V1", "V2"), c("C1", "C2")))
  clone_mat <- data.frame(cell_label = c("A", "A", "B", "B"), cellID = c("C1", "C2", "C1", "C2"))
  min_cells_group <- 1
  
  # Test 1: AD and DP matrices with identical row and column names
  tryCatch({
    valid_labels <- ValidateAllCellGroups(AD_mat, DP_mat, clone_mat, min_cells_group)
    if (length(valid_labels) == length(unique(clone_mat$cell_label))) {
      cat("Test 1 passed: AD and DP matrices with identical row and column names.\n")
    } else {
      cat("Test 1 failed: Unexpected number of valid labels.\n")
    }
  }, error = function(e) {
    cat("Test 1 failed: ", e$message, "\n")
  })
  
  # Test 2: Non-matching column names in AD and DP matrices
  AD_mat_diff_cols <- matrix(1:4, 2, 2, dimnames = list(c("V1", "V2"), c("C1", "C3")))
  tryCatch({
    ValidateAllCellGroups(AD_mat_diff_cols, DP_mat, clone_mat, min_cells_group)
    cat("Test 2 failed: Function did not stop as expected with non-matching column names.\n")
  }, error = function(e) {
    cat("Test 2 passed: Correctly identified non-matching column names.\n")
  })
  
  # Test 3: Label with insufficient cells
  clone_mat_insufficient_cells <- data.frame(cell_label = c("A", "B", "C"), cellID = c("C1", "C2", "C3"))
  tryCatch({
    valid_labels <- ValidateAllCellGroups(AD_mat, DP_mat, clone_mat_insufficient_cells, 2) # Increased min_cells_group to force failure
    if ("C" %in% valid_labels) {
      cat("Test 3 failed: Insufficient cells for a label did not exclude it.\n")
    } else {
      cat("Test 3 passed: Correctly excluded labels with insufficient cells.\n")
    }
  }, error = function(e) {
    cat("Test 3 failed: ", e$message, "\n")
  })
  
  # Note: Ensure that the ValidateAllCellGroups function is loaded into the R environment before running this test function.
}


# Execute the tests
testValidateCellGroupsValidInput()
testValidateCellGroupsNonIdenticalCells()
testValidateCellGroupsNonIdenticalVariants()
testValidateCellGroupsBadCells1()
testValidateCellGroupsBadCells2()
testIdentsToCellsValidInput()
testIdentsToCellsIdent1NotProvided()
testIdentsToCellsIdent2NotProvided()
testIdentsToCellsBadIdent1()
testIdentsToCellsBadIdent2()
test_performStatisticalTests()
testValidateAllCellGroups()
