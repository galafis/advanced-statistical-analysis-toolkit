# R Unit Tests
# Advanced Statistical Analysis Toolkit
# Author: Gabriel Demetrios Lafis

library(testthat)

# Source R files
source("../R/hypothesis_testing.R")
source("../R/regression_analysis.R")
source("../R/multivariate_analysis.R")
source("../R/resampling_methods.R")
source("../R/utils.R")

cat("Running R unit tests...\n\n")

# Test Hypothesis Testing Functions
test_that("T-test works correctly", {
  set.seed(42)
  group1 <- rnorm(30, mean = 100, sd = 15)
  group2 <- rnorm(30, mean = 105, sd = 15)
  
  result <- perform_t_test(group1, group2)
  
  expect_type(result, "list")
  expect_true("statistic" %in% names(result))
  expect_true("p_value" %in% names(result))
  expect_true("conf_int" %in% names(result))
  expect_true(!is.na(result$statistic))
  expect_true(result$p_value >= 0 && result$p_value <= 1)
})

test_that("ANOVA works correctly", {
  set.seed(42)
  data <- data.frame(
    value = c(rnorm(20, 100), rnorm(20, 105), rnorm(20, 110)),
    group = factor(rep(c("A", "B", "C"), each = 20))
  )
  
  result <- perform_anova(data, value ~ group)
  
  expect_type(result, "list")
  expect_true("anova_table" %in% names(result))
  expect_true("model" %in% names(result))
})

test_that("Chi-square test works correctly", {
  table <- matrix(c(30, 20, 15, 35), nrow = 2)
  
  result <- perform_chi_square(table)
  
  expect_type(result, "list")
  expect_true("statistic" %in% names(result))
  expect_true("p_value" %in% names(result))
  expect_true("df" %in% names(result))
  expect_true(result$statistic >= 0)
  expect_true(result$p_value >= 0 && result$p_value <= 1)
})

test_that("Normality test works correctly", {
  set.seed(42)
  normal_data <- rnorm(100, mean = 50, sd = 10)
  
  result <- test_normality(normal_data)
  
  expect_type(result, "list")
  expect_true("shapiro" %in% names(result))
  expect_true("ks" %in% names(result))
  expect_true("is_normal" %in% names(result))
  expect_type(result$is_normal, "logical")
})

# Test Utility Functions
test_that("Standardization works correctly", {
  x <- c(1, 2, 3, 4, 5)
  z <- standardize(x)
  
  expect_equal(mean(z), 0, tolerance = 1e-10)
  expect_equal(sd(z), 1, tolerance = 1e-10)
})

test_that("Cohen's d calculation works", {
  group1 <- rnorm(30, mean = 100, sd = 15)
  group2 <- rnorm(30, mean = 110, sd = 15)
  
  d <- cohens_d(group1, group2)
  
  expect_type(d, "double")
  expect_true(abs(d) > 0)
})

test_that("Confidence interval calculation works", {
  set.seed(42)
  data <- rnorm(100, mean = 50, sd = 10)
  
  ci <- confidence_interval(data, conf_level = 0.95)
  
  expect_type(ci, "list")
  expect_true("lower" %in% names(ci))
  expect_true("upper" %in% names(ci))
  expect_true(ci$lower < ci$mean)
  expect_true(ci$upper > ci$mean)
})

test_that("Outlier detection works", {
  data <- c(1, 2, 3, 4, 5, 100)  # 100 is an outlier
  
  outliers <- detect_outliers(data)
  
  expect_type(outliers, "logical")
  expect_true(outliers[6])  # Last element should be flagged
})

# Test Resampling Methods
test_that("Bootstrap works correctly", {
  set.seed(42)
  data <- rnorm(50, mean = 100, sd = 15)
  
  result <- perform_bootstrap(data, statistic = mean, R = 100)
  
  expect_type(result, "list")
  expect_true("original" %in% names(result))
  expect_true("bootstrap_mean" %in% names(result))
  expect_true("bootstrap_se" %in% names(result))
  expect_true(!is.na(result$original))
})

test_that("Jackknife works correctly", {
  set.seed(42)
  data <- rnorm(20, mean = 50, sd = 10)
  
  result <- jackknife(data, statistic = mean)
  
  expect_type(result, "list")
  expect_true("original" %in% names(result))
  expect_true("jackknife_mean" %in% names(result))
  expect_true("jackknife_se" %in% names(result))
  expect_equal(length(result$estimates), length(data))
})

test_that("Permutation test works correctly", {
  set.seed(42)
  group1 <- rnorm(30, mean = 100, sd = 15)
  group2 <- rnorm(30, mean = 105, sd = 15)
  
  result <- permutation_test(group1, group2, R = 100)
  
  expect_type(result, "list")
  expect_true("observed_statistic" %in% names(result))
  expect_true("p_value" %in% names(result))
  expect_true("significant" %in% names(result))
  expect_true(result$p_value >= 0 && result$p_value <= 1)
})

# Test Multivariate Analysis
test_that("PCA works correctly", {
  set.seed(42)
  data <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50),
    x3 = rnorm(50),
    x4 = rnorm(50)
  )
  
  result <- perform_pca(data, scale = TRUE)
  
  expect_type(result, "list")
  expect_true("variance_explained" %in% names(result))
  expect_true("loadings" %in% names(result))
  expect_true("scores" %in% names(result))
  expect_equal(sum(result$variance_explained), 1, tolerance = 1e-10)
})

test_that("Correlation analysis works correctly", {
  set.seed(42)
  data <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50),
    x3 = rnorm(50)
  )
  
  result <- correlation_analysis(data)
  
  expect_type(result, "list")
  expect_true("correlation_matrix" %in% names(result))
  expect_true("p_values" %in% names(result))
  expect_equal(nrow(result$correlation_matrix), 3)
  expect_equal(ncol(result$correlation_matrix), 3)
})

cat("\n")
cat("="*70, "\n")
cat("All R tests completed!\n")
cat("="*70, "\n")
