# Resampling Methods Module
# Author: Gabriel Demetrios Lafis

library(boot)

#' Perform bootstrap analysis
#'
#' @param data Numeric vector
#' @param statistic Function to compute statistic
#' @param R Number of bootstrap replicates (default: 10000)
#' @param conf_level Confidence level (default: 0.95)
#' @return Bootstrap results with confidence intervals
perform_bootstrap <- function(data, statistic = mean, R = 10000, conf_level = 0.95) {
  # Define bootstrap function
  boot_stat <- function(data, indices) {
    return(statistic(data[indices]))
  }
  
  # Perform bootstrap
  boot_result <- boot(data = data, statistic = boot_stat, R = R)
  
  # Calculate confidence intervals
  ci_percentile <- boot.ci(boot_result, conf = conf_level, type = "perc")
  ci_bca <- tryCatch({
    boot.ci(boot_result, conf = conf_level, type = "bca")
  }, error = function(e) {
    NULL
  })
  
  return(list(
    original = boot_result$t0,
    bootstrap_mean = mean(boot_result$t),
    bootstrap_se = sd(boot_result$t),
    ci_percentile = if(!is.null(ci_percentile)) ci_percentile$percent[4:5] else NULL,
    ci_bca = if(!is.null(ci_bca)) ci_bca$bca[4:5] else NULL,
    bootstrap_values = boot_result$t,
    boot_object = boot_result
  ))
}

#' Bootstrap comparison of two groups
#'
#' @param group1 First group data
#' @param group2 Second group data
#' @param statistic Function to compute statistic (default: mean)
#' @param R Number of bootstrap replicates (default: 10000)
#' @return Bootstrap comparison results
bootstrap_comparison <- function(group1, group2, statistic = mean, R = 10000) {
  # Combine data
  combined <- list(group1 = group1, group2 = group2)
  
  # Bootstrap function for difference
  boot_diff <- function(data, indices) {
    sample1 <- data$group1[indices[1:length(data$group1)]]
    sample2 <- data$group2[indices[(length(data$group1)+1):length(indices)]]
    return(statistic(sample1) - statistic(sample2))
  }
  
  # Create indices for both groups
  indices_matrix <- matrix(
    sample(1:length(c(group1, group2)), 
           size = length(c(group1, group2)) * R, 
           replace = TRUE),
    nrow = R
  )
  
  # Calculate differences
  differences <- numeric(R)
  for (i in 1:R) {
    sample1 <- sample(group1, replace = TRUE)
    sample2 <- sample(group2, replace = TRUE)
    differences[i] <- statistic(sample1) - statistic(sample2)
  }
  
  # Calculate CI
  ci <- quantile(differences, c(0.025, 0.975))
  
  # P-value (proportion of differences that cross zero)
  p_value <- min(mean(differences <= 0), mean(differences >= 0)) * 2
  
  return(list(
    observed_diff = statistic(group1) - statistic(group2),
    bootstrap_mean_diff = mean(differences),
    bootstrap_se = sd(differences),
    ci = ci,
    p_value = p_value,
    differences = differences
  ))
}

#' Jackknife resampling
#'
#' @param data Numeric vector
#' @param statistic Function to compute statistic
#' @return Jackknife results
jackknife <- function(data, statistic = mean) {
  n <- length(data)
  
  # Leave-one-out estimates
  jackknife_estimates <- numeric(n)
  for (i in 1:n) {
    jackknife_estimates[i] <- statistic(data[-i])
  }
  
  # Jackknife mean and SE
  jackknife_mean <- mean(jackknife_estimates)
  jackknife_se <- sqrt(((n - 1) / n) * sum((jackknife_estimates - jackknife_mean)^2))
  
  return(list(
    original = statistic(data),
    jackknife_mean = jackknife_mean,
    jackknife_se = jackknife_se,
    estimates = jackknife_estimates,
    bias = (n - 1) * (jackknife_mean - statistic(data))
  ))
}

#' Permutation test for two groups
#'
#' @param group1 First group data
#' @param group2 Second group data
#' @param statistic Function to compute test statistic (default: difference in means)
#' @param R Number of permutations (default: 10000)
#' @return Permutation test results
permutation_test <- function(group1, group2, 
                             statistic = function(x, y) mean(x) - mean(y),
                             R = 10000) {
  # Observed test statistic
  observed_stat <- statistic(group1, group2)
  
  # Combined data
  combined <- c(group1, group2)
  n1 <- length(group1)
  n2 <- length(group2)
  n <- n1 + n2
  
  # Permutation distribution
  perm_stats <- numeric(R)
  for (i in 1:R) {
    # Random permutation
    perm_indices <- sample(1:n)
    perm_group1 <- combined[perm_indices[1:n1]]
    perm_group2 <- combined[perm_indices[(n1+1):n]]
    perm_stats[i] <- statistic(perm_group1, perm_group2)
  }
  
  # P-value (two-tailed)
  p_value <- mean(abs(perm_stats) >= abs(observed_stat))
  
  return(list(
    observed_statistic = observed_stat,
    p_value = p_value,
    permutation_distribution = perm_stats,
    significant = p_value < 0.05
  ))
}

#' Cross-validation for regression
#'
#' @param data Data frame
#' @param formula Model formula
#' @param k Number of folds (default: 10)
#' @return Cross-validation results
cross_validate <- function(data, formula, k = 10) {
  n <- nrow(data)
  fold_size <- ceiling(n / k)
  indices <- sample(1:n)
  
  predictions <- numeric(n)
  mse_folds <- numeric(k)
  
  for (i in 1:k) {
    # Test indices for this fold
    test_start <- (i - 1) * fold_size + 1
    test_end <- min(i * fold_size, n)
    test_indices <- indices[test_start:test_end]
    train_indices <- setdiff(1:n, test_indices)
    
    # Train and test
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    model <- lm(formula, data = train_data)
    pred <- predict(model, newdata = test_data)
    
    # Store predictions
    predictions[test_indices] <- pred
    
    # Calculate MSE for this fold
    actual <- test_data[[all.vars(formula)[1]]]
    mse_folds[i] <- mean((actual - pred)^2)
  }
  
  # Overall metrics
  actual <- data[[all.vars(formula)[1]]]
  mse <- mean((actual - predictions)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predictions))
  r_squared <- 1 - sum((actual - predictions)^2) / sum((actual - mean(actual))^2)
  
  return(list(
    mse = mse,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    mse_by_fold = mse_folds,
    predictions = predictions
  ))
}

#' Bootstrap regression coefficients
#'
#' @param data Data frame
#' @param formula Model formula
#' @param R Number of bootstrap replicates (default: 1000)
#' @return Bootstrap results for coefficients
bootstrap_regression <- function(data, formula, R = 1000) {
  # Original model
  original_model <- lm(formula, data = data)
  original_coefs <- coef(original_model)
  
  # Bootstrap function
  boot_coefs <- function(data, indices) {
    boot_data <- data[indices, ]
    boot_model <- lm(formula, data = boot_data)
    return(coef(boot_model))
  }
  
  # Perform bootstrap
  boot_result <- boot(data = data, statistic = boot_coefs, R = R)
  
  # Calculate confidence intervals for each coefficient
  ci_list <- list()
  for (i in 1:length(original_coefs)) {
    ci <- tryCatch({
      boot.ci(boot_result, conf = 0.95, type = "perc", index = i)$percent[4:5]
    }, error = function(e) {
      c(NA, NA)
    })
    ci_list[[names(original_coefs)[i]]] <- ci
  }
  
  return(list(
    original_coefficients = original_coefs,
    bootstrap_means = colMeans(boot_result$t),
    bootstrap_se = apply(boot_result$t, 2, sd),
    confidence_intervals = ci_list,
    boot_object = boot_result
  ))
}
