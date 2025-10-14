# Utility Functions for Statistical Analysis
# Author: Gabriel Demetrios Lafis

#' Standardize numeric data
#'
#' @param x Numeric vector or matrix
#' @return Standardized data (mean=0, sd=1)
standardize <- function(x) {
  if (is.matrix(x) || is.data.frame(x)) {
    return(scale(x))
  } else {
    return((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
  }
}

#' Calculate effect size (Cohen's d)
#'
#' @param group1 First group
#' @param group2 Second group
#' @return Cohen's d effect size
cohens_d <- function(group1, group2) {
  mean_diff <- mean(group1, na.rm = TRUE) - mean(group2, na.rm = TRUE)
  pooled_sd <- sqrt((var(group1, na.rm = TRUE) + var(group2, na.rm = TRUE)) / 2)
  return(mean_diff / pooled_sd)
}

#' Calculate confidence interval for mean
#'
#' @param x Numeric vector
#' @param conf_level Confidence level (default: 0.95)
#' @return List with lower and upper bounds
confidence_interval <- function(x, conf_level = 0.95) {
  n <- length(x)
  mean_x <- mean(x, na.rm = TRUE)
  se <- sd(x, na.rm = TRUE) / sqrt(n)
  alpha <- 1 - conf_level
  t_critical <- qt(1 - alpha/2, df = n - 1)
  
  lower <- mean_x - t_critical * se
  upper <- mean_x + t_critical * se
  
  return(list(
    mean = mean_x,
    lower = lower,
    upper = upper,
    conf_level = conf_level
  ))
}

#' Remove outliers using IQR method
#'
#' @param x Numeric vector
#' @param k Multiplier for IQR (default: 1.5)
#' @return Vector with outliers removed
remove_outliers <- function(x, k = 1.5) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  
  lower_bound <- q1 - k * iqr
  upper_bound <- q3 + k * iqr
  
  return(x[x >= lower_bound & x <= upper_bound])
}

#' Detect outliers using IQR method
#'
#' @param x Numeric vector
#' @param k Multiplier for IQR (default: 1.5)
#' @return Logical vector indicating outliers
detect_outliers <- function(x, k = 1.5) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  
  lower_bound <- q1 - k * iqr
  upper_bound <- q3 + k * iqr
  
  return(x < lower_bound | x > upper_bound)
}

#' Calculate skewness
#'
#' @param x Numeric vector
#' @return Skewness value
skewness <- function(x) {
  n <- length(x)
  x_centered <- x - mean(x, na.rm = TRUE)
  m3 <- sum(x_centered^3, na.rm = TRUE) / n
  s3 <- (sd(x, na.rm = TRUE))^3
  return(m3 / s3)
}

#' Calculate kurtosis
#'
#' @param x Numeric vector
#' @return Kurtosis value (excess kurtosis)
kurtosis <- function(x) {
  n <- length(x)
  x_centered <- x - mean(x, na.rm = TRUE)
  m4 <- sum(x_centered^4, na.rm = TRUE) / n
  s4 <- (sd(x, na.rm = TRUE))^4
  return(m4 / s4 - 3)  # Excess kurtosis
}

#' Perform data quality checks
#'
#' @param data Data frame
#' @return List with quality metrics
data_quality_check <- function(data) {
  n_rows <- nrow(data)
  n_cols <- ncol(data)
  
  # Missing values
  missing_counts <- colSums(is.na(data))
  missing_pct <- (missing_counts / n_rows) * 100
  
  # Data types
  data_types <- sapply(data, class)
  
  # Unique values
  unique_counts <- sapply(data, function(x) length(unique(x)))
  
  # Numeric columns statistics
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  numeric_summary <- NULL
  if (length(numeric_cols) > 0) {
    numeric_summary <- summary(data[, numeric_cols])
  }
  
  return(list(
    dimensions = c(rows = n_rows, cols = n_cols),
    missing_values = data.frame(
      column = names(missing_counts),
      count = missing_counts,
      percentage = missing_pct
    ),
    data_types = data.frame(
      column = names(data_types),
      type = as.character(data_types)
    ),
    unique_counts = data.frame(
      column = names(unique_counts),
      unique = unique_counts
    ),
    numeric_summary = numeric_summary
  ))
}

#' Winsorize data (replace extreme values)
#'
#' @param x Numeric vector
#' @param lower Lower percentile (default: 0.05)
#' @param upper Upper percentile (default: 0.95)
#' @return Winsorized vector
winsorize <- function(x, lower = 0.05, upper = 0.95) {
  lower_val <- quantile(x, lower, na.rm = TRUE)
  upper_val <- quantile(x, upper, na.rm = TRUE)
  
  x[x < lower_val] <- lower_val
  x[x > upper_val] <- upper_val
  
  return(x)
}

#' Calculate power for t-test
#'
#' @param n Sample size per group
#' @param d Effect size (Cohen's d)
#' @param sig_level Significance level (default: 0.05)
#' @return Statistical power
power_t_test <- function(n, d, sig_level = 0.05) {
  result <- power.t.test(
    n = n,
    delta = d,
    sd = 1,
    sig.level = sig_level,
    type = "two.sample"
  )
  return(result$power)
}

#' Calculate required sample size for t-test
#'
#' @param power Desired power (default: 0.80)
#' @param d Effect size (Cohen's d)
#' @param sig_level Significance level (default: 0.05)
#' @return Required sample size per group
sample_size_t_test <- function(power = 0.80, d, sig_level = 0.05) {
  result <- power.t.test(
    power = power,
    delta = d,
    sd = 1,
    sig.level = sig_level,
    type = "two.sample"
  )
  return(ceiling(result$n))
}
