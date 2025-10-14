# Bootstrap Example
# Advanced Statistical Analysis Toolkit
# Author: Gabriel Demetrios Lafis

library(boot)

# Source custom functions
source("../R/resampling_methods.R")
source("../R/utils.R")

cat("="*70, "\n")
cat("Bootstrap Methods Example\n")
cat("="*70, "\n\n")

# Load sample data
cat("Loading income data...\n")
data <- read.csv("../data/sample_datasets/income.csv")
income <- data$income

cat(sprintf("Sample size: %d\n", length(income)))
cat(sprintf("Original mean: $%.2f\n", mean(income)))
cat(sprintf("Original median: $%.2f\n", median(income)))
cat(sprintf("Original SD: $%.2f\n\n", sd(income)))

# ============================================================================
# Example 1: Bootstrap Confidence Interval for Mean
# ============================================================================
cat("="*70, "\n")
cat("Example 1: Bootstrap CI for Mean\n")
cat("="*70, "\n\n")

set.seed(42)
boot_mean <- perform_bootstrap(income, statistic = mean, R = 10000)

cat("Bootstrap Results:\n")
cat(sprintf("  Original mean: $%.2f\n", boot_mean$original))
cat(sprintf("  Bootstrap mean: $%.2f\n", boot_mean$bootstrap_mean))
cat(sprintf("  Bootstrap SE: $%.2f\n", boot_mean$bootstrap_se))

if (!is.null(boot_mean$ci_percentile)) {
  cat(sprintf("  95%% CI (Percentile): [$%.2f, $%.2f]\n",
              boot_mean$ci_percentile[1], boot_mean$ci_percentile[2]))
}

if (!is.null(boot_mean$ci_bca)) {
  cat(sprintf("  95%% CI (BCa): [$%.2f, $%.2f]\n",
              boot_mean$ci_bca[1], boot_mean$ci_bca[2]))
}

# ============================================================================
# Example 2: Bootstrap Confidence Interval for Median
# ============================================================================
cat("\n", "="*70, "\n")
cat("Example 2: Bootstrap CI for Median\n")
cat("="*70, "\n\n")

boot_median <- perform_bootstrap(income, statistic = median, R = 10000)

cat("Bootstrap Results:\n")
cat(sprintf("  Original median: $%.2f\n", boot_median$original))
cat(sprintf("  Bootstrap median: $%.2f\n", boot_median$bootstrap_mean))
cat(sprintf("  Bootstrap SE: $%.2f\n", boot_median$bootstrap_se))

if (!is.null(boot_median$ci_percentile)) {
  cat(sprintf("  95%% CI (Percentile): [$%.2f, $%.2f]\n",
              boot_median$ci_percentile[1], boot_median$ci_percentile[2]))
}

# ============================================================================
# Example 3: Bootstrap Comparison of Two Groups
# ============================================================================
cat("\n", "="*70, "\n")
cat("Example 3: Bootstrap Comparison of Groups\n")
cat("="*70, "\n\n")

# Split into two groups by education level
bachelor <- income[data$education == "Bachelor"]
master <- income[data$education == "Master"]

cat(sprintf("Bachelor group: n=%d, mean=$%.2f\n", length(bachelor), mean(bachelor)))
cat(sprintf("Master group: n=%d, mean=$%.2f\n\n", length(master), mean(master)))

boot_comp <- bootstrap_comparison(bachelor, master, statistic = mean, R = 10000)

cat("Bootstrap Comparison Results:\n")
cat(sprintf("  Observed difference: $%.2f\n", boot_comp$observed_diff))
cat(sprintf("  Bootstrap mean difference: $%.2f\n", boot_comp$bootstrap_mean_diff))
cat(sprintf("  Bootstrap SE: $%.2f\n", boot_comp$bootstrap_se))
cat(sprintf("  95%% CI: [$%.2f, $%.2f]\n", boot_comp$ci[1], boot_comp$ci[2]))
cat(sprintf("  P-value: %.4f\n", boot_comp$p_value))

if (boot_comp$p_value < 0.05) {
  cat("  → Significant difference detected (p < 0.05)\n")
} else {
  cat("  → No significant difference (p ≥ 0.05)\n")
}

# ============================================================================
# Example 4: Jackknife Estimation
# ============================================================================
cat("\n", "="*70, "\n")
cat("Example 4: Jackknife Estimation\n")
cat("="*70, "\n\n")

# Use smaller sample for jackknife
income_sample <- sample(income, 50)

jack_result <- jackknife(income_sample, statistic = mean)

cat("Jackknife Results:\n")
cat(sprintf("  Original mean: $%.2f\n", jack_result$original))
cat(sprintf("  Jackknife mean: $%.2f\n", jack_result$jackknife_mean))
cat(sprintf("  Jackknife SE: $%.2f\n", jack_result$jackknife_se))
cat(sprintf("  Bias: $%.2f\n", jack_result$bias))

# ============================================================================
# Visualization
# ============================================================================
cat("\n", "="*70, "\n")
cat("Creating Bootstrap Distribution Plots\n")
cat("="*70, "\n\n")

pdf("../data/results/bootstrap_distributions.pdf", width=12, height=10)
par(mfrow=c(2,2))

# Bootstrap distribution of mean
hist(boot_mean$bootstrap_values, breaks=50, 
     main="Bootstrap Distribution of Mean",
     xlab="Mean Income ($)", col="skyblue", border="white")
abline(v=boot_mean$original, col="red", lwd=2, lty=2)
if (!is.null(boot_mean$ci_percentile)) {
  abline(v=boot_mean$ci_percentile, col="darkgreen", lwd=2, lty=2)
}
legend("topright", legend=c("Original", "95% CI"), 
       col=c("red", "darkgreen"), lty=2, lwd=2)

# Bootstrap distribution of median
hist(boot_median$bootstrap_values, breaks=50,
     main="Bootstrap Distribution of Median",
     xlab="Median Income ($)", col="lightcoral", border="white")
abline(v=boot_median$original, col="red", lwd=2, lty=2)
if (!is.null(boot_median$ci_percentile)) {
  abline(v=boot_median$ci_percentile, col="darkgreen", lwd=2, lty=2)
}

# Bootstrap distribution of difference
hist(boot_comp$differences, breaks=50,
     main="Bootstrap Distribution of Mean Difference",
     xlab="Difference in Mean Income ($)", col="lightyellow", border="white")
abline(v=boot_comp$observed_diff, col="red", lwd=2, lty=2)
abline(v=boot_comp$ci, col="darkgreen", lwd=2, lty=2)
abline(v=0, col="blue", lwd=2, lty=1)

# Jackknife estimates
plot(jack_result$estimates, type="l",
     main="Jackknife Estimates",
     xlab="Leave-one-out iteration", ylab="Mean Income ($)",
     col="blue", lwd=2)
abline(h=jack_result$original, col="red", lwd=2, lty=2)
abline(h=jack_result$jackknife_mean, col="darkgreen", lwd=2, lty=2)
legend("topright", legend=c("Original", "Jackknife Mean"),
       col=c("red", "darkgreen"), lty=2, lwd=2)

dev.off()

cat("✓ Plots saved to ../data/results/bootstrap_distributions.pdf\n")

cat("\n", "="*70, "\n")
cat("Example completed successfully!\n")
cat("="*70, "\n")
