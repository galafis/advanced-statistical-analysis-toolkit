# Regression Diagnostics Example
# Advanced Statistical Analysis Toolkit
# Author: Gabriel Demetrios Lafis

# Load required libraries
library(MASS)
library(car)
library(lmtest)

# Source custom functions
source("../R/regression_analysis.R")
source("../R/utils.R")

cat("="*70, "\n")
cat("Regression Diagnostics Example\n")
cat("="*70, "\n\n")

# Load sample data
cat("Loading housing data...\n")
data <- read.csv("../data/sample_datasets/housing.csv")

cat(sprintf("Dataset dimensions: %d rows × %d columns\n\n", nrow(data), ncol(data)))

# Display first few rows
cat("First few rows:\n")
print(head(data))

# Perform multiple linear regression
cat("\n", "="*70, "\n")
cat("Fitting Multiple Linear Regression Model\n")
cat("="*70, "\n\n")

# Fit the model
formula <- price ~ bedrooms + bathrooms + sqft + age
result <- multiple_linear_regression(data, formula)

# Display model summary
cat("Model Summary:\n")
print(result$summary)

# Display fit statistics
cat("\nModel Fit Statistics:\n")
cat(sprintf("R-squared: %.4f\n", result$fit_statistics$r_squared))
cat(sprintf("Adjusted R-squared: %.4f\n", result$fit_statistics$adj_r_squared))
cat(sprintf("AIC: %.2f\n", result$fit_statistics$aic))
cat(sprintf("BIC: %.2f\n", result$fit_statistics$bic))
cat(sprintf("RMSE: %.2f\n", result$fit_statistics$rmse))

# Diagnostics
cat("\n", "="*70, "\n")
cat("Regression Diagnostics\n")
cat("="*70, "\n\n")

# Normality of residuals
if (!is.null(result$diagnostics$shapiro_test)) {
  cat("Shapiro-Wilk Test for Normality:\n")
  cat(sprintf("  W = %.4f, p-value = %.4f\n", 
              result$diagnostics$shapiro_test$statistic,
              result$diagnostics$shapiro_test$p_value))
  if (result$diagnostics$shapiro_test$p_value > 0.05) {
    cat("  → Residuals are normally distributed\n")
  } else {
    cat("  → Residuals deviate from normality\n")
  }
}

cat("\nBreusch-Pagan Test for Homoscedasticity:\n")
cat(sprintf("  BP = %.4f, p-value = %.4f\n",
            result$diagnostics$breusch_pagan_test$statistic,
            result$diagnostics$breusch_pagan_test$p_value))
if (result$diagnostics$breusch_pagan_test$p_value > 0.05) {
  cat("  → Homoscedasticity assumption met\n")
} else {
  cat("  → Heteroscedasticity detected\n")
}

# VIF for multicollinearity
if (!is.null(result$diagnostics$vif)) {
  cat("\nVariance Inflation Factors (VIF):\n")
  print(result$diagnostics$vif)
  cat("\nInterpretation:\n")
  cat("  VIF < 5: Low multicollinearity\n")
  cat("  VIF 5-10: Moderate multicollinearity\n")
  cat("  VIF > 10: High multicollinearity\n")
}

# Influential observations
if (length(result$diagnostics$influential_obs) > 0) {
  cat(sprintf("\nInfluential observations detected: %d\n", 
              length(result$diagnostics$influential_obs)))
  cat("Row indices:", paste(result$diagnostics$influential_obs, collapse=", "), "\n")
}

if (length(result$diagnostics$high_influence) > 0) {
  cat(sprintf("\nHigh influence points (Cook's D > 4/n): %d\n",
              length(result$diagnostics$high_influence)))
  cat("Row indices:", paste(result$diagnostics$high_influence, collapse=", "), "\n")
}

# Residual statistics
cat("\nResidual Statistics:\n")
cat(sprintf("  Mean: %.4f\n", result$diagnostics$residuals$mean))
cat(sprintf("  SD: %.4f\n", result$diagnostics$residuals$sd))
cat(sprintf("  Skewness: %.4f\n", result$diagnostics$residuals$skewness))
cat(sprintf("  Kurtosis: %.4f\n", result$diagnostics$residuals$kurtosis))

# Create diagnostic plots
cat("\n", "="*70, "\n")
cat("Creating Diagnostic Plots\n")
cat("="*70, "\n\n")

# Set up plotting area
pdf("../data/results/regression_diagnostics.pdf", width=12, height=10)
par(mfrow=c(2,2))

# Standard diagnostic plots
plot(result$model, main="Regression Diagnostics")

# Additional plots
par(mfrow=c(2,2))

# Histogram of residuals
hist(result$residuals, breaks=30, main="Histogram of Residuals",
     xlab="Residuals", col="skyblue", border="white")

# Residuals vs fitted values
plot(result$fitted_values, result$residuals,
     main="Residuals vs Fitted Values",
     xlab="Fitted Values", ylab="Residuals",
     pch=19, col=rgb(0,0,1,0.3))
abline(h=0, col="red", lwd=2, lty=2)

# Q-Q plot
qqnorm(result$residuals, main="Normal Q-Q Plot")
qqline(result$residuals, col="red", lwd=2)

# Scale-location plot
sqrt_abs_resid <- sqrt(abs(result$residuals / sd(result$residuals)))
plot(result$fitted_values, sqrt_abs_resid,
     main="Scale-Location Plot",
     xlab="Fitted Values", ylab="√|Standardized Residuals|",
     pch=19, col=rgb(0,0,1,0.3))

dev.off()

cat("✓ Diagnostic plots saved to ../data/results/regression_diagnostics.pdf\n")

cat("\n", "="*70, "\n")
cat("Example completed successfully!\n")
cat("="*70, "\n")
