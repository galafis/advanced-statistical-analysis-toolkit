# Install Required R Packages
# Advanced Statistical Analysis Toolkit
# Author: Gabriel Demetrios Lafis

cat("Installing required R packages...\n")
cat("This may take several minutes.\n\n")

# List of required packages
packages <- c(
  "MASS",          # Statistical functions
  "car",           # Regression diagnostics
  "psych",         # Psychological statistics
  "FactoMineR",    # Multivariate analysis
  "boot",          # Bootstrap methods
  "lmtest",        # Linear model tests
  "nortest",       # Normality tests
  "corrplot",      # Correlation visualization
  "testthat",      # Unit testing
  "moments",       # Skewness and kurtosis
  "ggplot2",       # Advanced plotting
  "reshape2",      # Data reshaping
  "plyr",          # Data manipulation
  "dplyr",         # Data manipulation
  "tidyr"          # Data tidying
)

# Function to install packages if not already installed
install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", package))
    install.packages(package, repos = "https://cloud.r-project.org/", dependencies = TRUE)
    if (require(package, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("✓ %s installed successfully\n", package))
    } else {
      cat(sprintf("✗ Failed to install %s\n", package))
    }
  } else {
    cat(sprintf("✓ %s already installed\n", package))
  }
}

# Install all packages
for (pkg in packages) {
  install_if_missing(pkg)
}

cat("\n")
cat("="*70, "\n")
cat("Installation complete!\n")
cat("="*70, "\n")

# Verify installations
cat("\nVerifying installations...\n")
missing <- character()
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    missing <- c(missing, pkg)
  }
}

if (length(missing) > 0) {
  cat("\n⚠ WARNING: The following packages failed to install:\n")
  cat(paste("  -", missing, collapse = "\n"), "\n")
  cat("\nPlease install them manually using:\n")
  cat(sprintf("install.packages(c(%s))\n", paste(sprintf('"%s"', missing), collapse = ", ")))
} else {
  cat("\n✓ All packages installed and verified successfully!\n")
}

# Print session info
cat("\nR Session Information:\n")
print(sessionInfo())
