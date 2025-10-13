# Multivariate Analysis Module
# Author: Gabriel Demetrios Lafis

library(MASS)
library(car)

#' Perform Principal Component Analysis (PCA)
#'
#' @param data Data frame with numeric variables
#' @param scale Logical, whether to scale variables (default: TRUE)
#' @return List with PCA results and plots
perform_pca <- function(data, scale = TRUE) {
  # Remove non-numeric columns
  numeric_data <- data[, sapply(data, is.numeric)]
  
  # Perform PCA
  pca_result <- prcomp(numeric_data, scale. = scale)
  
  # Calculate variance explained
  variance_explained <- (pca_result$sdev^2) / sum(pca_result$sdev^2)
  cumulative_variance <- cumsum(variance_explained)
  
  # Create results list
  results <- list(
    pca = pca_result,
    variance_explained = variance_explained,
    cumulative_variance = cumulative_variance,
    loadings = pca_result$rotation,
    scores = pca_result$x
  )
  
  return(results)
}

#' Perform Linear Discriminant Analysis (LDA)
#'
#' @param formula Formula for LDA
#' @param data Data frame
#' @return LDA model and predictions
perform_lda <- function(formula, data) {
  # Fit LDA model
  lda_model <- lda(formula, data = data)
  
  # Make predictions
  predictions <- predict(lda_model, data)
  
  # Calculate confusion matrix
  actual <- data[[all.vars(formula)[1]]]
  confusion <- table(Predicted = predictions$class, Actual = actual)
  accuracy <- sum(diag(confusion)) / sum(confusion)
  
  results <- list(
    model = lda_model,
    predictions = predictions,
    confusion_matrix = confusion,
    accuracy = accuracy
  )
  
  return(results)
}

#' Perform MANOVA (Multivariate Analysis of Variance)
#'
#' @param formula Formula for MANOVA
#' @param data Data frame
#' @return MANOVA results
perform_manova <- function(formula, data) {
  # Fit MANOVA model
  manova_model <- manova(formula, data = data)
  
  # Get summary
  manova_summary <- summary(manova_model, test = "Wilks")
  
  # Perform univariate ANOVAs
  univariate_results <- summary.aov(manova_model)
  
  results <- list(
    model = manova_model,
    multivariate_test = manova_summary,
    univariate_tests = univariate_results
  )
  
  return(results)
}

#' Calculate correlation matrix with significance tests
#'
#' @param data Data frame with numeric variables
#' @return Correlation matrix with p-values
correlation_analysis <- function(data) {
  # Remove non-numeric columns
  numeric_data <- data[, sapply(data, is.numeric)]
  
  # Calculate correlation matrix
  cor_matrix <- cor(numeric_data, use = "complete.obs")
  
  # Calculate p-values
  n <- nrow(numeric_data)
  p_values <- matrix(NA, ncol(numeric_data), ncol(numeric_data))
  
  for (i in 1:ncol(numeric_data)) {
    for (j in 1:ncol(numeric_data)) {
      if (i != j) {
        test_result <- cor.test(numeric_data[,i], numeric_data[,j])
        p_values[i, j] <- test_result$p.value
      }
    }
  }
  
  colnames(p_values) <- colnames(numeric_data)
  rownames(p_values) <- colnames(numeric_data)
  
  results <- list(
    correlation_matrix = cor_matrix,
    p_values = p_values,
    significant_pairs = which(p_values < 0.05, arr.ind = TRUE)
  )
  
  return(results)
}

#' Perform Factor Analysis
#'
#' @param data Data frame with numeric variables
#' @param n_factors Number of factors to extract
#' @param rotation Rotation method (default: "varimax")
#' @return Factor analysis results
perform_factor_analysis <- function(data, n_factors, rotation = "varimax") {
  # Remove non-numeric columns
  numeric_data <- data[, sapply(data, is.numeric)]
  
  # Perform factor analysis
  fa_result <- factanal(numeric_data, factors = n_factors, rotation = rotation)
  
  # Extract loadings
  loadings_matrix <- fa_result$loadings
  
  # Calculate communalities
  communalities <- rowSums(loadings_matrix^2)
  
  results <- list(
    model = fa_result,
    loadings = loadings_matrix,
    communalities = communalities,
    uniquenesses = fa_result$uniquenesses,
    variance_explained = colSums(loadings_matrix^2) / nrow(loadings_matrix)
  )
  
  return(results)
}

#' Perform Cluster Analysis (Hierarchical)
#'
#' @param data Data frame with numeric variables
#' @param method Linkage method (default: "ward.D2")
#' @param k Number of clusters
#' @return Cluster analysis results
hierarchical_clustering <- function(data, method = "ward.D2", k = 3) {
  # Remove non-numeric columns
  numeric_data <- data[, sapply(data, is.numeric)]
  
  # Scale data
  scaled_data <- scale(numeric_data)
  
  # Calculate distance matrix
  dist_matrix <- dist(scaled_data)
  
  # Perform hierarchical clustering
  hclust_result <- hclust(dist_matrix, method = method)
  
  # Cut tree to get clusters
  clusters <- cutree(hclust_result, k = k)
  
  # Calculate cluster statistics
  cluster_means <- aggregate(numeric_data, by = list(cluster = clusters), FUN = mean)
  cluster_sizes <- table(clusters)
  
  results <- list(
    model = hclust_result,
    clusters = clusters,
    cluster_means = cluster_means,
    cluster_sizes = cluster_sizes,
    distance_matrix = dist_matrix
  )
  
  return(results)
}

