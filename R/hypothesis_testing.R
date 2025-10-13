# Hypothesis Testing Module
# Author: Gabriel Demetrios Lafis

#' Perform comprehensive t-test analysis
#'
#' @param group1 Numeric vector for group 1
#' @param group2 Numeric vector for group 2
#' @param paired Logical indicating if test is paired
#' @param alternative Character: "two.sided", "less", or "greater"
#' @return List with test results
perform_t_test <- function(group1, group2, paired = FALSE, alternative = "two.sided") {
  result <- t.test(group1, group2, paired = paired, alternative = alternative)
  
  list(
    statistic = result$statistic,
    p_value = result$p.value,
    conf_int = result$conf.int,
    mean_diff = result$estimate[1] - result$estimate[2],
    method = result$method
  )
}

#' Perform ANOVA analysis
#'
#' @param data Data frame
#' @param formula Formula for ANOVA
#' @return ANOVA table and post-hoc tests
perform_anova <- function(data, formula) {
  model <- aov(formula, data = data)
  summary_result <- summary(model)
  
  # Post-hoc Tukey HSD if significant
  tukey_result <- NULL
  if (summary_result[[1]]$`Pr(>F)`[1] < 0.05) {
    tukey_result <- TukeyHSD(model)
  }
  
  list(
    anova_table = summary_result,
    tukey_hsd = tukey_result,
    model = model
  )
}

#' Chi-square test of independence
#'
#' @param table Contingency table
#' @return Chi-square test results
perform_chi_square <- function(table) {
  result <- chisq.test(table)
  
  list(
    statistic = result$statistic,
    p_value = result$p.value,
    df = result$parameter,
    expected = result$expected,
    residuals = result$residuals
  )
}

#' Kolmogorov-Smirnov normality test
#'
#' @param data Numeric vector
#' @return KS test results
test_normality <- function(data) {
  shapiro_result <- shapiro.test(data)
  ks_result <- ks.test(data, "pnorm", mean(data), sd(data))
  
  list(
    shapiro = list(
      statistic = shapiro_result$statistic,
      p_value = shapiro_result$p.value
    ),
    ks = list(
      statistic = ks_result$statistic,
      p_value = ks_result$p.value
    ),
    is_normal = shapiro_result$p.value > 0.05
  )
}

#' Levene's test for homogeneity of variance
#'
#' @param data Data frame
#' @param formula Formula specifying groups
#' @return Levene's test results
test_homogeneity_variance <- function(data, formula) {
  library(car)
  result <- leveneTest(formula, data = data)
  
  list(
    f_statistic = result$`F value`[1],
    p_value = result$`Pr(>F)`[1],
    equal_variances = result$`Pr(>F)`[1] > 0.05
  )
}

