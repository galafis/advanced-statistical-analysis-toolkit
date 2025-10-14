# Regression Analysis Module
# Author: Gabriel Demetrios Lafis

library(MASS)
library(car)
library(lmtest)

#' Perform multiple linear regression with comprehensive diagnostics
#'
#' @param data Data frame containing variables
#' @param formula Regression formula (e.g., y ~ x1 + x2 + x3)
#' @return List containing model, diagnostics, and plots
multiple_linear_regression <- function(data, formula) {
  # Fit model
  model <- lm(formula, data = data)
  
  # Model summary
  model_summary <- summary(model)
  
  # Diagnostics
  diagnostics <- list()
  
  # Normality of residuals (Shapiro-Wilk test)
  if (nrow(data) <= 5000) {
    shapiro_test <- shapiro.test(residuals(model))
    diagnostics$shapiro_test <- list(
      statistic = shapiro_test$statistic,
      p_value = shapiro_test$p.value
    )
  }
  
  # Homoscedasticity (Breusch-Pagan test)
  bp_test <- bptest(model)
  diagnostics$breusch_pagan_test <- list(
    statistic = bp_test$statistic,
    p_value = bp_test$p.value
  )
  
  # Multicollinearity (VIF)
  if (length(coef(model)) > 2) {
    tryCatch({
      vif_values <- vif(model)
      diagnostics$vif <- vif_values
    }, error = function(e) {
      diagnostics$vif <- NULL
    })
  }
  
  # Influence measures
  influence_stats <- influence.measures(model)
  diagnostics$influential_obs <- which(apply(influence_stats$is.inf, 1, any))
  
  # Cook's distance
  cooks_d <- cooks.distance(model)
  diagnostics$high_influence <- which(cooks_d > 4/nrow(data))
  
  # Residual statistics
  diagnostics$residuals <- list(
    mean = mean(residuals(model)),
    sd = sd(residuals(model)),
    skewness = moments::skewness(residuals(model)),
    kurtosis = moments::kurtosis(residuals(model))
  )
  
  # Model fit statistics
  fit_stats <- list(
    r_squared = model_summary$r.squared,
    adj_r_squared = model_summary$adj.r.squared,
    aic = AIC(model),
    bic = BIC(model),
    rmse = sqrt(mean(residuals(model)^2))
  )
  
  return(list(
    model = model,
    summary = model_summary,
    diagnostics = diagnostics,
    fit_statistics = fit_stats,
    residuals = residuals(model),
    fitted_values = fitted(model)
  ))
}

#' Perform logistic regression
#'
#' @param data Data frame containing variables
#' @param formula Regression formula for binary outcome
#' @return List containing model and diagnostics
logistic_regression <- function(data, formula) {
  # Fit model
  model <- glm(formula, data = data, family = binomial(link = "logit"))
  
  # Model summary
  model_summary <- summary(model)
  
  # Odds ratios
  odds_ratios <- exp(coef(model))
  
  # Confidence intervals for odds ratios
  ci <- exp(confint(model))
  
  # Predictions
  predictions <- predict(model, type = "response")
  predicted_class <- ifelse(predictions > 0.5, 1, 0)
  
  # Model fit statistics
  fit_stats <- list(
    aic = AIC(model),
    bic = BIC(model),
    null_deviance = model$null.deviance,
    residual_deviance = model$deviance,
    pseudo_r_squared = 1 - (model$deviance / model$null.deviance)
  )
  
  return(list(
    model = model,
    summary = model_summary,
    odds_ratios = odds_ratios,
    confidence_intervals = ci,
    predictions = predictions,
    predicted_class = predicted_class,
    fit_statistics = fit_stats
  ))
}

#' Perform polynomial regression
#'
#' @param data Data frame containing variables
#' @param x_var Name of predictor variable
#' @param y_var Name of response variable
#' @param degree Degree of polynomial (default: 2)
#' @return List containing model and predictions
polynomial_regression <- function(data, x_var, y_var, degree = 2) {
  # Create formula with polynomial terms
  formula_str <- paste0(y_var, " ~ poly(", x_var, ", ", degree, ", raw=TRUE)")
  formula <- as.formula(formula_str)
  
  # Fit model
  model <- lm(formula, data = data)
  
  # Predictions for plotting
  x_range <- seq(min(data[[x_var]]), max(data[[x_var]]), length.out = 100)
  pred_data <- data.frame(x = x_range)
  names(pred_data) <- x_var
  predictions <- predict(model, newdata = pred_data, interval = "confidence")
  
  return(list(
    model = model,
    summary = summary(model),
    x_range = x_range,
    predictions = predictions,
    r_squared = summary(model)$r.squared
  ))
}

#' Perform ridge regression
#'
#' @param data Data frame containing variables
#' @param formula Regression formula
#' @param lambda Regularization parameter
#' @return Ridge regression model
ridge_regression <- function(data, formula, lambda = NULL) {
  # Extract response and predictors
  response <- all.vars(formula)[1]
  predictors <- all.vars(formula)[-1]
  
  # Prepare matrices
  X <- as.matrix(data[, predictors])
  y <- data[[response]]
  
  # Fit ridge regression
  if (is.null(lambda)) {
    # Cross-validation to find optimal lambda
    ridge_model <- MASS::lm.ridge(formula, data = data, lambda = seq(0, 10, 0.1))
    optimal_lambda <- ridge_model$lambda[which.min(ridge_model$GCV)]
    ridge_model <- MASS::lm.ridge(formula, data = data, lambda = optimal_lambda)
  } else {
    ridge_model <- MASS::lm.ridge(formula, data = data, lambda = lambda)
  }
  
  return(list(
    model = ridge_model,
    coefficients = coef(ridge_model),
    lambda = ridge_model$lambda,
    gcv = ridge_model$GCV
  ))
}

#' Perform stepwise regression
#'
#' @param data Data frame containing variables
#' @param formula Full model formula
#' @param direction Direction of stepwise selection ("both", "forward", "backward")
#' @return Selected model
stepwise_regression <- function(data, formula, direction = "both") {
  # Fit full model
  full_model <- lm(formula, data = data)
  
  # Perform stepwise selection
  if (direction == "both") {
    selected_model <- step(full_model, direction = "both", trace = 0)
  } else if (direction == "forward") {
    null_model <- lm(update(formula, . ~ 1), data = data)
    selected_model <- step(null_model, scope = formula, direction = "forward", trace = 0)
  } else if (direction == "backward") {
    selected_model <- step(full_model, direction = "backward", trace = 0)
  }
  
  return(list(
    model = selected_model,
    summary = summary(selected_model),
    formula = formula(selected_model),
    aic = AIC(selected_model),
    bic = BIC(selected_model)
  ))
}
