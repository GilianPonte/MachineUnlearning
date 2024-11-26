# Clear all objects in the R environment
rm(list=ls())

# Load libraries for data manipulation, visualization, and causal inference
library(tidyverse)    # Data manipulation and visualization
library(hrbrthemes)   # ggplot2 themes
library(viridis)      # Color palettes
library(grf)          # Generalized Random Forest for causal inference
library(rlearner)     # Implements R-learner framework
library(xgboost)      # Gradient Boosting framework
require(caret)        # Tools for cross-validation and data splitting
library(matrixStats)

# Function to generate synthetic data for training and evaluation
gen_test <- function(n.sample) {
  # Simulate covariates as independent uniform random variables
  X1 <- runif(n.sample, 0, 5)
  X2 <- runif(n.sample, 0, 5)
  X3 <- runif(n.sample, 0, 5)
  X4 <- runif(n.sample, 0, 5)
  X5 <- runif(n.sample, 0, 5)
  X6 <- runif(n.sample, 0, 5)
  
  covariate <- data.frame(X1, X2, X3, X4, X5, X6)  # Combine covariates into a data frame
  
  # Compute expected outcomes under control and treatment
  exp.Y.control <- sin(pi * X1 * X2) - 2 * (X1 - X3 - 0.5)^2 + X2 * X4 -
    (2 * X1 - X2 + 0.5 * X3^2 - X4 - log(X1) * (X4 - 1.5)) / 2
  exp.Y.treatment <- sin(pi * X1 * X2) - 2 * (X1 - X3 - 0.5)^2 + X2 * X4 + 
    (2 * X1 - X2 + 0.5 * X3^2 - X4 - log(X1) * (X4 - 1.5)) / 2
  
  true.tau <- exp.Y.treatment - exp.Y.control  # Compute true treatment effect
  
  noise <- rnorm(n.sample, 0, 1)  # Add Gaussian noise to the outcomes
  
  # Generate noisy outcomes under control and treatment
  Y.treatment <- exp.Y.treatment + noise
  Y.control <- exp.Y.control + noise
  
  return(list(Y.treatment, Y.control, covariate, true.tau, exp.Y.treatment, exp.Y.control))
}

# Assuming 'my_list' is your list structure
which_list <- function(value, list_data) {
  return(names(list_data)[sapply(list_data, function(x) value %in% x)])
}

# Number of bootstrap samples
B <- 1

# Training and validation sample sizes
n = 1000
train.n.sample <- n
val.n.sample <- n*0.25

#shards
shards = 2

# unlearn individual's index
customers_to_unlearn = 1 # specifies how many customers should be unlearned
unlearning_index = sample(1:train.n.sample, customers_to_unlearn) # samples a customers to be unlearned

# Generate validation data for control and treatment
val.control.data <- gen_test(val.n.sample)
val.control.X <- as.matrix(val.control.data[[3]])  # Covariates
val.control.tau <- val.control.data[[4]]           # True treatment effect
val.control.Y <- val.control.data[[2]]             # Observed outcomes (control)

val.treatment.data <- gen_test(val.n.sample)
val.treatment.X <- as.matrix(val.treatment.data[[3]])  # Covariates
val.treatment.tau <- val.treatment.data[[4]]           # True treatment effect
val.treatment.Y <- val.treatment.data[[1]]             # Observed outcomes (treatment)

# Combine control and treatment data for validation
val.Y <- c(val.control.Y, val.treatment.Y)             # Combined outcomes
val.W <- c(rep(0, val.n.sample), rep(1, val.n.sample)) # Treatment indicator
val.X <- rbind(val.control.X, val.treatment.X)         # Combined covariates
val.true.tau <- c(val.control.tau, val.treatment.tau)  # Combined true treatment effect

# Initialize matrices to store predictions and computation time
val.pred.all <- matrix(0, nrow = length(val.Y), ncol = B)
val.pred.shard <- matrix(0, nrow = length(val.Y), ncol = B)
time.spent <- matrix(0, nrow = B, ncol = 2)

# Bootstrap sampling loop
for(b in 1:B) {
  print(paste0("running bootstrap:", b))
  
  # Generate training data for control and treatment
  train.control.data <- gen_test(train.n.sample)
  train.control.X <- as.matrix(train.control.data[[3]])
  train.control.tau <- train.control.data[[4]]
  train.control.Y <- train.control.data[[2]]
  
  train.treatment.data <- gen_test(train.n.sample)
  train.treatment.X <- as.matrix(train.treatment.data[[3]])
  train.treatment.tau <- train.treatment.data[[4]]
  train.treatment.Y <- train.treatment.data[[1]]
  
  # Combine training data
  train.Y <- c(train.control.Y, train.treatment.Y)
  train.W <- c(rep(0, train.n.sample), rep(1, train.n.sample))
  train.X <- rbind(train.control.X, train.treatment.X)
  train.true.tau <- c(train.control.tau, train.treatment.tau)
  
  # Split data into "shards" shards
  shard.id <- split(1:length(train.Y), cut(sample(1:length(train.Y)), breaks = shards, labels = FALSE))
  which_shard_needs_retraining <- which_list(unlearning_index, shard.id)
  
  # Train causal forest on the entire dataset
  start.time <- Sys.time()
  mod.all <- causal_forest(X = train.X, W = train.W, Y = train.Y, num.trees = 2000)  # Fit causal forest model
  val.pred.all[, b] <- predict(mod.all, val.X)[[1]]  # Predict treatment effects
  end.time <- Sys.time()
  time.spent[b, 1] <- end.time - start.time  # Record computation time
  
  # Train causal forest on each shard and aggregate predictions
  pred.shard.matrix <- matrix(0, nrow = length(val.Y), ncol = shards)
  for(s in 1:shards) {
    print(paste0("working on shard #",s))
    if(which_shard_needs_retraining == s){
      start.time <- Sys.time()
      mod.shard <- causal_forest(X = train.X[shard.id[[s]], ], W = train.W[shard.id[[s]]], Y = train.Y[shard.id[[s]]])
      end.time <- Sys.time()
      time.spent[b, 2] <- end.time - start.time  # Record computation time
    } else{
      mod.shard <- causal_forest(X = train.X[shard.id[[s]], ], W = train.W[shard.id[[s]]], Y = train.Y[shard.id[[s]]]) 
    }
    pred.shard.matrix[, s] <- predict(mod.shard, val.X)[[1]]
  }
  val.pred.shard[, b] <- rowMeans(pred.shard.matrix)  # Average shard predictions
  print(time.spent[b, ])
}


# Initialize storage for RMSE, AUTOC, and Profit
phis <- c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
rmse.summary <- matrix(0, nrow = B, ncol = 2)
autoc.summary <- matrix(0, nrow = B, ncol = 2)
profit.summary <- array(0, dim = c(B, 2, length(phis)))  # [B x 2 x phis]

# Pre-compute indices for phi
phi_indices <- lapply(phis, function(phi) {
  list(
    top_n_all = floor(length(val.true.tau) * phi),
    top_n_shard = floor(length(val.true.tau) * phi)
  )
})

# Main computation loop
for (b in 1:B) {
  # RMSE calculations
  rmse.summary[b, ] <- c(
    mean((val.pred.all[, b] - val.true.tau)^2),  # Full model
    mean((val.pred.shard[, b] - val.true.tau)^2)  # Shard models
  )
  
  # AUTOC calculations
  autoc.summary[b, ] <- c(
    rank_average_treatment_effect.fit(val.true.tau, val.pred.all[, b], R = 3)$estimate,
    rank_average_treatment_effect.fit(val.true.tau, val.pred.shard[, b], R = 3)$estimate
  )
  
  # Profit calculations for each phi
  for (phi_idx in seq_along(phis)) {
    phi <- phis[phi_idx]
    indices <- phi_indices[[phi_idx]]
    
    profit.summary[b, 1, phi_idx] <- sum(
      val.true.tau[order(val.pred.all[, b], decreasing = TRUE)[1:indices$top_n_all]]
    )
    profit.summary[b, 2, phi_idx] <- sum(
      val.true.tau[order(val.pred.shard[, b], decreasing = TRUE)[1:indices$top_n_shard]]
    )
  }
}

# Summarize RMSE and AUTOC results
rmse_summary_df <- data.frame(
  Model = c("Full", "Shard"),
  Mean = colMeans(rmse.summary),
  SD = apply(rmse.summary, 2, sd)
)

autoc_summary_df <- data.frame(
  Model = c("Full", "Shard"),
  Mean = colMeans(autoc.summary),
  SD = apply(autoc.summary, 2, sd)
)

# Summarize Profit results
profit_summary_df <- expand.grid(
  Phi = phis,
  Model = c("Full", "Shard")
) %>%
  mutate(
    Mean = as.vector(apply(profit.summary, c(2, 3), mean)),
    SD = as.vector(apply(profit.summary, c(2, 3), sd))
  )

# Tidy profit details for further analysis
profit_df <- expand.grid(
  Bootstrap = 1:B,
  Phi = phis
) %>%
  rowwise() %>%
  mutate(
    Full_Profit = profit.summary[Bootstrap, 1, which(phis == Phi)],
    Shard_Profit = profit.summary[Bootstrap, 2, which(phis == Phi)]
  ) %>%
  group_by(Phi) %>%
  summarize(
    Mean_Full_Profit = mean(Full_Profit),
    Mean_Shard_Profit = mean(Shard_Profit),
    Profit_Difference = Mean_Full_Profit - Mean_Shard_Profit,
    .groups = 'drop'
  )

# Output results
list(
  RMSE = rmse_summary_df,
  AUTOC = autoc_summary_df,
  Profit_Details = profit_df,
  Time = data.frame(model = c("Full model", "Shard model"), time = colMeans(time.spent), sd_time= colSds(time.spent))
)
