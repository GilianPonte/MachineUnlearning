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

# Number of bootstrap samples
B <- 10

# Training and validation sample sizes
train.n.sample <- 1000
val.n.sample <- 1000

#shards
shards = 5

# unlearn individual's index
customers_to_unlearn = 1
unlearning_index = sample(1:train.n.sample, customers_to_unlearn)

# Assuming 'my_list' is your list structure
which_list <- function(value, list_data) {
  return(names(list_data)[sapply(list_data, function(x) value %in% x)])
}


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
  pred.shard.matrix <- matrix(0, nrow = length(train.Y), ncol = shards)
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
rmse.summary <- matrix(0, nrow = B, ncol = 2)
autoc.summary <- matrix(0, nrow = B, ncol = 2)
phis <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
profit.summary <- array(0, dim = c(B, 2, length(phis)))  # 3D array: [B x 2 x phis]

for(b in 1:B) {
  # Compute RMSE
  rmse.summary[b, 1] <- mean((val.pred.all[, b] - val.true.tau)^2)  # Full model
  rmse.summary[b, 2] <- mean((val.pred.shard[, b] - val.true.tau)^2)  # Shard models
  
  # Compute AUTOC
  autoc.summary[b, 1] <- rank_average_treatment_effect.fit(val.true.tau, val.pred.all[, b], R = 3)$estimate
  autoc.summary[b, 2] <- rank_average_treatment_effect.fit(val.true.tau, val.pred.shard[, b], R = 3)$estimate
  
  # Compute profits for each phi
  for (phi_idx in seq_along(phis)) {
    phi <- phis[phi_idx]
    top_n_all <- floor(length(val.pred.all[, b]) * phi)  # Number of top observations for full model
    top_n_shard <- floor(length(val.pred.shard[, b]) * phi)  # Number for shard models
    
    # Store profit in corresponding location
    profit.summary[b, 1, phi_idx] <- sum(val.true.tau[order(val.pred.all[, b], decreasing = TRUE)[1:top_n_all]])
    profit.summary[b, 2, phi_idx] <- sum(val.true.tau[order(val.pred.shard[, b], decreasing = TRUE)[1:top_n_shard]])
  }
}

# Summarize results
data.frame(
  RMSE = paste0(round(colMeans(rmse.summary), 2), " (", round(colSds(rmse.summary), 2), ")"),
  AUTOC = paste0(round(colMeans(autoc.summary), 2), " (", round(colSds(autoc.summary), 2), ")")
)

# Compute average and standard deviation of computation time
colMeans(time.spent)
colSds(time.spent)

# Summarize profit for each phi
profit.mean <- apply(profit.summary, c(2, 3), mean)  # Mean profit [2 x phis]
profit.sd <- apply(profit.summary, c(2, 3), sd)      # Std dev profit [2 x phis]

# Create a summary data frame
profit_summary <- data.frame(
  Phi = rep(phis, each = 2),  # Repeat phi values for full and shard models
  Model = rep(c("Full", "Shard"), times = length(phis)),  # Model type
  Mean = as.vector(profit.mean),  # Mean profits
  SD = as.vector(profit.sd)       # Standard deviations
)

# Convert the 3D array into a tidy data frame for detailed analysis
profit_df <- data.frame()

for (phi_idx in seq_along(phis)) {
  for (b in 1:B) {
    profit_df <- rbind(profit_df, data.frame(
      Bootstrap = b,
      Phi = phis[phi_idx],
      Full_Profit = profit.summary[b, 1, phi_idx],
      Shard_Profit = profit.summary[b, 2, phi_idx]
    ))
  }
}

profit_df %>%
  group_by(Phi) %>%
  summarize(mean_full_profit = mean(Full_Profit),
            mean_shard_profit = mean(Shard_Profit),
            difference = mean_full_profit - mean_shard_profit)


