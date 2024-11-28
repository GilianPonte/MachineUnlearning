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
gen_test <- function(n.sample, complexity = "uniform") {
  
  if (complexity == "uniform"){
    # Simulate covariates as independent uniform random variables
    X1 <- runif(n.sample, 0, 5)
    X2 <- runif(n.sample, 0, 5)
    X3 <- runif(n.sample, 0, 5)
    X4 <- runif(n.sample, 0, 5)
    X5 <- runif(n.sample, 0, 5)
    X6 <- runif(n.sample, 0, 5)
  } else if(complexity == "normal"){
    X1 <- rnorm(n.sample, 0, 1)
    X2 <- rnorm(n.sample, 0, 1)
    X3 <- rnorm(n.sample, 0, 1)
    X4 <- rnorm(n.sample, 0, 1)
    X5 <- rnorm(n.sample, 0, 1)
    X6 <- rnorm(n.sample, 0, 1)
  }
  
  
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
val.n.sample <- n

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

# Define shard counts to test
shard_counts <- c(2, 5, 10)

# phis for profit
phis <- c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Initialize storage for results across shard counts
results <- list()

for (shards in shard_counts) {
  cat(paste0("Testing with ", shards, " shards...\n"))
  
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
  
  # Compute RMSE, AUTOC, and profit
  rmse.summary <- data.frame(
    Model = c("Full", "Shard"),
    RMSE = c(
      mean((val.pred.all - val.true.tau)^2),
      mean((val.pred.shard - val.true.tau)^2)
    )
  )
  
  autoc.summary <- data.frame(
    Model = c("Full", "Shard"),
    AUTOC = c(
      cor(val.pred.all, val.true.tau),
      cor(val.pred.shard, val.true.tau)
    )
  )
  
  profit.summary <- data.frame(
    Phi = rep(phis, each = 2),
    Model = rep(c("Full", "Shard"), times = length(phis)),
    Profit = unlist(lapply(phis, function(phi) {
      top_n <- floor(length(val.true.tau) * phi)
      c(
        sum(val.true.tau[order(val.pred.all, decreasing = TRUE)[1:top_n]]),
        sum(val.true.tau[order(val.pred.shard, decreasing = TRUE)[1:top_n]])
      )
    }))
  )
  
  time.summary <- data.frame(
    Model = c("Full", "Shard"),
    Mean_Time = colMeans(time.spent),
    SD_Time = colSds(time.spent)
  )
  
  # Store results
  results[[as.character(shards)]] <- list(
    RMSE = rmse.summary,
    AUTOC = autoc.summary,
    Profit = profit.summary,
    Time = time.summary
  )
}
# Assuming `results` contains the nested list structure as shown
clean_results <- function(results) {
  # Initialize empty data frames for combined results
  final_rmse <- data.frame()
  final_autoc <- data.frame()
  final_profit <- data.frame()
  final_time <- data.frame()
  
  # Loop over the shard levels
  for (shards in names(results)) {
    shard_results <- results[[shards]]
    
    # Add shard count to each sub-result and combine into final data frames
    rmse <- shard_results$RMSE %>%
      mutate(Shards = as.numeric(shards))
    final_rmse <- bind_rows(final_rmse, rmse)
    
    autoc <- shard_results$AUTOC %>%
      mutate(Shards = as.numeric(shards))
    final_autoc <- bind_rows(final_autoc, autoc)
    
    profit <- shard_results$Profit %>%
      mutate(Shards = as.numeric(shards))
    final_profit <- bind_rows(final_profit, profit)
    
    time <- shard_results$Time %>%
      mutate(Shards = as.numeric(shards))
    final_time <- bind_rows(final_time, time)
  }
  
  # Return a list of cleaned data frames
  list(
    RMSE = final_rmse,
    AUTOC = final_autoc,
    Profit = final_profit,
    Time = final_time
  )
}

# Apply the function to clean the results
cleaned_results <- clean_results(results)

# Display the cleaned results
profit = cleaned_results$Profit 
profit %>%
  ggplot(aes(x = Phi, y = Profit, color = as.factor(Model), group = as.factor(Model))) + 
  geom_line() + 
  theme_bw() +
  facet_wrap(~Shards)


