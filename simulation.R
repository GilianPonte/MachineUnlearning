# Clear all objects in the R environment
rm(list = ls())
set.seed(1)

# Load required libraries
library(tidyverse)

# Define baseline and parameter increments
baseline <- list(
  B = 50,                  # Default level of uncertainty
  variables = 6,           # Smallest dimensionality of X
  complexity = "uniform",  # Simplest complexity
  confounding = FALSE,     # No confounding
  sample_size = 1e3,       # Smallest sample size
  unlearning_decision = "random" # Only type specified
)

# Define parameter increments
increments <- list(
  variables = c(12, 72),       # Additional dimensionality values
  complexity = c("normal", "mixed"),  # More complex distributions
  confounding = c(TRUE),       # Introduce confounding
  sample_size = c(1e4)         # Larger sample size
)

# Start with baseline
param_grid <- list(baseline)

# Incrementally add configurations, changing one parameter at a time
for (param in names(increments)) {
  current_length <- length(param_grid)
  for (value in increments[[param]]) {
    for (i in seq_len(current_length)) {
      new_setting <- param_grid[[i]]
      new_setting[[param]] <- value
      param_grid <- append(param_grid, list(new_setting))
    }
  }
}

# Remove duplicates to ensure a clean parameter grid
param_grid <- unique(param_grid)

# Convert to data frame for clarity (optional)
param_grid_df <- do.call(rbind, lapply(param_grid, as.data.frame))

# Print the parameter grid to check configurations
print(param_grid_df)

# Initialize storage for results
all_results <- list()

# Simulation placeholder: Utility functions (replace with your actual implementation)
gen_test <- function(n, variables, complexity, confounding) {
  # Generate test data
  list(
    x = matrix(runif(n * variables), ncol = variables),
    w = sample(0:1, n, replace = TRUE),
    y = rnorm(n),
    tau = rnorm(n) # True treatment effect
  )
}
which_list <- function(index, shard_list) {
  # Find which shard contains the index
  for (i in seq_along(shard_list)) {
    if (index %in% shard_list[[i]]) return(i)
  }
  return(NULL)
}

# Run simulations for each configuration in the parameter grid
for (i in seq_along(param_grid)) {
  params <- param_grid[[i]]
  cat(paste0("Running with parameters: ", paste(unlist(params), collapse = ", "), "\n"))
  
  # Extract parameters
  B <- params$B
  variables <- params$variables
  complexity <- params$complexity
  confounding <- params$confounding
  n <- params$sample_size
  unlearning_decision <- params$unlearning_decision
  
  # Initialize storage for this configuration
  perf_results <- list()
  profit_results <- list()
  
  # Generate validation data
  val <- gen_test(n, variables = variables, complexity = complexity, confounding = FALSE)
  
  # Initialize matrices for predictions and computation time
  val.pred.all <- matrix(0, nrow = length(val$y), ncol = B)
  time.spent <- matrix(0, nrow = B, ncol = 2)
  
  # Shard settings
  shard_counts <- c(2, 5, 10)
  phis <- c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  
  for (shards in shard_counts) {
    cat(paste0("Testing with ", shards, " shards...\n"))
    
    for (b in 1:B) {
      # Generate training data
      train <- gen_test(n, variables = variables, complexity = complexity, confounding = confounding)
      
      # Unlearn individual's index
      customers_to_unlearn <- 1
      unlearning_index <- if (unlearning_decision == "random") {
        sample(1:n, customers_to_unlearn)
      } else {
        stop("Only 'random' unlearning_decision supported in this demo")
      }
      
      # Split data into shards
      shard.id <- split(1:length(train$y), cut(sample(1:length(train$y)), breaks = shards, labels = FALSE))
      which_shard_needs_retraining <- which_list(unlearning_index, shard.id)
      
      # Train causal forest on the entire dataset (placeholder)
      mod.all <- causal_forest(X = train$x, W = train$w, Y = train$y, num.trees = 2000)
      val.pred.all[, b] <- predict(mod.all, val$x)[[1]]
      
      # Performance metrics
      perf <- data.frame(
        shard = shards,
        bootstrap = b,
        RMSE = mean((val.pred.all[, b] - val$tau)^2),
        AUTOC = cor(val.pred.all[, b], val$tau)
      )
      perf_results <- rbind(perf_results, perf)
      
      # Profit metrics (placeholder for computation)
      profit <- data.frame(
        shard = shards,
        Phi = phis,
        Profit = sapply(phis, function(phi) {
          top_n <- floor(length(val$tau) * phi)
          sum(val$tau[order(val.pred.all[, b], decreasing = TRUE)[1:top_n]])
        })
      )
      profit_results <- rbind(profit_results, profit)
    }
  }
  
  # Store results
  all_results[[i]] <- list(
    parameters = params,
    performance = perf_results,
    profit = profit_results
  )
}

# Combine all results into a data frame
final_perf_results <- do.call(rbind, lapply(all_results, function(x) {
  data.frame(x$parameters, x$performance)
}))
final_profit_results <- do.call(rbind, lapply(all_results, function(x) {
  data.frame(x$parameters, x$profit)
}))

# Save or visualize results
# saveRDS(final_perf_results, "final_perf_results.RDS")
# saveRDS(final_profit_results, "final_profit_results.RDS")
