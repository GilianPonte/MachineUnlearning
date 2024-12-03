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

# davids wd
try(setwd(""))
# gilian wd
try(setwd("C:/Users/Gilia/Dropbox/RSM/projects/Machine Unlearning & CATE estimation/MachineUnlearning"))
source("utils.R")


# GDPR info:
# companies has a month to fulfill the request to be forgotten.

# Number of bootstrap samples
B <- 1 # uncertainty
variables = 6 # dim of X (min 6)
complexity = "uniform" # complexity X
confounding = FALSE # confounding
sample_sizes <- c(1e3, 1e4) # Example sizes


# unlearning settings
shard_counts <- c(2, 5, 10) # shards

phis <- c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) # phis for profit

# Initialize storage for results across all sample sizes
all_perf_results <- list()
all_profit_results <- list()

# Loop over sample sizes
for (n in sample_sizes) {
  cat(paste0("Running with sample size: ", n, "...\n"))
  
  train.n.sample <- n
  val.n.sample <- n
  
  # Generate validation data for control and treatment
  val <- gen_test(val.n.sample,
                  variables = variables,
                  complexity = complexity,
                  confounding = FALSE)
  
  # Initialize matrices to store predictions and computation time
  val.pred.all <- matrix(0, nrow = length(val$y), ncol = B)
  val.pred.shard <- matrix(0, nrow = length(val$y), ncol = B)
  time.spent <- matrix(0, nrow = B, ncol = 2)
  
  # Initialize storage for results across shard counts
  perf_results <- list()
  profit_results <- list()
  
  for (shards in shard_counts) {
    cat(paste0("Testing with ", shards, " shards...\n"))
    
    for (b in 1:B) {
      #print(paste0("Running bootstrap: ", b))
      
      # Generate training data for control and treatment
      train <- gen_test(train.n.sample,
                        variables = variables,
                        complexity = "uniform",
                        confounding = FALSE)
      
      # Unlearn individual's index
      customers_to_unlearn <- 1
      unlearning_index <- sample(1:train.n.sample, customers_to_unlearn)
      
      # Split data into "shards" shards
      shard.id <- split(1:length(train$y), cut(sample(1:length(train$y)), breaks = shards, labels = FALSE))
      which_shard_needs_retraining <- which_list(unlearning_index, shard.id)
      
      # Train causal forest on the entire dataset
      start.time <- Sys.time()
      mod.all <- causal_forest(X = train$x, W = train$w, Y = train$y, num.trees = 2000)
      val.pred.all[, b] <- predict(mod.all, val$x)[[1]]
      end.time <- Sys.time()
      time.spent[b, 1] <- as.numeric(difftime(end.time, start.time, units = "secs"))
      
      # Train causal forest on each shard and aggregate predictions
      pred.shard.matrix <- matrix(0, nrow = length(val$y), ncol = shards)
      for (s in 1:shards) {
        if (which_shard_needs_retraining == s) {
          start.time <- Sys.time()
          mod.shard <- causal_forest(X = train$x[shard.id[[s]], ], W = train$w[shard.id[[s]]], Y = train$y[shard.id[[s]]])
          end.time <- Sys.time()
          time.spent[b, 2] <- as.numeric(difftime(end.time, start.time, units = "secs"))
        } else {
          mod.shard <- causal_forest(X = train$x[shard.id[[s]], ], W = train$w[shard.id[[s]]], Y = train$y[shard.id[[s]]])
        }
        pred.shard.matrix[, s] <- predict(mod.shard, val$x)[[1]]
      }
      val.pred.shard[, b] <- rowMeans(pred.shard.matrix)
      print(time.spent[b, ])
      
      # Compute RMSE, AUTOC, and profit (same as in your original code)
      perf <- data.frame(
        shard = rep(s, 2),
        bootstrap = rep(b, 2),
        Model = c("Full", "Shard"),
        RMSE = c(
          mean((val.pred.all[, b] - val$tau)^2),
          mean((val.pred.shard[, b] - val$tau)^2)
        ),
        AUTOC = c(
          cor(val.pred.all[, b], val$tau),
          cor(val.pred.shard[, b], val$tau)
        ),
        Time = c(
          time.spent[b, 1],
          time.spent[b, 2]
        ),
        SampleSize = rep(n, 2) # Add sample size
      )
      
      profit <- data.frame(
        shard = rep(s, length(phis)),
        bootstrap = rep(b, length(phis)),
        Phi = rep(phis, each = 2),
        Model = rep(c("Full", "Shard"), times = length(phis)),
        Profit = unlist(lapply(phis, function(phi) {
          top_n <- floor(length(val$tau) * phi)
          c(
            sum(val$tau[order(val.pred.all[, b], decreasing = TRUE)[1:top_n]]),
            sum(val$tau[order(val.pred.shard[, b], decreasing = TRUE)[1:top_n]])
          )
        })),
        SampleSize = rep(n, length(phis) * 2) # Add sample size
      )
      
      perf_results <- rbind(perf_results, perf)
      profit_results <- rbind(profit_results, profit)
    }
  }
  
  # Append results to overall storage
  all_perf_results[[as.character(n)]] <- perf_results
  all_profit_results[[as.character(n)]] <- profit_results

}

# Combine results across all sample sizes
final_perf_results <- do.call(rbind, all_perf_results)
final_profit_results <- do.call(rbind, all_profit_results)


# profit plot
final_profit_results %>%
  group_by(shard, Phi, Model, SampleSize) %>%
  summarize(mean_profit = mean(Profit), .groups = "drop") %>%
  pivot_wider(names_from = Model, values_from = mean_profit) %>%
  mutate(profit_difference = Full - Shard) %>%
  ggplot(aes(x = Phi, y = profit_difference, color = as.factor(shard))) + 
  geom_line(size = 1) + # Thicker line for better visibility
  geom_point(size = 1) +  # Add points to emphasize data
  facet_wrap(~SampleSize) +
  labs(
    x = "Phi (log scale)",
    y = "Profit difference (Full - Shard)",
    color = "Shards"
  ) +
  theme_minimal(base_size = 12) + # Clean and modern theme
  theme(
    legend.position = "top", # Move legend to top for clarity
    plot.title = element_text(size = 12), # Bold title
    plot.subtitle = element_text(size = 12), # Add emphasis to subtitle
    axis.title = element_text(size = 12)
  ) +
  scale_x_log10()

# performance plot
final_perf_results %>% 
  group_by(shard, Model) %>%
  pivot_longer(c(RMSE, AUTOC, Time)) %>%
  ggplot(aes(x = interaction(Model, SampleSize), y = value, color = name)) +
  facet_grid(name~shard, scales = "free_y") +
  geom_boxplot() +
  theme_bw() +
  labs(
    x = "Model and # shards",
    y = "",
    color = "Metric"
  )
