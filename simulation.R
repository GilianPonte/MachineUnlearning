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
gen_test <- function(n.sample, 
                     variables = 5, 
                     complexity = c("uniform", "normal", "cauchy", "mixed"),
                     confounding = FALSE) {
  
  if (variables < 5){
    print("Error: variables should be bigger than 5")
    stop()
  }
  # Initialize an empty list to store generated variables
  data <- list()
  
  # Check complexity and generate variables accordingly
  if (complexity == "normal") {
    # Generate variables with normal distribution
    for (i in 1:variables) {
      data[[paste0("X", i)]] <- rnorm(n.sample, 0, 1)
    }
  } else if (complexity == "uniform") {
    # Generate variables with uniform distribution
    for (i in 1:variables) {
      data[[paste0("X", i)]] <- runif(n.sample, 0, 5)
    }
  } else if (complexity == "cauchy") {
    # Generate variables with Cauchy distribution
    for (i in 1:variables) {
      data[[paste0("X", i)]] <- rcauchy(n.sample)
    }
  } else if (complexity == "mixed") {
    # Generate variables with a mix of Normal, Binomial, and Cauchy distributions
    for (i in 1:variables) {
      if (i %% 3 == 1) { # Normal distribution
        data[[paste0("X", i)]] <- rnorm(n.sample, 0, 1)
      } else if (i %% 3 == 2) { # Binomial distribution
        data[[paste0("X", i)]] <- rbinom(n.sample, size = 10, prob = 0.5)
      } else { # Cauchy distribution
        data[[paste0("X", i)]] <- rcauchy(n.sample)
      }
    }
  } else {
    stop("Unknown complexity type. Use 'normal', 'uniform', 'cauchy', or 'mixed'.")
  }
  
  # Combine covariates into a data frame
  x <- as.data.frame(data)

  if (confounding == TRUE){
    p = 1/(1 + exp(x[,2] + x[,3]))
  } else{
    p = 0.5 
  }
  w = as.numeric(rbinom(n.sample,1,p)==1)
  m = pmax(0, x[,1] + x[,2], x[,3]) + pmax(0, x[,4] + x[,5])
  tau = x[,1] + log(1 + exp(x[,2]))^2
  mu1 = m + tau/2
  mu0 = m - tau/2
  y = w*mu1 + (1-w) * mu0 + 0.5*rnorm(n.sample)
  
  return(list(x=x, w=w, y=y, p=p, m=m, mu0=mu0, mu1=mu1, tau=tau))
}

# Assuming 'my_list' is your list structure
which_list <- function(value, list_data) {
  return(names(list_data)[sapply(list_data, function(x) value %in% x)])
}

# Number of bootstrap samples
B <- 2

# Training and validation sample sizes
n = 1000
train.n.sample <- n
val.n.sample <- n

# unlearn individual's index
customers_to_unlearn = 1 # specifies how many customers should be unlearned
unlearning_index = sample(1:train.n.sample, customers_to_unlearn) # samples a customers to be unlearned

# Generate validation data for control and treatment
val <- gen_test(val.n.sample,
                variables = 5,
                complexity = "uniform",
                confounding = FALSE)

# Initialize matrices to store predictions and computation time
val.pred.all <- matrix(0, nrow = length(val$y), ncol = B)
val.pred.shard <- matrix(0, nrow = length(val$y), ncol = B)
time.spent <- matrix(0, nrow = B, ncol = 2)

# Define shard counts to test
shard_counts <- c(2, 5, 10)

# phis for profit
phis <- c(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Initialize storage for results across shard counts
perf_results <- list()
profit_results = list()

for (shards in shard_counts) {
  cat(paste0("Testing with ", shards, " shards...\n"))
  
  # Bootstrap sampling loop
  for(b in 1:B) {
    print(paste0("running bootstrap:", b))
  
    # Generate training data for control and treatment
    train <- gen_test(train.n.sample,
                    variables = 5,
                    complexity = "uniform",
                    confounding = FALSE)
  
    # Split data into "shards" shards
    shard.id <- split(1:length(train$y), cut(sample(1:length(train$y)), breaks = shards, labels = FALSE))
    which_shard_needs_retraining <- which_list(unlearning_index, shard.id)
  
    # Train causal forest on the entire dataset
    start.time <- Sys.time()
    mod.all <- causal_forest(X = train$x, W = train$w, Y = train$y, num.trees = 2000)  # Fit causal forest model
    val.pred.all[, b] <- predict(mod.all, val$x)[[1]]  # Predict treatment effects
    end.time <- Sys.time()
    time.spent[b, 1] <- end.time - start.time  # Record computation time
  
    # Train causal forest on each shard and aggregate predictions
    pred.shard.matrix <- matrix(0, nrow = length(val$y), ncol = shards)
    for(s in 1:shards) {
      print(paste0("working on shard #",s))
      if(which_shard_needs_retraining == s){
        start.time <- Sys.time()
        mod.shard <- causal_forest(X = train$x[shard.id[[s]], ], W = train$w[shard.id[[s]]], Y = train$y[shard.id[[s]]])
        end.time <- Sys.time()
        time.spent[b, 2] <- end.time - start.time  # Record computation time
      } else{
        mod.shard <- causal_forest(X = train$x[shard.id[[s]], ], W = train$w[shard.id[[s]]], Y = train$y[shard.id[[s]]]) 
      }
      pred.shard.matrix[, s] <- predict(mod.shard, val$x)[[1]]
    } # shards loop stops here
    val.pred.shard[, b] <- rowMeans(pred.shard.matrix)  # Average shard predictions
    print(time.spent[b, ])
    
    # Compute RMSE, AUTOC, and profit
    perf <- data.frame(
      shard = rep(s,2),
      bootstrap = rep(b, 2), # Repeat `b` for each model type
      Model = c("Full", "Shard"), # Two models: Full and Shard
      RMSE = c(
        mean((val.pred.all[, b] - val$tau)^2),
        mean((val.pred.shard[, b] - val$tau)^2)
      ),
      AUTOC = c(
        cor(val.pred.all[, b], val$tau),
        cor(val.pred.shard[, b], val$tau)
      ),
      Time = c(
        time.spent[b, 1], # Time for "Full" model
        time.spent[b, 2]  # Time for "Shard" model
      )
    )
    
    # profit
    profit = data.frame(
      shard = rep(s,length(phis)),
      bootstrap = rep(b, length(phis)), # Repeat `b` for each model type
      Phi = rep(phis, each = 2),
      Model = rep(c("Full", "Shard"), times = length(phis)),
      Profit = unlist(lapply(phis, function(phi) {
        top_n <- floor(length(val$tau) * phi)
        c(
          sum(val$tau[order(val.pred.all[,b], decreasing = TRUE)[1:top_n]]),
          sum(val$tau[order(val.pred.shard[,b], decreasing = TRUE)[1:top_n]])
        )
      }))
    )
    
    perf_results = rbind(perf_results, perf)
    
    profit_results = rbind(profit_results, profit)
    
    
  } # bootstrap loop finishes here
}

print(perf_results)
print(profit_results)



# Display the cleaned results
profit = cleaned_results$Profit 
profit %>%
  ggplot(aes(x = Phi, y = Profit, color = as.factor(Model), group = as.factor(Model))) + 
  geom_line() + 
  theme_bw() +
  facet_wrap(~Shards)


