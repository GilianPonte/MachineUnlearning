# Function to generate synthetic data for training and evaluation
gen_test <- function(n.sample, 
                     variables = 5, 
                     complexity = c("uniform", "normal", "mixed"),
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
