# Load libraries, register cores
library(data.table)
library(ranger)
library(tidyverse)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(123, kind = "L'Ecuyer-CMRG")

# Import, prep data
z <- readRDS('simulations/baseline.rds')
phis <- readRDS('simulations/phis.rds')

# Adjacency matrix
imp <- fread('adj_mat.csv')
adj_mat <- ifelse(imp >= 10, 1, 0)
outdegree <- colSums(adj_mat)
keep <- which(outdegree >= 100)
most_causal <- order(outdegree, decreasing = TRUE)[seq_len(10)]
w <- rep(colSums(imp)[most_causal], each = 1000)

# Random forests to predict E[phi|Z,W]
trn_idx <- c(1:5000, 7001:1000)
phi_loop <- function(phi_idx) {
  trn <- sample(trn_idx, 6400)
  tst <- setdiff(trn_idx, trn)
  tf <- keep[phi_idx]
  x <- cbind(z[, c(tf, 334 + which(adj_mat[, tf] == 1))], w)
  f <- ranger(x = x[trn, ], y = phis[trn, phi_idx], 
              mtry = ncol(x)/3, num.trees = 500, num.threads = 8)
  saveRDS(f, paste0('phi_models/phi', phi_idx, '.rds'))
  y_tst <- phis[tst, phi_idx]
  y_hat <- predict(f, x[tst, ])$predictions
  out <- data.table(
    'phi' = paste0('phi', phi_idx),
     'r2' = cor(y_tst, y_hat)^2,
    'mse' = mean((y_tst - y_hat)^2)
  )
  return(out)
}
phi_models_summary <- foreach(g = seq_along(keep), .combine = rbind) %do%
  phi_loop(g)

# Propagate phis
phi_predictor <- function(phi_idx) {
  # Prep data
  tf <- keep[phi_idx]
  x <- cbind(z[, c(tf, 334 + which(adj_mat[, tf] == 1))], w)
  # Import model, export phi
  f <- readRDS(paste0('phi_models/phi', phi_idx, '.rds'))
  phi_j_hat <- predict(f, x)$predictions
  return(phi_j_hat)
}
phi_hat <- foreach(j = seq_len(k), .combine = cbind) %dopar% 
  phi_predictor(j)
colnames(phi_hat) <- paste0('phi', seq_len(k))
saveRDS(phi_hat, 'simulations/phi_hat.rds')







