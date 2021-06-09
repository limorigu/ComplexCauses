# Load libraries, register cores
library(data.table)
library(randomForest)
library(kernlab)
library(Rfast)
library(tidyverse)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(123, kind = "L'Ecuyer-CMRG")

# Import data, downloaded from:
# http://dreamchallenges.org/project/dream-5-network-inference-challenge/
mat <- as.matrix(fread('net3_expression_data.tsv'))
imp <- fread('adj_mat.csv')

# Scale
mat <- scale(mat)

# First 334 genes are transcription factors, rest are not
x <- mat[, seq_len(334)]
y <- mat[, 335:ncol(mat)]

# Compute outdegree with (arbitrary) threshold of 10
adj_mat <- ifelse(imp >= 10, 1, 0)
outdegree <- colSums(adj_mat)

# Simulate x function
Sigma <- cov(x)
sim_x_fn <- function(n) {
  mu <- matrix(rnorm(n * 334), nrow = n)
  sim_x <- mu %*% chol(Sigma)
  colnames(sim_x) <- colnames(x)
  return(sim_x)
}

# Simulate y function
sim_y_fn <- function(sim_x, y_gene) {
  f <- readRDS(paste0('genie3_models/G', 334 + y_gene, '.rds'))
  rmse <- sqrt(mean((f$y - f$predicted)^2))
  n <- nrow(sim_x)
  sim_y <- predict(f, sim_x) + rnorm(n, sd = rmse)
  return(sim_y)
}

# Simulate baseline data
n <- 1e4
sim_x <- sim_x_fn(n)
sim_y <- foreach(g = seq_len(ncol(y)), .combine = cbind) %dopar%
  sim_y_fn(sim_x, g)
colnames(sim_y) <- colnames(y)
baseline <- cbind(sim_x, sim_y)
saveRDS(baseline, 'simulations/baseline.rds')

# Simulate interventions on top 10 TFs by outdegree
most_causal <- order(outdegree, decreasing = TRUE)[seq_len(10)]
idx <- split(seq_len(n), rep(seq_len(10), each = 1000))
names(idx) <- most_causal
sim_w_fn <- function(sim_x, ko) {
  i <- idx[[as.character(ko)]]
  sim_x_prime <- sim_x[i, ]
  sim_x_prime[, ko] <- min(x[, ko]) - 1
  sim_y <- foreach(g = seq_len(ncol(y)), .combine = cbind) %dopar%
    sim_y_fn(sim_x_prime, g)
  colnames(sim_y) <- colnames(y)
  out <- cbind(sim_x_prime, sim_y, W = ko)
  return(out)
}
interventions <- foreach(g = most_causal, .combine = rbind) %do%
  sim_w_fn(sim_x, g)
saveRDS(interventions, 'simulations/interventions.rds')

# Compute phi via kernel PCA for all TFs with outdegree 100 or greater
keep <- which(outdegree >= 100)
phi_fn <- function(tf) {
  # Train on 1k baseline samples
  baseline_x_trn <- baseline[1:1000, tf]
  baseline_y_trn <- baseline[1:1000, 334 + which(adj_mat[, tf] == 1)]
  baseline_trn <- cbind(baseline_x_trn, baseline_y_trn)
  d <- Dist(baseline_trn) %>% keep(lower.tri(.))
  s <- 1 / median(d)
  pca <- kpca(baseline_trn, kernel = 'rbfdot', kpar = list(sigma = s), 
              features = 1)
  # Project remaining baseline data
  baseline_x_tst <- baseline[1001:10000, tf]
  baseline_y_tst <- baseline[1001:10000, 334 + which(adj_mat[, tf] == 1)]
  baseline_tst <- cbind(baseline_x_tst, baseline_y_tst)
  phi0 <- c(rotated(pca), predict(pca, baseline_tst))
  # Project on interventional data
  intervention_x_tst <- interventions[, tf]
  intervention_y_tst <- interventions[, 334 + which(adj_mat[, tf] == 1)]
  intervention_tst <- cbind(intervention_x_tst, intervention_y_tst)
  phi_prime <- as.numeric(predict(pca, intervention_tst))
  # Take difference in phis
  out <- phi_prime - phi0
  return(out)
}
phis <- foreach(g = keep, .combine = cbind) %dopar%
  phi_fn(g)
colnames(phis) <- paste0('phi', seq_len(length(keep)))
saveRDS(phis, 'simulations/phis.rds')



