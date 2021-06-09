# Load libraries, register cores
library(data.table)
library(glmnet)
library(broom)
library(tidyverse)
library(doMC)
registerDoMC(8)

# Import data
z <- readRDS('simulations/baseline.rds')
w <- readRDS('simulations/w.rds')
phis <- readRDS('simulations/phis.rds')
phi_hat <- readRDS('simulations/phi_hat.rds')
p <- ncol(phis)

# Load lasso
lasso_f <- readRDS('lasso_f.rds')
beta_hat <- as.numeric(coef(lasso_f, s = 'lambda.min'))[-1]
nonzero <- which(beta_hat != 0)

# Select true mediators
rho_fn <- function(phi_idx) {
  cor.test(phis[, phi_idx], w, method = 'spearman') %>%
    tidy(.) %>%
    mutate(idx = phi_idx) 
}
df <- foreach(j = seq_len(p), .combine = rbind) %dopar%
  rho_fn(j)
true_mediators <- df %>%
  arrange(p.value) %>%
  pull(idx) %>%
  head(25)

# Which phi's are significantly correlated with W at (adjusted) alpha = 1%?
pass_ci_pre <- df %>%
  mutate(q.value = p.adjust(p.value, method = 'holm')) %>%
  filter(q.value <= 0.01) %>%
  pull(idx)

# Same but just use training data, and only check nonzero thetas
trn_idx <- c(1:5000, 7001:1000)
rho_fn <- function(phi_idx) {
  cor.test(phis[trn_idx, phi_idx], w[trn_idx], method = 'spearman') %>%
    tidy(.) %>%
    mutate(idx = phi_idx) %>%
    select(idx, p.value)
}
pass_ci_post <- foreach(j = nonzero, .combine = rbind) %dopar%
  rho_fn(j) %>%
  mutate(q.value = p.adjust(p.value, method = 'holm')) %>%
  filter(q.value <= 0.01) %>%
  pull(idx)

# Export
out <- data.table(
  'phi' = paste0('phi', seq_len(p)),
  'true_mediator' = ifelse(seq_len(p) %in% true_mediators, 1, 0),
  'selected_by_lasso' = ifelse(beta_hat != 0, 1, 0),
  'selected_by_ci_pre_lasso' = ifelse(seq_len(p) %in% pass_ci_pre, 1, 0),
  'selected_by_ci_post_lasso' = ifelse(seq_len(p) %in% pass_ci_post, 1, 0)
)
out[, declared_mediator := ifelse(selected_by_lasso == 1 & 
                                  selected_by_ci_post_lasso == 1, 1, 0)]
fwrite(out, 'genomic_mediator_discovery.csv')




