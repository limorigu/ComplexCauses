# Load libraries, register cores
library(data.table)
library(glmnet)
library(broom)
library(tidyverse)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(123, kind = "L'Ecuyer-CMRG")

# Import, prep data
z <- readRDS('simulations/baseline.rds')
w <- readRDS('simulations/w.rds')
x <- readRDS('simulations/interventions.rds')
phis <- readRDS('simulations/phis.rds')
phi_hat <- readRDS('simulations/phi_hat.rds')
k <- ncol(phis)

# Most/least variable in W?
rho_fn <- function(phi_idx) {
  cor.test(phis[, phi_idx], w, method = 'spearman') %>%
    tidy(.) %>%
    mutate(idx = phi_idx) %>%
    select(idx, statistic, p.value)
}
df <- foreach(j = seq_len(k), .combine = rbind) %dopar%
  rho_fn(j) %>%
  mutate(q.value = p.adjust(p.value, method = 'fdr')) %>%
  arrange(p.value)

# Desiderata for Y: 
# (1) linear combination of phis such that
# (2) some phis receive zero weight,
# (3) some phis with nonzero weight are invariant w/r/t W, and
# (4) some phis with nonzero weight vary w/r/t W -- i.e., are causal mediators.

# Define beta vector
most_variable <- head(df$idx, 25)
least_variable <- tail(df$idx, 25)
amplitude <- 4
nonzeros <- rnorm(50, mean = amplitude, sd = 1) * 
  sample(c(1, -1), size = 50, replace = TRUE)
beta <- double(k)
beta[c(most_variable, least_variable)] <- nonzeros

# Define response
n <- nrow(phis)
y <- phis %*% beta + rnorm(n, sd = 1)
saveRDS(y, 'simulations/y.rds')

################################################################################

### LEARN LASSO WEIGHTS ###
trn_idx <- c(1:5000, 7001:1000)
tst_idx <- 5001:7000
lasso_f <- cv.glmnet(x = phi_hat[trn_idx, ], y = y[trn_idx], parallel = TRUE)
saveRDS(lasso_f, 'lasso_f.rds')

# Coefficients look ok?
beta_df <- data.frame(
  expected = beta, observed = as.numeric(coef(lasso_f, s = 'lambda.min'))[-1]
)
with(beta_df, cor(expected, observed)^2)
lo <- min(beta_df$expected, beta_df$observed)
hi <- max(beta_df$expected, beta_df$observed)
ggplot(beta_df, aes(expected, observed)) + 
  geom_point() + 
  geom_abline(slope = 1, intercept = 0, color = 'red') + 
  xlim(lo, hi) + ylim(lo, hi) + 
  theme_bw()

# Model fit?
y_hat <- predict(lasso_f, phi_hat[tst_idx, ], s = 'lambda.min')
lasso_df <- data.frame(
  expected = y[tst_idx], observed = as.numeric(y_hat)
)
with(lasso_df, cor(expected, observed)^2)
with(lasso_df, mean((expected - observed)^2))
lo <- min(lasso_df$expected, lasso_df$observed)
hi <- max(lasso_df$expected, lasso_df$observed)
ggplot(lasso_df, aes(expected, observed)) + 
  geom_point(size = 0.5, alpha = 0.5) + 
  geom_abline(slope = 1, intercept = 0, color = 'red') + 
  xlim(lo, hi) + ylim(lo, hi) + 
  theme_bw()








