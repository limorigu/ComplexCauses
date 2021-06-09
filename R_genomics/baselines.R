# Load libraries, register cores
library(data.table)
library(glmnet)
library(kernlab)
library(ranger)
library(gbm)
library(tidyverse)
library(ggsci)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(999, kind = "L'Ecuyer-CMRG")

# Import, prep data
z <- readRDS('simulations/baseline.rds')
w <- readRDS('simulations/w.rds')
x <- readRDS('simulations/interventions.rds')
phis <- readRDS('simulations/phis.rds')
phi_hat <- readRDS('simulations/phi_hat.rds')
y <- readRDS('simulations/y.rds')

# Load pretrained lasso
trn_idx <- c(1:5000, 7001:1000)
tst_idx <- 5001:7000
lasso_f <- readRDS('lasso_f.rds')
y_hat <- predict(lasso_f, phi_hat[tst_idx, ], s = 'lambda.min')
lasso_df <- data.frame(
  expected = y[tst_idx], observed = as.numeric(y_hat)
)

################################################################################

### Benchmarks ###

# Preliminaries
n <- 500
x <- cbind(w, z)
p <- ncol(x)
dat <- as.data.frame(x)
dat$y <- y

# Loop function
outer_loop <- function(b) {
  # Where are we?
  print(paste('b =', b))
  # Draw random test sample
  tst <- sample(tst_idx, 0.2 * n)
  # Permute training sample
  trn <- setdiff(tst_idx, tst)[sample.int(0.8 * n)]
  # Run inner loop
  inner_loop <- function(prop) {
    # Where are we?
    print(paste('prop =', prop))
    # Use first prop * length(trn) samples
    trn_p <- trn[seq_len(prop * length(trn))]
    # Lasso
    f1 <- cv.glmnet(x = x[trn_p, ], y = y[trn_p])
    # SVR
    f2 <- ksvm(x = x[trn_p, ], y = y[trn_p])
    # RF
    f3 <- ranger(y ~ ., data = dat[trn_p, ], mtry = floor(p / 3), 
                 num.threads = 8)
    # GBM
    f4 <- gbm(y ~ ., data = dat[trn_p, ], distribution = 'gaussian',
              interaction.depth = 2, n.minobsinnode = 5)
    # Test performance
    yhat_f1 <- as.numeric(predict(f1, x[tst, ], s = 'lambda.min'))
    yhat_f2 <- as.numeric(predict(f2, x[tst, ]))
    yhat_f3 <- predict(f3, dat[tst, ])$predictions
    yhat_f4 <- predict(f4, dat[tst, ])
    # Export
    out <- data.frame(
      'proportion' = prop, 'b' = b, 
      'model' = c('lasso', 'svr', 'rf', 'gbm', 'ours'),
      'mse' = c(
        mean((y[tst] - yhat_f1)^2), mean((y[tst] - yhat_f2)^2), 
        mean((y[tst] - yhat_f3)^2), mean((y[tst] - yhat_f4)^2), 
        with(lasso_df, mean((expected - observed)^2))
      )
    )
    return(out)
  }
  out <- foreach(p = seq(0.1, 1, 0.1), .combine = rbind) %do% 
    inner_loop(p)
  return(out)
}
res <- foreach(i = 1:10, .combine = rbind) %do%
  outer_loop(i)
df <- as.data.table(res)
fwrite(df, 'genomic_baselines.csv')

# Plot results
df[, err := mean(mse), by = .(proportion, model)]
df <- unique(select(df, -b, -mse))
colnames(df)[3] <- 'mse'
ggplot(df, aes(proportion, mse, color = model)) + 
  geom_line() + 
  scale_color_d3() +
  theme_bw()


