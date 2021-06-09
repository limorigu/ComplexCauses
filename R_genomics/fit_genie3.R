# Load libraries, register cores
library(data.table)
library(randomForest)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(42, kind = "L'Ecuyer-CMRG")

# Import data, downloaded from:
# http://dreamchallenges.org/project/dream-5-network-inference-challenge/
mat <- as.matrix(fread('net3_expression_data.tsv'))

# Scale (per Huyn-Thu et al., 2010)
mat <- scale(mat)

# First 334 genes are transcription factors, rest are not
x <- mat[, seq_len(334)]
y <- mat[, 335:ncol(mat)]

# Loop through, with hyperparameters as in Huyn-Thu et al., 2010
rf_loop <- function(gene) {
  f <- randomForest(x, y[, gene], ntree = 1000, mtry = floor(sqrt(334)),
                    importance = TRUE)
  saveRDS(f, paste0('genie3_models/G', 334 + gene, '.rds'))
  out <- f$importance[, 2]
  return(out)
}

# Execute in parallel, save to disk
imp <- foreach(g = seq_len(ncol(y)), .combine = rbind) %dopar%
  rf_loop(g)
fwrite(imp, 'adj_mat.csv')