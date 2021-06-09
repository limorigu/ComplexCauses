# Load libraries
library(data.table)
library(tidyverse)
library(doMC)

# Loop function
ci_test <- function(dat) {
  df <- rbind(
    fread(paste0('../Python_Img_Humor/out/viz/ImgPertSim/residuals_nested=TRUE_', dat, '.csv')), 
    fread(paste0('../Python_Img_Humor/out/viz/ImgPertSim/residuals_nested=FALSE_', dat, '.csv'))
  )
  n <- nrow(df) / 2
  wilcox_fn <- function(j) {
    err <- abs(df[[j]])
    delta <- err[1:n] - err[(n+1):(2*n)]
    p_value <- wilcox.test(delta, alt = 'greater')$p.value
    out <- data.table(phi = paste0('phi', j), p_value = p_value)
    return(out)
  }
  out <- foreach(j = seq_len(ncol(df)), .combine = rbind) %do% wilcox_fn(j)
  out[, adj_p_value := p.adjust(p_value, method = 'holm')][, dataset := dat]
  return(out)
}
dats <- c('PertImgSim')
res <- foreach(d = dats, .combine = rbind) %do% ci_test(d)
fwrite(res, 'ci_test_results_img.csv')


