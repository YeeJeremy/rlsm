// Copyright 2017 <jeremyyee@outlook.com.au>
// Some random number generation
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/random.h"
#include <Rcpp.h>

// Generate correlated Gaussians using cholesky decomposition
//[[Rcpp::export]]
arma::mat CorrNormal(const int& n,
                     const arma::mat& corr) {
  int ncols = corr.n_cols;
  arma::mat Z = arma::randn(n, ncols);
  return Z * arma::chol(corr);
}
