// Copyright 2017 <jeremyyee@outlook.com.au>
// Some random number generation
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/random.h"

// Generate correlated Gaussians using cholesky decomposition
// Returns matric where eacl col = each dim
//[[Rcpp::export]]
arma::mat CorrNormal(const int& n,
                     const arma::mat& corr) {
  int ncols = corr.n_cols;
  arma::mat Z = arma::randn(n, ncols);
  return Z * arma::chol(corr);
}
