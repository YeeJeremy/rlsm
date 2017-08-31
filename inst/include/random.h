// Copyright 2017 <jeremyyee@outlook.com.au>
// Header file for some random number generation
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_RANDOM_H_
#define INST_INCLUDE_RANDOM_H_

#include <RcppArmadillo.h>

// Generate correlated Gaussians using cholesky decomposition
arma::mat CorrNormal(const int& n,
                     const arma::mat& corr);

#endif  // INST_INCLUDE_RANDOM_H_
