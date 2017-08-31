// Copyright 2017 <jeremyyee@outlook.com.au>
// Header file for some random number generation
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_RANDOM_H_
#define INST_INCLUDE_RANDOM_H_

#include <RcppArmadillo.h>

// Generate correlated Gaussians using cholesky decomposition
arma::mat CorrNormal(const int& n,
                     const arma::mat& corr);

// Generate correlated Brownian motion with drift
arma::cube CBM(const arma::vec& start,
               const arma::vec& mu,
               const arma::vec& vol,
               const arma::mat& corr,
               const int& n_dec,
               const int& n_path,
               const bool& antithetic);


#endif  // INST_INCLUDE_RANDOM_H_
