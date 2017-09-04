// Copyright 2017 <jeremyyee@outlook.com.au>
// Header file for regression basis
////////////////////////////////////////////////////////////////////////////////

#ifndef INST_INCLUDE_BASIS_H_
#define INST_INCLUDE_BASIS_H_

#include <RcppArmadillo.h>
#include <string>

// Power polynomial regression basis
arma::mat PBasis(const arma::mat& data,
                 const arma::umat& basis,
                 const bool& intercept,
                 const std::size_t& n_terms, 
                 const arma::uvec& reccur_limit);

// Finding the ending 1 of each row in the basis (for recurrence limit)
arma::uvec ReccurLimit(const arma::umat& basis);

// Laguerre polynomial regression basis
arma::mat LBasis(const arma::mat& data,
                 const arma::umat& basis,
                 const bool& intercept,
                 const std::size_t& n_terms,
                 const arma::uvec& recurr_limit);

#endif  // INST_INCLUDE_BASIS_H_
