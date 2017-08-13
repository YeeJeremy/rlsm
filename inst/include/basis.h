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
                 const std::size_t& n_terms); 

// Least squares Monte Carlo
Rcpp::List LSM(Rcpp::NumericVector path_,
               const Rcpp::Function& Reward_,
               const Rcpp::Function& Scrap_,
               Rcpp::NumericVector control_,
               const arma::umat& basis,
               const bool& intercept,
               const std::string& basis_type);

// Extracting the prescribed policy
arma::ucube PathPolicy(Rcpp::NumericVector path_,
                       const arma::cube& expected_value,
                       const Rcpp::Function& Reward_,
                       Rcpp::NumericVector control_,
                       const arma::umat& basis,
                       const bool& intercept,
                       const std::string& basis_type); 

#endif  // INST_INCLUDE_BASIS_H_
