// Copyright 2017 <jeremyyee@outlook.com.au>
// Regression basis
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"

// Power polynomial regression basis
//[[Rcpp::export]]
arma::mat PBasis(const arma::mat& data,
                 const arma::umat& basis,
                 const bool& intercept,
                 const std::size_t& n_terms) {
  std::size_t n_basis1 = basis.n_rows;
  std::size_t n_basis2 = basis.n_cols;
  arma::mat output(data.n_rows, n_terms);
  std::size_t counter = 0;
  // Fill in the ordinary terms
  for (std::size_t rr = 0; rr < n_basis1; rr++) {
    for (std::size_t cc = 0; cc < n_basis2; cc++) {
      if (basis(rr, cc) != 0) {
        output.col(counter) = arma::pow(data.col(rr), cc + 1);
        counter++;
      }
    }
  }
  if (intercept) {
    output.col(n_terms - 1).fill(1.);  // Intercept goes at the end
  }
  return output;
}
