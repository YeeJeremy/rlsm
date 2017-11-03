// Copyright 2017 <jeremyyee@outlook.com.au>
// Regression basis
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"

// Power polynomial regression basis
arma::mat PBasis(const arma::mat& data,
                 const arma::umat& basis,
                 const bool& intercept,
                 const std::size_t& n_terms,
                 const arma::uvec& reccur_limit) {
  std::size_t n_basis1 = basis.n_rows;
  std::size_t n_basis2 = basis.n_cols;
  arma::mat output(data.n_rows, n_terms);
  std::size_t counter = 0;
  // Fill in the basis
  for (std::size_t rr = 0; rr < n_basis1; rr++) {
    for (std::size_t cc = 0; cc < n_basis2; cc++) {
      if (basis(rr, cc) != 0) {
        output.col(counter) = arma::pow(data.col(rr), cc + 1);
        counter++;
      }
      if ((cc + 2) > reccur_limit(rr)) {  // no cont since zeros trailing
        break;
      }
    }
  }
  if (intercept) {
    output.col(n_terms - 1).fill(1.);  // Intercept goes at the end
  }
  return output;
}

// Position of ending 1 of each row in the basis (for recurrence limit)
arma::uvec ReccurLimit(const arma::umat& basis) {
  std::size_t n_basis1 = basis.n_rows;
  int n_basis2 = basis.n_cols;
  arma::uvec output(n_basis1);
  arma::uword counter;
  for (std::size_t rr = 0; rr < n_basis1; rr++) {
    counter = 0;
    for (int cc = n_basis2 - 1; cc >= 0; cc--) {
      if (basis(rr, cc) == 0) {
        counter++;
      } else {
        break;
      }
    }
    output(rr) = n_basis2 - counter;
  }
  return output;
}

// Definition for Laguerre polynomials
arma::mat Laguerre(const arma::vec data,
                   const std::size_t n) {
  std::size_t n_terms = data.n_elem;
  arma::mat output(n_terms, n);
  output.col(0) = 1.0 - data;
  for (std::size_t ii = 1; ii < n; ii++) {
    if (ii == 1) {
      output.col(ii) =
          ((2 * ii + 1 - data) % output.col(ii - 1) - ii) / (ii + 1);
    } else {
       output.col(ii) =
           ((2 * ii + 1 - data) % output.col(ii - 1) - ii * output.col(ii - 2))
           / (ii + 1);
    }
  }
  return output;
}

// Laguerre polynomial regression basis
arma::mat LBasis(const arma::mat& data,
                 const arma::umat& basis,
                 const bool& intercept,
                 const std::size_t& n_terms,
                 const arma::uvec& reccur_limit) {
  std::size_t n_basis1 = basis.n_rows;
  std::size_t n_basis2 = basis.n_cols;
  std::size_t n_data = data.n_rows;
  arma::mat output(n_data, n_terms);
  std::size_t counter = 0;
  arma::mat laguerre(n_data, n_basis2);
  // Fill in the basis
  for (std::size_t rr = 0; rr < n_basis1; rr++) {
    laguerre = Laguerre(data.col(rr), reccur_limit(rr));
    for (std::size_t cc = 0; cc < n_basis2; cc++) {
      if (basis(rr, cc) != 0) {
        output.col(counter) = laguerre.col(cc);
        counter++;
      }
      if ((cc + 2) > reccur_limit(rr)) {  // no cont since zeros trailing
        break;
      }
    }
  }
  if (intercept) {
    output.col(n_terms - 1).fill(1.);  // Intercept goes at the end
  }
  return output;
}

// Position of ending knot of each row in the basis
arma::uvec ReccurLimit2(const arma::mat& knots) {
  std::size_t n_knots1 = knots.n_rows;
  std::size_t n_knots2 = knots.n_cols;
  arma::uvec output(n_knots1);
  arma::uword counter;
  for (std::size_t rr = 0; rr < n_knots1; rr++) {
    counter = 0;
    for (std::size_t cc = 0; cc < n_knots2; cc++) {
      if (Rcpp::NumericVector::is_na(knots(rr, cc))) {
        break;
      } else {
        counter++;
      }
    }
    output(rr) = counter;
  }
  return output;
}

// Linear spline regression basis
arma::mat LSplineBasis(const arma::mat& data,
                       const arma::mat& knots,
                       const std::size_t& n_knots,
                       const arma::uvec& reccur_limit2) {
  std::size_t n_knots1 = knots.n_rows;
  std::size_t n_knots2 = knots.n_cols;
  std::size_t n_data = data.n_rows;
  arma::mat output(n_data, n_knots);
  std::size_t counter = 0;
  arma::vec knotsVec(n_data);
  arma::vec zeroVec = arma::zeros<arma::vec>(n_data);
  // Fill in the basis
  for (std::size_t rr = 0; rr < n_knots1; rr++) {
    for (std::size_t cc = 0; cc < n_knots2; cc++) {
      if ((cc + 1) > reccur_limit2(rr)) {
        break;
      } 
      knotsVec.fill(knots(rr, cc));
      output.col(counter) = arma::max(data.col(rr) - knotsVec, zeroVec);
      counter++;
    }
  }
  return output;
}
