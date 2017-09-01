// Copyright 2017 <jeremyyee@outlook.com.au>
// Compute the additive duals using our approximations
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"

// Compute the additive duals
//[[Rcpp::export]]
arma::cube AddDual(const arma::cube& path,
                   Rcpp::NumericVector subsim_,
                   const arma::cube& fitted_value,
                   const Rcpp::Function& Scrap_,
                   const arma::umat& basis,
                   const std::string& basis_type) {
  // Extract parameters
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  const std::size_t n_pos = fitted_value.n_cols;
  const std::size_t n_terms = fitted_value.n_rows;
  bool intercept = true;
  if (n_terms == arma::accu(basis)) {
    intercept = false;
  }
  const arma::ivec s_dims = subsim_.attr("dim");
  const std::size_t n_subsim = s_dims(0);
  arma::cube subsim(subsim_.begin(), n_subsim, n_dim, n_path * (n_dec - 1), false);
  // Additive duals
  arma::cube add_dual(n_path, n_pos, n_dec - 1, arma::fill::zeros);
  arma::mat path_basis(n_path, n_terms);
  arma::mat subsim_basis(n_subsim, n_terms);
  std::size_t tt;
  for (tt = 0; tt < n_dec - 2; tt++) {
    // Find the average of the subsimulation paths
    for (std::size_t pp = 0; pp < n_path; pp++) {
      if (basis_type == "power") {
        subsim_basis =
            PBasis(subsim.slice(n_path * tt + pp), basis, intercept, n_terms);
      }
      // Average for each path
      add_dual.slice(tt).row(pp) +=
          arma::sum(subsim_basis * fitted_value.slice(tt + 1), 0);
    }
    add_dual.slice(tt) = (1.0 / n_subsim) * add_dual.slice(tt);
    // Find the realised value
    if (basis_type == "power") {
      path_basis = PBasis(path.slice(tt + 1), basis, intercept, n_terms);
    }
    add_dual.slice(tt) -= path_basis * fitted_value.slice(tt + 1);
  }
  // Find the duals for the scrap
  // Find the average of the subsimulation paths
  tt = n_dec - 2;
  for (std::size_t pp = 0; pp < n_path; pp++) {
    // Average for each path
    add_dual.slice(tt).row(pp) += arma::sum(Rcpp::as<arma::mat>(
        Scrap_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(subsim.slice(n_path * tt + pp))))),
                                            0);
  }
  add_dual.slice(tt) = (1.0 / n_subsim) * add_dual.slice(tt);
  // Find the realised value
  add_dual.slice(tt) -= Rcpp::as<arma::mat>(
        Scrap_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt + 1)))));
  return add_dual;
}
