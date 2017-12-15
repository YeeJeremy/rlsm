// Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Extracting the prescribed policy
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"

// Extracting the prescribed policy on a set of sample paths
//[[Rcpp::export]]
arma::ucube PathPolicy(const arma::cube& path,
                       const arma::cube& expected,
                       const Rcpp::Function& Reward_,
                       Rcpp::NumericVector control_,
                       const arma::umat& basis,
                       const std::string& basis_type,
                       const bool& spline,
                       const arma::mat& knots,
                       const Rcpp::Function& Basis_,
                       const std::size_t n_rbasis) {
  // Extract parameters
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  const std::size_t n_basis = expected.n_rows;
  std::size_t n_terms = arma::accu(basis);
  arma::uvec reccur_limit2(knots.n_rows);
  std::size_t n_knots = 0;
  bool intercept = true;
  if (spline) {
    reccur_limit2 = ReccurLimit2(knots);
    n_knots = arma::sum(reccur_limit2);
    if ((n_terms + n_knots + n_rbasis) == n_basis) {
      intercept = false;
    }
  } else {
    if ((n_terms + n_rbasis) == n_basis) {
      intercept = false;
    }
  }
  if (intercept) {
    n_terms++;
  }
  const arma::ivec c_dims = control_.attr("dim");
  const std::size_t n_pos = c_dims(0);
  const std::size_t n_action = c_dims(1);
  // Determine if full control or partial control of finite state Markov chain
  arma::cube control2;
  arma::imat control;
  bool full_control;
  if (c_dims.n_elem == 3) {
    full_control = false;
    arma::cube temp_control2(control_.begin(), n_pos, n_action, n_pos, false);
    control2 = temp_control2;
  } else {
    full_control = true;
    arma::mat temp_control(control_.begin(), n_pos, n_action, false);
    control = arma::conv_to<arma::imat>::from(temp_control);
  }
  // Extract information about regression basis
  arma::mat reg_basis(n_path, n_terms + n_knots + n_rbasis);
  arma::uvec reccur_limit(basis.n_rows);
  reccur_limit = ReccurLimit(basis);
  // Extract the prescribed policy
  arma::ucube policy(n_path, n_pos, n_dec - 1);
  std::size_t tt, pp, aa, nn;
  arma::mat fitted_expected(n_path, n_pos);
  arma::mat compare(n_path, n_action);
  arma::cube reward_values(n_path, n_action, n_pos);
  arma::mat states(n_path, n_dim);
  if (full_control) {
    for (tt = 0; tt < n_dec - 1; tt++) {
      states = path.slice(tt);
      if (n_terms > 0) {
        if (basis_type == "power") {
          reg_basis.cols(0, n_terms - 1) =
              PBasis(states, basis, intercept, n_terms, reccur_limit);
        } else if (basis_type == "laguerre") {
          reg_basis.cols(0, n_terms - 1) =
              LBasis(states, basis, intercept, n_terms, reccur_limit);
        }
      }
      if (spline) {
        reg_basis.cols(n_terms, n_terms + n_knots - 1) =
            LSplineBasis(states, knots, n_knots, reccur_limit2);
      }
      if (n_rbasis > 0) {
        reg_basis.cols(n_terms + n_knots, n_terms + n_knots + n_rbasis - 1) =
            Rcpp::as<arma::mat>(Basis_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
      }
      reward_values = Rcpp::as<arma::cube>(
          Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states)), tt + 1));
      fitted_expected = reg_basis * expected.slice(tt);  // fitted values
      for (pp = 0; pp < n_pos; pp++) {
        compare = reward_values.slice(pp);
        for (aa = 0; aa < n_action; aa++) {
          nn = control(pp, aa) - 1;  // Next position + R index starts at 1
          compare.col(aa) += fitted_expected.col(nn);
        }
        policy.slice(tt).col(pp) = arma::index_max(compare, 1);
      }
    }
  } else {
    arma::vec trans_prob(n_pos);  // The transition probabilities
    for (tt = 0; tt < n_dec - 1; tt++) {
      states = path.slice(tt);
      if (n_terms > 0) {
        if (basis_type == "power") {
          reg_basis.cols(0, n_terms - 1) =
              PBasis(states, basis, intercept, n_terms, reccur_limit);
        } else if (basis_type == "laguerre") {
          reg_basis.cols(0, n_terms - 1) =
              LBasis(states, basis, intercept, n_terms, reccur_limit);
        }
      }
      if (spline) {
        reg_basis.cols(n_terms, n_terms + n_knots - 1) =
            LSplineBasis(states, knots, n_knots, reccur_limit2);
      }
      if (n_rbasis > 0) {
        reg_basis.cols(n_terms + n_knots, n_terms + n_knots + n_rbasis - 1) =
            Rcpp::as<arma::mat>(Basis_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
      }
      reward_values = Rcpp::as<arma::cube>(
          Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states)), tt + 1));
      fitted_expected = reg_basis * expected.slice(tt);  // fitted values
      for (pp = 0; pp < n_pos; pp++) {
        compare = reward_values.slice(pp);
        for (aa = 0; aa < n_action; aa++) {
          trans_prob = control2.tube(pp, aa);
          compare.col(aa) += fitted_expected * trans_prob;
        }
        policy.slice(tt).col(pp) = arma::index_max(compare, 1);
      }
    }
  }
  return (policy + 1);  // C++ indexing to R indexing
}
