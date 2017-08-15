// Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Extracting the prescribed policy
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"
#include <Rcpp.h>

// Extracting the prescribed policy on a set of sample paths
//[[Rcpp::export]]
arma::ucube PathPolicy(const arma::cube& path,
                       const arma::cube& expected_value,
                       const Rcpp::Function& Reward_,
                       Rcpp::NumericVector control_,
                       const arma::umat& basis,
                       const bool& intercept,
                       const std::string& basis_type) {
  // Extract parameters
  std::size_t n_dec, n_path, n_dim, n_pos, n_action;
  n_dec = path.n_rows;
  n_path = path.n_cols;
  n_dim = path.n_slices;
  const arma::ivec c_dims = control_.attr("dim");
  n_pos = c_dims(0);
  n_action = c_dims(1);
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
  std::size_t n_terms = arma::accu(basis);  // Number of features in basis
  if (intercept) { n_terms++; }
  arma::mat reg_basis(n_path, n_terms);
  // Extract the prescribed policy
  arma::ucube policy(n_dec - 1, n_pos, n_path);
  std::size_t tt, pp, aa, nn;
  arma::mat fitted_expected(n_path, n_pos);
  arma::mat compare(n_path, n_action);
  arma::cube reward_values(n_path, n_action, n_pos);
  arma::mat t_states(n_dim, n_path);
  arma::mat states(n_path, n_dim);
  if (full_control) {
    for (tt = 0; tt < n_dec - 1; tt++) {
      if (n_dim != 1) {
        states = path.tube(arma::span(tt), arma::span::all);
      } else {  // armadillo doesnt behave the way I want when n_dim = 1
        t_states = path.tube(arma::span(tt), arma::span::all);
        states = t_states.t();
      }
      if (basis_type == "power") {
        reg_basis = PBasis(states, basis, intercept, n_terms);
      }
      reward_values = Rcpp::as<arma::cube>(
          Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states)), tt + 1));
      fitted_expected = reg_basis * expected_value.slice(tt);  // fitted values
      for (pp = 0; pp < n_pos; pp++) {
        compare = reward_values.slice(pp);
        for (aa = 0; aa < n_action; aa++) {
          nn = control(pp, aa) - 1;  // Next position + R index starts at 1
          compare.col(aa) += fitted_expected.col(nn);
        }
        policy.tube(tt, pp) = arma::index_max(compare, 1);  // R indexing
      }
    }
  } else {
    arma::vec trans_prob(n_pos);  // The transition probabilities
    for (tt = 0; tt < n_dec - 1; tt++) {
      if (n_dim != 1) {
        states = path.tube(arma::span(tt), arma::span::all);
      } else {  // armadillo doesnt behave the way I want when n_dim = 1
        t_states = path.tube(arma::span(tt), arma::span::all);
        states = t_states.t();
      }
      if (basis_type == "power") {
        reg_basis = PBasis(states, basis, intercept, n_terms);
      }
      reward_values = Rcpp::as<arma::cube>(
          Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states)), tt + 1));
      fitted_expected = reg_basis * expected_value.slice(tt);  // fitted values
      for (pp = 0; pp < n_pos; pp++) {
        compare = reward_values.slice(pp);
        for (aa = 0; aa < n_action; aa++) {
          trans_prob = control2.tube(pp, aa);
          compare.col(aa) += fitted_expected * trans_prob;
        }
        policy.tube(tt, pp) = arma::index_max(compare, 1);  // R indexing
      }
    }
  }
  return (policy + 1);
}
