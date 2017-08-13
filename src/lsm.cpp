// Copyright 2017 <jeremyyee@outlook.com.au>
// Least squares Monte Carlo
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"
#include <Rcpp.h>

// Regression coefficients from singular value decomposition
//[[Rcpp::export]]
arma::vec SVDCoeff(const arma::mat& xreg,
                   const arma::vec& yreg) {
  arma::mat u;
  arma::vec s;
  arma::mat v;
  arma::svd_econ(u, s, v, xreg, "both", "dc");
  arma::vec d = (u.t()) * yreg;
  arma::vec temp(xreg.n_cols, arma::fill::zeros);
  int r = rank(xreg);
  temp(arma::span(0, r - 1)) =
      d(arma::span(0, r - 1)) / s(arma::span(0, r - 1));
  arma::vec svd_coeff = v * temp;
  return svd_coeff;
}

// Best action given regression basis (position control deterministic)
void Optimal(arma::cube& path_values,
             arma::ucube& path_policy,
             const arma::cube& expected_value,
             const arma::mat& reg_basis,
             const arma::cube& reward_values,
             const arma::imat& control,
             const int& tt,
             const std::size_t& n_path,
             const std::size_t& n_pos,
             const std::size_t& n_action,
             const std::size_t& n_dim) {
  // Get fitted values
  arma::mat fitted_expected(n_path, n_pos);
  fitted_expected = reg_basis * expected_value.slice(tt);  // fitted values
  // Compute the fitted values based on position and action
  arma::mat best(n_path, n_pos);  // the best values
  arma::mat compare(n_path, n_action);
  std::size_t pp, aa, nn, ww;
  arma::uword best_action;
  for (pp = 0; pp < n_pos; pp++) {
    compare = reward_values.slice(pp);
    for (aa = 0; aa < n_action; aa++) {
      nn = control(pp, aa) - 1;  // Next position + R index starts at 1
      compare.col(aa) += fitted_expected.col(nn);
    }
    for (ww = 0; ww < n_path; ww++) {
      compare.row(ww).max(best_action);  // Best action according to regression
      path_policy(ww, pp, tt) = best_action + 1;
      path_values(ww, pp, tt) = reward_values(ww, best_action, pp) +
          path_values(ww, control(pp, best_action) - 1, tt + 1);
    }
  }
}

// Best action given regression basis (position control not deterministic)
void Optimal(arma::cube& path_values,
             arma::ucube& path_policy,
             const arma::cube& expected_value,
             const arma::mat& reg_basis,
             const arma::cube& reward_values,
             const arma::cube& control,
             const int& tt,
             const std::size_t& n_path,
             const std::size_t& n_pos,
             const std::size_t& n_action,
             const std::size_t& n_dim) {
  // Get fitted values
  arma::mat fitted_expected(n_path, n_pos);
  fitted_expected = reg_basis * expected_value.slice(tt);  // fitted values
  // Compute the fitted values based on position and action
  arma::mat best(n_path, n_pos);  // The best values
  arma::mat compare(n_path, n_action);
  arma::vec trans_prob(n_pos);  // The transition probabilities
  std::size_t pp, aa, nn, ww;
  arma::uword best_action;
  for (pp = 0; pp < n_pos; pp++) {
    compare = reward_values.slice(pp);
    for (aa = 0; aa < n_action; aa++) {
      trans_prob = control.tube(pp, aa);
      compare.col(aa) += fitted_expected * trans_prob;
    }
    for (ww = 0; ww < n_path; ww++) {
      compare.row(ww).max(best_action);  // Best action according to regression
      path_policy(ww, pp, tt) = best_action + 1;
      trans_prob = control.tube(pp, best_action);
      path_values(ww, pp, tt) = reward_values(ww, best_action, pp)
          + arma::accu(path_values.slice(tt + 1).row(ww) * trans_prob);
    }
  }
}

// Least squares Monte Carlo
//[[Rcpp::export]]
Rcpp::List LSM(Rcpp::NumericVector path_,
               const Rcpp::Function& Reward_,
               const Rcpp::Function& Scrap_,
               Rcpp::NumericVector control_,
               const arma::umat& basis,
               const bool& intercept,
               const std::string& basis_type) {
  // Extract parameters
  std::size_t n_dec, n_path, n_dim, n_pos, n_action;
  const arma::ivec p_dims = path_.attr("dim");
  n_dec = p_dims(0);
  n_path = p_dims(1);
  n_dim = (p_dims.n_elem == 2) ? 1 : p_dims(2);
  arma::cube path(path_.begin(), n_dec, n_path, n_dim, false);
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
  // Perform the Bellman recursion starting at last time epoch
  Rcpp::Rcout << "At dec: " << n_dec - 1 << "...";
  arma::cube path_values(n_path, n_pos, n_dec);
  arma::ucube path_policy(n_path, n_pos, n_dec - 1);
  arma::mat temp_states(n_dim, n_path);
  arma::mat states(n_path, n_dim);
  temp_states = path.tube(arma::span(n_dec - 1), arma::span::all);
  states = temp_states.t();
  path_values.slice(n_dec - 1) = Rcpp::as<arma::mat>(
      Scrap_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
  arma::cube expected_value(n_terms, n_pos, n_dec - 1);  // Regression fit
  arma::mat reg_basis(n_path, n_terms);
  arma::cube reward_values(n_path, n_action, n_pos);
  // Perform Backward induction
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << tt << "...";
    temp_states = path.tube(arma::span(tt), arma::span::all);
    states = temp_states.t();
    // Compute the fitted continuation value
    if (basis_type == "power") {
      reg_basis = PBasis(states, basis, intercept, n_terms);
    }
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      expected_value.slice(tt).col(pp) =
          SVDCoeff(reg_basis, path_values.slice(tt + 1).col(pp));
    }
    reward_values = Rcpp::as<arma::cube>(
        Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states)), tt + 1));  // Indexing of this??
    if (full_control) {
      Optimal(path_values, path_policy, expected_value, reg_basis,
              reward_values, control, tt, n_path, n_pos, n_action, n_dim);
    } else {
      Optimal(path_values, path_policy, expected_value, reg_basis,
              reward_values, control2, tt, n_path, n_pos, n_action, n_dim);
    }
  }
  Rcpp::Rcout << "end\n";
  return Rcpp::List::create(Rcpp::Named("value") = path_values,
                            Rcpp::Named("policy") = path_policy,
                            Rcpp::Named("expected") = expected_value);
}

