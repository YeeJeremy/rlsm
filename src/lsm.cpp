// Copyright 2017 <jeremyyee@outlook.com.au>
// Least squares Monte Carlo
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"

// Regression coefficients from singular value decomposition
// Accounting for rank defficiency
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
      path_values(ww, pp, tt) = reward_values(ww, best_action, pp) +
          path_values(ww, control(pp, best_action) - 1, tt + 1);
    }
  }
}

// Best action given regression basis (position control not deterministic)
void Optimal(arma::cube& path_values,
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
      trans_prob = control.tube(pp, best_action);
      path_values(ww, pp, tt) = reward_values(ww, best_action, pp)
          + arma::accu(path_values.slice(tt + 1).row(ww) * trans_prob);
    }
  }
}

// Least squares Monte Carlo
//[[Rcpp::export]]
Rcpp::List LSM(const arma::cube& path,
               const Rcpp::Function& Reward_,
               const Rcpp::Function& Scrap_,
               Rcpp::NumericVector control_,
               const arma::umat& basis,
               const bool& intercept,
               const std::string& basis_type,
               const bool& spline,
               const arma::mat& knots,
               const Rcpp::Function& Basis_,
               const std::size_t n_rbasis,
               const Rcpp::Function& Reg_,
               const bool& useSVD) {
  // Extract parameters
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  const arma::ivec c_dims = control_.attr("dim");
  const std::size_t n_pos = c_dims(0);
  const std::size_t n_action = c_dims(1);
  // Determine if full control or partial control of finite state Markov chain
  arma::imat control;  // full control
  arma::cube control2;  // partial control
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
  if (intercept) {
    n_terms++;
  }
  arma::uvec reccur_limit(basis.n_rows);
  reccur_limit = ReccurLimit(basis);
  arma::uvec reccur_limit2(knots.n_rows);
  std::size_t n_knots = 0;
  if (spline) {
    reccur_limit2 = ReccurLimit2(knots);
    n_knots = arma::sum(reccur_limit2);
  }
  // Perform the Bellman recursion starting at last time epoch
  Rcpp::Rcout << "At dec: " << n_dec  << "...";
  arma::cube path_values(n_path, n_pos, n_dec);
  arma::vec temp_values(n_path);
  arma::mat states(n_path, n_dim);
  states = path.slice(n_dec - 1);
  path_values.slice(n_dec - 1) = Rcpp::as<arma::mat>(
      Scrap_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
  arma::cube expected_value(n_terms + n_knots + n_rbasis, n_pos, n_dec - 1);
  arma::mat reg_basis(n_path, n_terms + n_knots + n_rbasis);
  arma::cube reward_values(n_path, n_action, n_pos);
  // Perform Backward induction
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << tt + 1 << "...";
    states = path.slice(tt);
    // Construct regression basis
    if (basis_type == "power") {
      reg_basis.cols(0, n_terms - 1) =
          PBasis(states, basis, intercept, n_terms, reccur_limit);
    } else if (basis_type == "laguerre") {
      reg_basis.cols(0, n_terms - 1) =
          LBasis(states, basis, intercept, n_terms, reccur_limit);
    }
    if (spline) {
      reg_basis.cols(n_terms, n_terms + n_knots - 1) =
          LSplineBasis(states, knots, n_knots, reccur_limit2);
    }
    if (n_rbasis > 0) {
      reg_basis.cols(n_terms + n_knots, n_terms + n_knots + n_rbasis - 1) =
          Rcpp::as<arma::mat>(Basis_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
    }
    // Compute the fitted continuation value
    if (useSVD) {
      for (std::size_t pp = 0; pp < n_pos; pp++) {
        temp_values = path_values.slice(tt + 1).col(pp);
        expected_value.slice(tt).col(pp) = SVDCoeff(reg_basis, temp_values);
      }
    } else {
      for (std::size_t pp = 0; pp < n_pos; pp++) {
        temp_values = path_values.slice(tt + 1).col(pp);
        expected_value.slice(tt).col(pp) =
            Rcpp::as<arma::vec>(Reg_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(reg_basis)),
                                     Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(temp_values)),
                                     tt + 1));
      }
    }
    reward_values = Rcpp::as<arma::cube>(
        Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states)), tt + 1));
    if (full_control) {
      Optimal(path_values, expected_value, reg_basis, reward_values,
              control, tt, n_path, n_pos, n_action, n_dim);
    } else {
      Optimal(path_values, expected_value, reg_basis, reward_values,
              control2, tt, n_path, n_pos, n_action, n_dim);
    }
  }
  Rcpp::Rcout << "end\n";
  return Rcpp::List::create(Rcpp::Named("value") = path_values,
                            Rcpp::Named("expected") = expected_value);
}
