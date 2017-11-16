// Copyright 2017 <jeremyyee@outlook.com.au>
// Least squares Monte Carlo
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"

// Regression coefficients from singular value decomposition
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
               const arma::mat& knots) {
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
  arma::mat states(n_path, n_dim);
  states = path.slice(n_dec - 1);
  path_values.slice(n_dec - 1) = Rcpp::as<arma::mat>(
      Scrap_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
  arma::cube expected_value(n_terms + n_knots, n_pos, n_dec - 1);  // fit
  arma::mat reg_basis(n_path, n_terms + n_knots);
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
    // Compute the fitted continuation value
    for (std::size_t pp = 0; pp < n_pos; pp++) {
      expected_value.slice(tt).col(pp) =
          SVDCoeff(reg_basis, path_values.slice(tt + 1).col(pp));
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


// Pricing Bermuda put using LSM (all paths)
//[[Rcpp::export]]
Rcpp::List BermudaPutLSM(const arma::cube& path,
                         const double& strike,
                         const double& discount,
                         const arma::umat& basis,
                         const bool& intercept,
                         const std::string& basis_type) {
  // Extract parameters
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  // Extract information about regression basis
  std::size_t n_terms = arma::accu(basis);  // Number of features in basis
  if (intercept) { n_terms++; }
  arma::uvec reccur(basis.n_rows);
  reccur = ReccurLimit(basis);
  // Perform the Bellman recursion starting at last time epoch
  arma::mat states = path.slice(n_dec - 1);
  arma::mat path_values(n_path, 2, arma::fill::zeros);  // (exercise, value)
  path_values.col(0) = strike - states.col(0);
  path_values.col(1) = arma::max(path_values, 1);
  arma::mat expected(n_terms, n_dec -1, arma::fill::zeros);  // Regression fit
  arma::mat reg_basis(n_path, n_terms);
  std::size_t n_money;  // number of in_money paths
  arma::uvec sorted(n_path);  // sort index
  arma::uvec cols_value = arma::linspace<arma::uvec>(0, 1, 2);  // col indices
  arma::uvec cols_path = arma::linspace<arma::uvec>(0, n_dim - 1, n_dim);
  arma::mat path_money(n_path, 2);  // Store in_money paths
  arma::mat value_money(n_path, 2);  // Store in_money values, (states, value)
  arma::vec fitted(n_path);  // Fitted continuation value
  // Perform Backward induction
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    path_values.col(1) = discount * path_values.col(1);  // Discount one step
    // Exercise value
    states = path.slice(tt);
    path_values.col(0) = strike - states.col(0);
    // Select paths that are in the money
    n_money = 0;
    for (std::size_t pp = 0; pp < n_path; pp++) {
      if (path_values(pp, 0) > 0) {
        n_money++;
      }
    }
    if (n_money == 0) {
      continue;
    }
    n_money--;  // Change to C++ indexing
    // Sort by sorted
    sorted = arma::sort_index(path_values.col(0), "descend");
    path_money.submat(0, 0, n_money, n_dim - 1) =
        states.submat(sorted.subvec(0, n_money), cols_path);
    value_money.submat(0, 0, n_money, 1) =
        path_values.submat(sorted.subvec(0, n_money), cols_value);
    // Regression basis
    if (basis_type == "power") {
      reg_basis.submat(0, 0, n_money, n_terms - 1) =
          PBasis(path_money.submat(0, 0, n_money, n_dim - 1), basis, intercept, n_terms, reccur);
    } else if (basis_type == "laguerre") {
      reg_basis.submat(0, 0, n_money, n_terms - 1) =
          LBasis(path_money.submat(0, 0, n_money, n_dim - 1), basis, intercept, n_terms, reccur);
    }
    expected.col(tt) = SVDCoeff(reg_basis.submat(0, 0, n_money, n_terms - 1),
                                value_money.col(1).subvec(0, n_money));
    // The fitted continuation value
    fitted.subvec(0, n_money) = reg_basis.rows(0, n_money) * expected.col(tt);
    for (std::size_t ww = 0; ww < n_money; ww++) {
      if ((strike - path_money(ww, 0)) > fitted(ww)) {
        path_values(sorted(ww), 1) = strike - path_money(ww, 0);
      }
    }
  }
  arma::vec value(n_path);
  value= path_values.col(1);
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("expected") = expected);
}

