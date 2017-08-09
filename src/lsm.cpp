// Copyright 2017 <jeremyyee@outlook.com.au>
// Least squares Monte Carlo
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <string>

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

// Regression coefficients from singular value decomposition
//[[Rcpp::export]]
arma::colvec SVDCoeff(const arma::mat& xreg,
                      const arma::vec& yreg) {
  arma::mat u;
  arma::vec s;
  arma::mat v;
  arma::svd_econ(u, s, v, xreg, "both", "dc");
  arma::colvec d = (u.t()) * yreg;
  arma::colvec temp(xreg.n_cols, arma::fill::zeros);
  int r = rank(xreg);
  temp(arma::span(0, r - 1)) =
      d(arma::span(0, r - 1)) / s(arma::span(0, r - 1));
  arma::colvec svd_coeff = v * temp;
  return svd_coeff;
}

// Best action given regression basis (position control deterministic)
arma::mat Optimal(const arma::mat& expected_value,
                  const arma::mat& reg_basis,
                  const arma::cube& reward_values,
                  const arma::imat& control,
                  const std::size_t& n_path,
                  const std::size_t& n_pos,
                  const std::size_t& n_action,
                  const std::size_t& n_dim) {
  // Get fitted values
  arma::mat fitted_expected(n_path, n_pos);
  fitted_expected = reg_basis * expected_value;  // row = state, col = position
  // Compute the fitted values based on position and action
  arma::mat best(n_path, n_pos);  // the best values
  arma::mat compare(n_path, n_action);
  std::size_t pp, aa, nn;
  for (pp = 0; pp < n_pos; pp++) {
    compare = reward_values.slice(pp);
    for (aa = 0; aa < n_action; aa++) {
      nn = control(pp, aa);  // Evolution of position
      compare.col(aa) += fitted_expected.col(nn);
    }
    best.col(pp) = arma::max(compare, 1);  // Choose best action
  }
  return best;
}

// Best action given regression basis (position control not deterministic)
arma::mat Optimal(const arma::mat& expected_value,
                  const arma::mat& reg_basis,
                  const arma::cube& reward_values,
                  const arma::cube& control,
                  const std::size_t& n_path,
                  const std::size_t& n_pos,
                  const std::size_t& n_action,
                  const std::size_t& n_dim) {
  // Get fitted values
  arma::mat fitted_expected(n_path, n_pos);
  fitted_expected = reg_basis * expected_value;  // row = state, col = position
  // Compute the fitted values based on position and action
  arma::mat best(n_path, n_pos);  // The best values
  arma::mat compare(n_path, n_action);
  arma::vec trans_prob(n_pos);  // The transition probabilities
  std::size_t pp, aa, nn;
  for (pp = 0; pp < n_pos; pp++) {
    compare = reward_values.slice(pp);
    for (aa = 0; aa < n_action; aa++) {
      trans_prob = control.tube(pp, aa);
      compare.col(aa) += fitted_expected * trans_prob;
    }
    best.col(pp) = arma::max(compare, 1);  // Choose best action
  }
  return best;
}

// Least squares Monte Carlo
//[[Rcpp::export]]
Rcpp::List LSM(const arma::cube& path,
               const Rcpp::Function& Reward_,
               const Rcpp::Function& Scrap_,
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
  // Perform the Bellman recursion starting at last time epoch
  Rcpp::Rcout << "At dec: " << n_dec - 1 << "...";
  arma::mat path_values(n_path, n_pos);
  arma::mat states(n_path, n_dim);
  states = path(arma::span(n_dec - 1), arma::span::all, arma::span::all);
  path_values = Rcpp::as<arma::mat>(
      Scrap_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
  arma::cube expected_value(n_terms, n_pos, n_dec - 1);  // Regression fit
  arma::mat reg_basis(n_path, n_terms);
  arma::cube reward_values(n_path, n_action, n_pos);
  // Perform Backward induction
  for (int tt = (n_dec - 2); tt >= 0; tt--) {
    Rcpp::Rcout << n_dec - 1 << "...";
    states = path(arma::span(tt), arma::span::all, arma::span::all);
    // Compute the fitted continuation value
    if (basis_type == "power") {
      reg_basis = PBasis(states, basis, intercept, n_terms);
    }
    for (std::size_t pp = 1; pp < n_pos; pp++) {
      expected_value.slice(tt).col(pp) =
          SVDCoeff(reg_basis, path_values.col(pp));
    }
    reward_values = Rcpp::as<arma::cube>(
        Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states))));
    if (full_control) {
      path_values = Optimal(expected_value.slice(tt), reg_basis, reward_values,
                            control, n_path, n_pos, n_action, n_dim);
    } else {
     path_values = Optimal(expected_value.slice(tt), reg_basis, reward_values,
                           control2, n_path, n_pos, n_action, n_dim);     
    }
  }
  Rcpp::Rcout << "end\n";
  return Rcpp::List::create(Rcpp::Named("value") = path_values,
                            Rcpp::Named("expected") = expected_value);
}

