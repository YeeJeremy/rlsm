// Copyright 2017 <jeremyyee@outlook.com.au>
// Compute the additive duals using our approximations
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/basis.h"

// Compute the additive duals
//[[Rcpp::export]]
arma::cube AddDual2(const arma::cube& path,
                    Rcpp::NumericVector subsim_,
                    const arma::cube& expected_fitted,
                    const Rcpp::Function& Reward_,
                    const Rcpp::Function& Scrap_,
                    Rcpp::NumericVector control_,
                    const arma::umat& basis,
                    const std::string& basis_type) {
  // Extract parameters
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  const std::size_t n_terms = expected_fitted.n_rows;
  bool intercept = true;
  if (n_terms == arma::accu(basis)) {
    intercept = false;
  }
  arma::uvec reccur_limit(basis.n_rows);
  reccur_limit = ReccurLimit(basis);
  const arma::ivec s_dims = subsim_.attr("dim");
  const std::size_t n_subsim = s_dims(0);
  arma::cube subsim(subsim_.begin(), n_subsim, n_dim, n_path * (n_dec - 1), false);
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
  // Additive duals
  arma::cube add_dual(n_path, n_pos, n_dec - 1, arma::fill::zeros);
  arma::mat path_basis(n_path, n_terms);
  arma::mat subsim_basis(n_subsim, n_terms);
  arma::cube reward_value(n_path, n_action, n_pos);
  arma::cube subsim_reward(n_subsim, n_action, n_pos);
  arma::mat expected(n_path, n_pos);
  arma::mat subsim_expected(n_subsim, n_pos);
  arma::mat states(n_subsim, n_path);
  arma::mat compare(n_path, n_action);
  arma::mat subsim_compare(n_subsim, n_action);
  std::size_t tt, pos, aa, nn, pp;
  arma::uword best_action;
  arma::vec trans_prob(n_pos);  // The transition probabilities
  arma::vec temp(n_subsim);
  for (tt = 0; tt < n_dec - 2; tt++) {
    // Find the average of the subsimulation paths
    for (pp = 0; pp < n_path; pp++) {
      states = subsim.slice(n_path * tt + pp);
      if (basis_type == "power") {
        subsim_basis = PBasis(states, basis, intercept, n_terms, reccur_limit);
      } else if (basis_type == "laguerre") {
        subsim_basis = LBasis(states, basis, intercept, n_terms, reccur_limit);
      }
      // Fitted expected value function for subsims
      subsim_expected = subsim_basis * expected_fitted.slice(tt + 1);
      // Reward for subsims
      subsim_reward = Rcpp::as<arma::cube>(
          Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(states)), tt + 2));
      if (full_control) {
        for (pos = 0; pos < n_pos; pos++) {
          subsim_compare = subsim_reward.slice(pos);
          for (aa = 0; aa < n_action; aa++) {
            nn = control(pos, aa) - 1;  // Next position + R index starts at 1
            subsim_compare.col(aa) += subsim_expected.col(nn);
          }
          temp = arma::max(subsim_compare, 1);
          add_dual(pp, pos, tt) += arma::sum(temp);
        }
      } else {
        for (pos = 0; pos < n_pos; pos++) {
          subsim_compare = subsim_reward.slice(pos);
          for (aa = 0; aa < n_action; aa++) {
            trans_prob = control2.tube(pos, aa);
            subsim_compare.col(aa) += subsim_expected * trans_prob;
          }
          temp = arma::max(subsim_compare, 1);
          add_dual(pp, pos, tt) += arma::sum(temp);
        }
      }
    }
    add_dual.slice(tt) = (1.0 / n_subsim) * add_dual.slice(tt);
    // Find the realised values. Reg basis for paths.
    if (basis_type == "power") {
      path_basis = PBasis(path.slice(tt + 1), basis, intercept, n_terms, reccur_limit);
    } else if (basis_type == "laguerre") {
      path_basis = LBasis(path.slice(tt + 1), basis, intercept, n_terms, reccur_limit);
    }
    // Fitted expected value function for subsims
    expected = path_basis * expected_fitted.slice(tt + 1);
    // Reward for subsims
    reward_value = Rcpp::as<arma::cube>(
        Reward_(Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(path.slice(tt + 1))),
                tt + 2));
    if (full_control) {
      for (pos = 0; pos < n_pos; pos++) {
        compare = reward_value.slice(pos);
        for (aa = 0; aa < n_action; aa++) {
          nn = control(pos, aa) - 1;  // Next position + R index starts at 1
          compare.col(aa) += expected.col(nn);
        }
        add_dual.slice(tt).col(pos) -= arma::max(compare, 1);
      }
    } else {
      for (pos = 0; pos < n_pos; pos++) {
        compare = reward_value.slice(pos);
        for (aa = 0; aa < n_action; aa++) {
          trans_prob = control2.tube(pos, aa);
          compare.col(aa) += expected * trans_prob;
        }
        add_dual.slice(tt).col(pos) -= arma::max(compare, 1);
      }
    }
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
