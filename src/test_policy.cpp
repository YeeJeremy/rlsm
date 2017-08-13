// Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
// Testing the prescribed policy
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>

// Determine the next position for partial controlled positions
arma::uword NextPosition(const arma::vec &prob_weight) {
  const std::size_t n_pos = prob_weight.n_elem;
  const arma::vec cum_prob = arma::cumsum(prob_weight);
  const double rand_unif = R::runif(0, 1);
  arma::uword next_state = 0;
  for (std::size_t i = 1; i < n_pos; i++) {
    if (rand_unif <= cum_prob(i)) {
      next_state = i;
      break;
    }
  }
  return next_state;
}

// Test the prescribed policy on a set of sample paths
//[[Rcpp::export]]
arma::vec TestPolicy(const int& start_position,
                     Rcpp::NumericVector path_,
                     Rcpp::NumericVector control_,
                     Rcpp::Function Reward_,
                     Rcpp::Function Scrap_,
                     const arma::ucube& path_action) {
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
  // Testing the prescribed policy
  arma::vec value(n_path, arma::fill::zeros);
  arma::uvec pos(n_path);
  pos.fill(start_position - 1);  // Initialise with starting position
  arma::mat state(n_path, n_dim);
  arma::cube reward(n_path, n_action, n_pos);
  arma::cube scrap(n_path, n_pos);
  arma::uword policy;
  if (full_control) {
    for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
      state = path(arma::span(tt), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), tt + 1));
      for (std::size_t ww = 0; ww < n_path; ww++) {
        policy = path_action(tt, pos(ww), ww) - 1;
        value(ww) += reward(ww, policy, pos(ww));
        pos(ww) = control(pos(ww), policy) - 1;
      }
    }
  } else {
    arma::vec prob_weight(n_pos);
    for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
      state = path(arma::span(tt), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), tt + 1));
      for (std::size_t ww = 0; ww < n_path; ww++) {
        policy = path_action(t, pos(ww), ww) - 1;
        value(ww) += reward(ww, policy, pos(ww));
        prob_weight = control2.tube(pos(ww), policy);
        pos(ww) = NextPosition(prob_weight);
      }
    }    
  }
  // Get the scrap value
  state = path(arma::span(n_dec - 1), arma::span::all, arma::span::all);
  scrap = Rcpp::as<arma::mat>(Scrap_(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state))));
  for (std::size_t ww = 0; ww < n_path; ww++) {
    value(ww) += scrap(ww, policy, pos(ww));
  }
  return value;
}

// Test the prescribed policy on a set of sample paths (return more info)
//[[Rcpp::export]]
Rcpp::List TestPolicy2(const int& start_position,
                       Rcpp::NumericVector path_,
                       Rcpp::NumericVector control_,
                       Rcpp::Function Reward_,
                       Rcpp::Function Scrap_,
                       const arma::ucube& path_action) {
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
  // Testing the prescribed policy
  arma::vec value(n_path, arma::fill::zeros);
  arma::umat action(n_path, n_dec - 1);  // Actions taken
  arma::umat position(n_path, n_dec);  // Evolution of the position
  arma::uvec pos(n_path);
  pos.fill(start_position - 1);  // Initialise with starting position
  position.col(0) = pos;
  arma::mat state(n_path, n_dim);
  arma::cube reward(n_path, n_action, n_pos);
  arma::cube scrap(n_path, n_pos);
  arma::uword policy;
  if (full_control) {
    for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
      state = path(arma::span(tt), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), tt + 1));
      for (std::size_t ww = 0; ww < n_path; ww++) {
        policy = path_action(tt, pos(ww), ww) - 1;
        action(ww, tt) = policy;
        value(ww) += reward(ww, policy, pos(ww));
        pos(ww) = control(pos(ww), policy) - 1;
      }
      positon.col(tt + 1) = pos;
    }
  } else {
    arma::vec prob_weight(n_pos);
    for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
      state = path(arma::span(tt), arma::span::all, arma::span::all);
      reward = Rcpp::as<arma::cube>(Reward_(
          Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state)), tt + 1));
      for (std::size_t ww = 0; ww < n_path; ww++) {
        policy = path_action(t, pos(ww), ww) - 1;
        action(ww, tt) = policy;
        value(ww) += reward(ww, policy, pos(ww));
        prob_weight = control2.tube(pos(ww), policy);
        pos(ww) = NextPosition(prob_weight);
      }
      positon.col(tt + 1) = pos;
    }
  }
  // Get the scrap value
  state = path(arma::span(n_dec - 1), arma::span::all, arma::span::all);
  scrap = Rcpp::as<arma::mat>(Scrap_(
      Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(state))));
  for (std::size_t ww = 0; ww < n_path; ww++) {
    value(ww) += scrap(ww, policy, pos(ww));
  }
  return Rcpp::List::create(Rcpp::Named("value") = value,
                            Rcpp::Named("position") = position + 1,
                            Rcpp::Named("action") = action + 1);
}
