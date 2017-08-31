// Copyright 2017 <jeremyyee@outlook.com.au>
// Nested simulation along a path for duality bounds
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>

// Simulate one step 1-d Brownian motion from each path node
//[[Rcpp::export]]
arma::cube NestedBM(const arma::cube& path,
                    const double& mu,
                    const double& vol,
                    const int& n_subsim,
                    const bool& antithetic) {
  // Extract parameters
  const std::size_t n_dec = path.n_rows;
  const std::size_t n_path = path.n_cols;
  const std::size_t n_dim = path.n_slices;
  // The subsimulation increments
  arma::mat increments(n_subsim, n_path);
  arma::mat temp(0.5 * n_subsim, n_path);
  // Perfrom the subsimulation
  arma::mat states(1, n_path);
  arma::cube subsim(n_subsim, n_path, n_dec - 1);
  for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
    if (antithetic) {  // use anti-thetic disturbances
      temp.randn();
      increments = arma::join_vert(temp, -temp);
    } else {
      increments.randn();
    }
    increments = mu + vol * increments;
    states = path.slice(0).row(tt);
    subsim.slice(tt) = arma::repmat(states, n_subsim, 1) + increments;
  }
  return subsim;
}

// Simulate one step 1-d geometric Brownian motion from path nodes
//[[Rcpp::export]]
arma::cube NestedGBM(const arma::cube& path,
                     const double& mu,
                     const double& vol,
                     const int& n_subsim,
                     const bool& antithetic) {
  // Extract parameters
  const std::size_t n_dec = path.n_rows;
  const std::size_t n_path = path.n_cols;
  const std::size_t n_dim = path.n_slices;
  // Perfrom the subsimulation
  arma::cube subsim(n_subsim, n_path, n_dec - 1);
  arma::cube pathBM(n_dec, n_path, 1, arma::fill::zeros);
  subsim = NestedBM(pathBM, mu, vol, n_subsim, antithetic);
  arma::mat states(1, n_path);
  const double ito = std::exp(-0.5 * vol * vol);
  for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
    states = path.slice(0).row(tt);
    subsim.slice(tt) = ito * arma::repmat(states, n_subsim, 1) %
        arma::exp(subsim.slice(tt));
  }
  return subsim;
}

