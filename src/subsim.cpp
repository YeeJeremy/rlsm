// Copyright 2017 <jeremyyee@outlook.com.au>
// Nested simulation along a path for duality bounds
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/random.h"

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

// Nested simulation of correlated Brownian motion
//[[Rcpp::export]]
arma::cube NestedCBM(const arma::cube& path,
                     const arma::vec& mu,
                     const arma::vec& vol,
                     const arma::mat& corr,
                     const int& n_subsim,
                     const bool& antithetic) {
  // Extract parameters
  const std::size_t n_dec = path.n_rows;
  const std::size_t n_path = path.n_cols;
  const std::size_t n_dim = path.n_slices;
  // Perfrom the subsimulation
  // Store output as cube because field<cube> less efficient
  // Slices 1:(n_dec - 1) is for the first dim, and so on.
  // Will transform this to 4-d array in R
  arma::cube subsim(n_subsim, n_path, (n_dec - 1) * n_dim);
  arma::cube increments(n_subsim, n_path, n_dim);
  arma::mat states(1, n_path);
  for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
    // Generate the correlated increments
    if (antithetic) {
      for (std::size_t ss = 0; ss < (n_subsim / 2); ss++) {
        increments.tube(arma::span(ss), arma::span::all) = CorrNormal(n_path, corr);
      }
      increments.tube(arma::span(n_subsim / 2, n_subsim - 1), arma::span::all) =
          -increments.tube(arma::span(0, n_subsim / 2 - 1), arma::span::all);
    } else {
      for (std::size_t ss = 0; ss < n_subsim; ss++) {
        increments.tube(arma::span(ss), arma::span::all) = CorrNormal(n_path, corr);
      }
    }
    for (std::size_t dd = 0; dd < n_dim; dd++) {
      increments.slice(dd) = mu(dd) + vol(dd) * increments.slice(dd);
      states = path.slice(dd).row(tt);
      subsim.slice((n_dec - 1) * dd + tt) =
          arma::repmat(states, n_subsim, 1) + increments.slice(dd);
    }
  }
  return subsim;
}

// Nested simulation for correlated geometric Brownian motion
//[[Rcpp::export]]
arma::cube NestedCGBM(const arma::cube& path,
                      const arma::vec& mu,
                      const arma::vec& vol,
                      const arma::mat& corr,
                      const int& n_subsim,
                      const bool& antithetic) {
  // Extract parameters
  const std::size_t n_dec = path.n_rows;
  const std::size_t n_path = path.n_cols;
  const std::size_t n_dim = path.n_slices;
  // Store output as cube because field<cube> less efficient
  // Slices 1:(n_dec - 1) is for the first dim, and so on.
  // Will transform this to 4-d array in R
  arma::cube subsim(n_subsim, n_path, (n_dec - 1) * n_dim);
  // Perfrom the subsimulation
  arma::cube pathCBM(n_dec, n_path, n_dim, arma::fill::zeros);
  subsim = NestedCBM(pathCBM, mu, vol, corr, n_subsim, antithetic);
  arma::mat states(1, n_path);
  const arma::vec ito = arma::exp(-0.5 * vol % vol);
  for (std::size_t dd = 0; dd < n_dim; dd++) {
    for (std::size_t tt =0; tt < n_dec - 1; tt++) {
      states = path.slice(dd).row(tt);
      subsim.slice((n_dec - 1) * dd + tt) =
          ito(dd) * arma::repmat(states, n_subsim, 1) %
          arma::exp(subsim.slice((n_dec - 1) * dd + tt));
    }
  }
  return subsim;
}
