// Copyright 2017 <jeremyyee@outlook.com.au>
// Nested simulation along a path for duality approach
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
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  // The subsimulation increments
  arma::mat increments(n_subsim, n_path);
  // Perfrom the subsimulation
  arma::mat states(n_path, 1);
  arma::cube subsim(n_subsim, n_path, n_dec - 1);
  for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
    if (antithetic) {  // use anti-thetic disturbances
      arma::mat temp(0.5 * n_subsim, n_path);
      temp.randn();
      increments = arma::join_vert(temp, -temp);
    } else {
      increments.randn();
    }
    increments = mu + vol * increments;
    states = path.slice(tt);
    subsim.slice(tt) = arma::repmat(states.t(), n_subsim, 1) + increments;
  }
  // Convert this to 4-D array in R
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
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  // Perfrom the subsimulation
  arma::cube subsim(n_subsim, n_path, n_dec - 1);
  arma::cube pathBM(n_path, 1, n_dec, arma::fill::zeros);
  subsim = NestedBM(pathBM, mu, vol, n_subsim, antithetic);
  arma::mat states(n_path, 1);
  const double ito = std::exp(-0.5 * vol * vol);
  for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
    states = path.slice(tt);
    subsim.slice(tt) = ito * arma::repmat(arma::trans(states), n_subsim, 1) %
        arma::exp(subsim.slice(tt));
  }
  // Convert this to 4-D array in R
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
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  // Perfrom the subsimulation
  // Store output as cube because field<cube> less efficient
  // Slices 0:(n_path - 1) is for the first time, and so on.
  // Will transform this to 4-d array in R
  arma::cube subsim(n_subsim, n_dim, n_path * (n_dec - 1));
  arma::cube increments(n_subsim, n_dim, n_path);
  arma::mat states(n_path, n_dim);
  for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
    // Generate the correlated increments for each sample path
    if (antithetic) {
      for (std::size_t pp = 0; pp < n_path; pp++) {
        increments.slice(pp).rows(0, n_subsim / 2 - 1) =
            CorrNormal(n_subsim / 2, corr);
      }
      increments.tube(arma::span(n_subsim / 2, n_subsim - 1), arma::span::all) =
          -increments.tube(arma::span(0, n_subsim / 2 - 1), arma::span::all);
    } else {
      for (std::size_t pp = 0; pp < n_path; pp++) {
        increments.slice(pp) = CorrNormal(n_subsim, corr);
      }
    }
    // Add drift and volitility to the increments
    for (std::size_t dd = 0; dd < n_dim; dd++) {
      subsim(arma::span::all, arma::span(dd),
             arma::span(n_path * tt, n_path * (tt + 1) - 1)) =
          mu(dd) + vol(dd) * increments.tube(arma::span::all, arma::span(dd));
    }
    // Add the increments to the path nodes
    states = path.slice(tt);
    for (std::size_t pp = 0; pp < n_path; pp++) {
      subsim.slice(n_path * tt + pp) = arma::repmat(states.row(pp), n_subsim, 1)
          + subsim.slice(n_path * tt + pp);
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
  const std::size_t n_dec = path.n_slices;
  const std::size_t n_path = path.n_rows;
  const std::size_t n_dim = path.n_cols;
  // Store output as cube because field<cube> less efficient
  // Slices 0:(n_path - 1) is for the first time, and so on.
  // Will transform this to 4-d array in R
  arma::cube subsim(n_subsim, n_dim, n_path * (n_dec - 1));
  // Perfrom the subsimulation
  arma::cube pathCBM(n_path, n_dim, n_dec, arma::fill::zeros);
  subsim = NestedCBM(pathCBM, mu, vol, corr, n_subsim, antithetic);
  arma::mat states(n_path, n_dim);
  const arma::vec ito = arma::exp(-0.5 * vol % vol);
  const arma::mat itoMat = arma::repmat(ito.t(), n_subsim, 1);
  for (std::size_t tt =0; tt < n_dec - 1; tt++) {
    states = path.slice(tt);
    for (std::size_t pp = 0; pp < n_path; pp++) {
      subsim.slice(n_path * tt + pp) = arma::repmat(states.row(pp), n_subsim, 1)
          % arma::exp(subsim.slice(n_path * tt + pp)) % itoMat;
    }
  }
  return subsim;
}
