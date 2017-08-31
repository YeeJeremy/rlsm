// Copyright 2017 <jeremyyee@outlook.com.au>
// Simulating stochastic processes
////////////////////////////////////////////////////////////////////////////////

#include "inst/include/random.h"

// Generate 1-dimensional Brownian motion with drift with unit time step
// d X_t = u dt + s dW_t, W_t is Wiener process
// [i, j, k]: state j for sample path i at time k
//[[Rcpp::export]]
arma::cube BM(const double& start,
              const double& mu,
              const double& vol,
              const int& n_dec,
              const int& n_path,
              const bool& antithetic) {
  // Standard Gaussian increments
  arma::mat increments(n_path, n_dec - 1);
  if (antithetic) {  // use anti-thetic disturbances
    arma::mat temp(0.5 * n_path, n_dec - 1, arma::fill::randn);
    increments = arma::join_vert(temp, -temp);
  } else {
    increments.randn();
  }
  increments = mu + vol * increments;
  // Simulating Brownian motion
  arma::cube path(n_path, 1, n_dec);
  path.slice(0).fill(start);
  for (std::size_t tt = 1; tt < n_dec; tt++) {
    path.slice(tt) = path.slice(tt - 1) + increments.col(tt - 1);
  }
  return path;
}

// Generate 1-dimensional geometric Brownian motion
// d X_t = u X_t dt + s X_t dW_t, W_t is Wiener process
// [i, j, k]: state j for sample path i at time k
//[[Rcpp::export]]
arma::cube GBM(const double& start,
               const double& mu,
               const double& vol,
               const int& n_dec,
               const int& n_path,
               const bool& antithetic) {
  arma::cube path(n_path, 1, n_dec);
  const double bm_start = 0;
  path = BM(bm_start, mu, vol, n_dec, n_path, antithetic);
  path = start * arma::exp(path);
  const double ito = -0.5 * vol * vol;
  for (std::size_t tt = 1 ; tt < n_dec; tt++) {
    path.slice(tt) = path.slice(tt) * std::exp(ito * tt);
  }
  return path;
}

// Generate correlated Brownian motion with drift
// [i, j, k]: state j for sample path i at time k
//[[Rcpp::export]]
arma::cube CBM(const arma::vec& start,
               const arma::vec& mu,
               const arma::vec& vol,
               const arma::mat& corr,
               const int& n_dec,
               const int& n_path,
               const bool& antithetic) {
  // Passing R objects to C++
  const std::size_t n_dim = start.n_elem;
  // Generating correlated increments
  arma::cube increments(n_path, n_dim, n_dec - 1);
  if (antithetic) {
    for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
      increments.slice(tt).rows(0, n_path/2 - 1) = CorrNormal(n_path / 2, corr);
    }
    increments.tube(arma::span(n_path / 2, n_path - 1), arma::span::all) =
        -increments.tube(arma::span(0, n_path / 2 - 1), arma::span::all);
  } else {
    for (std::size_t tt = 0; tt < n_dec - 1; tt++) {
      increments.slice(tt) = CorrNormal(n_path, corr);
    }
  }
  // Simulating Brownian motion
  arma::cube path(n_path, n_dim, n_dec);
  for (std::size_t dd = 0; dd < n_dim; dd++) {
    increments.tube(arma::span::all, arma::span(dd)) = mu(dd) +
        vol(dd) * increments.tube(arma::span::all, arma::span(dd));
    path.slice(0).col(dd).fill(start(dd));
  }
  for (std::size_t tt = 1; tt < n_dec; tt++) {
    path.slice(tt) = path.slice(tt - 1) + increments.slice(tt - 1);
  }
  return path;
}

// Generate correlated geometric Brownian motion
// [i, j, k]: state j for sample path i at time k
//[[Rcpp::export]]
arma::cube CGBM(const arma::vec& start,
                const arma::vec& mu,
                const arma::vec& vol,
                const arma::mat& corr,
                const int& n_dec,
                const int& n_path,
                const bool& antithetic) {
  // Passing R objects to C++
  const std::size_t n_dim = start.n_elem;
  // Generate paths
  arma::cube path(n_path, n_dim, n_dec);
  arma::vec zerostart(n_dim, arma::fill::zeros);
  path = CBM(zerostart, mu, vol, corr, n_dec, n_path, antithetic);
  path = arma::exp(path);
  arma::vec ito = -0.5 * vol % vol;
  for (std::size_t dd = 0; dd < n_dim; dd++) {
    path.slice(0).col(dd).fill(start(dd));
    for (std::size_t tt = 1 ; tt < n_dec; tt++) {
      path.slice(tt).col(dd) =
          start(dd) * std::exp(ito(dd) * tt) * path.slice(tt).col(dd);
    }
  }
  return path;
}

