// Copyright 2017 <jeremyyee@outlook.com.au>
// Simulating stochastic processes
////////////////////////////////////////////////////////////////////////////////

#include <RcppArmadillo.h>
#include <Rcpp.h>

// Generate 1-dimensional Brownian motion with drift with unit time step
// d X_t = u dt + s dW_t, W_t is Wiener process
// [i,j]: state at time i for sample path j
//[[Rcpp::export]]
arma::mat BM(const double& start,
             const double& mu,
             const double& vol,
             const int& n_dec,
             const int& n_path,
             const bool& antithetic) {
  // Standard Gaussian increments
  arma::mat increments(n_dec - 1, n_path);
  if (antithetic) {  // use anti-thetic disturbances
    arma::mat temp(n_dec - 1, 0.5 * n_path, arma::fill::randn);
    increments = arma::join_horiz(temp, -temp);
  } else {
    increments = arma::randn(n_dec - 1, n_path);
  }
  increments = mu + vol * increments;
  // Simulating Brownian motion
  arma::mat path(n_dec, n_path);
  path.row(0).fill(start);
  for (int tt = 1; tt < n_dec; tt++) {
    path.row(tt) = path.row(tt - 1) + increments.row(tt - 1);
  }
  return path;
}

// Generate 1-dimensional geometric Brownian motion
// d X_t = u X_t dt + s X_t dW_t, W_t is Wiener process
// [i,j]: state at time i for sample path j
//[[Rcpp::export]]
arma::mat GBM(const double& start,
              const double& mu,
              const double& vol,
              const int& n_dec,
              const int& n_path,
              const bool& antithetic) {
  arma::mat path(n_dec, n_path);
  double bm_start = 0;
  path = BM(bm_start, mu, vol, n_dec, n_path, antithetic);
  path = start * arma::exp(path);
  double ito = -0.5 * vol * vol;
  for (int tt = 1 ; tt < n_dec; tt++) {
    path.row(tt) = path.row(tt) * std::exp(ito * tt);
  }
  return path;
}

// Generate correlated Gaussians using cholesky decomposition
//[[Rcpp::export]]
arma::mat CorrNormal(const int& n,
                     const arma::mat& corr) {
  int ncols = corr.n_cols;
  arma::mat Z = arma::randn(n, ncols);
  return Z * arma::chol(corr);
}

// Generate correlated Brownian motion with drift
// [i,j,k]: state k at time i for sample path j
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
  arma::cube increments(n_dec - 1, n_path, n_dim);
  if (antithetic) {
    for (int pp = 0; pp < (n_path / 2); pp++) {
      increments.tube(arma::span::all, arma::span(pp)) = CorrNormal(n_dec - 1, corr);
    }
    increments.tube(arma::span::all, arma::span(n_path / 2, n_path - 1)) =
        -increments.tube(arma::span::all, arma::span(0, n_path / 2 - 1));
  } else {
    for (int pp = 0; pp < n_path; pp++) {
      increments.tube(arma::span::all, arma::span(pp)) = CorrNormal(n_dec - 1, corr);
    }
  }
  // Simulating Brownian motion
  arma::cube path(n_dec, n_path, n_dim);
  for (int dd = 0; dd < n_dim; dd++) {
    increments.slice(dd) = mu(dd) + vol(dd) * increments.slice(dd);
    path.slice(dd).row(0).fill(start(dd));
  }
  for (int tt = 1; tt < n_dec; tt++) {
    path.tube(arma::span(tt), arma::span::all) =
        path.tube(arma::span(tt - 1), arma::span::all) +
        increments.tube(arma::span(tt - 1), arma::span::all);
  }
  return path;
}

// Generate correlated geometric Brownian motion
// [i,j,k]: state k at time i for sample path j
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
  arma::cube path(n_dec, n_path, n_dim);
  arma::vec zerostart(n_dim, arma::fill::zeros);
  path = CBM(zerostart, mu, vol, corr, n_dec, n_path, antithetic);
  arma::vec ito = -0.5 * vol % vol;
  for (int dd = 0; dd < n_dim; dd++) {
    path.slice(dd) = start(dd) * arma::exp(path.slice(dd));
    for (int tt = 1 ; tt < n_dec; tt++) {
      path.slice(dd).row(tt) = std::exp(ito(dd) * tt) * path.slice(dd).row(tt);
    }
  }
  return path;
}

