## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Least squares Monte Carlo
################################################################################

LSM <- function(path, Reward_, Scrap_, control_, basis, intercept, basis_type,
                spline = FALSE, knots = matrix(NA, nrow = 1)) {
    .Call('_rlsm_LSM', PACKAGE = 'rlsm', path, Reward_, Scrap_,
          control_, basis, intercept, basis_type, spline, knots)
}
