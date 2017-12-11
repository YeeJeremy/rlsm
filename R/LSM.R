## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Least squares Monte Carlo
################################################################################

LSM <- function(path, Reward_, Scrap_, control_, basis, intercept, basis_type,
                spline = FALSE, knots = matrix(NA, nrow = 1),
                Basis_ = function(){}, n_rbasis = 0, Reg_) {
    if (missing(Reg_)) {
        Reg_ <- function(){}
        .Call('_rlsm_LSM', PACKAGE = 'rlsm', path, Reward_, Scrap_,
              control_, basis, intercept, basis_type, spline, knots, Basis_, n_rbasis,
              Reg_, TRUE)
    } else {
        .Call('_rlsm_LSM', PACKAGE = 'rlsm', path, Reward_, Scrap_,
              control_, basis, intercept, basis_type, spline, knots, Basis_, n_rbasis,
              Reg_, FALSE)
    }
}
