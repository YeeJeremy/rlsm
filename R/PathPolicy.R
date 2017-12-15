## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Path policy
################################################################################

PathPolicy <- function(path, expected, Reward, control,
                       basis = matrix(c(1), nrow = 1),
                       basis_type = "power", spline = FALSE,
                       knots = matrix(NA, nrow = 1), Basis = function(){},
                       n_rbasis = 0) {
    .Call('_rlsm_PathPolicy', PACKAGE = 'rlsm', path, expected,
          Reward, control, basis, basis_type, spline, knots, Basis, n_rbasis)
}
