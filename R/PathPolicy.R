## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Path policy
################################################################################

PathPolicy <- function(path, expected, Reward_, control_, basis, basis_type,
                       spline = FALSE, knots = matrix(NA, nrow = 1),
                       Basis_ = function(){}, n_rbasis = 0) {
    .Call('_rlsm_PathPolicy', PACKAGE = 'rlsm', path, expected,
          Reward_, control_, basis, basis_type, spline, knots, Basis_, n_rbasis)
}
