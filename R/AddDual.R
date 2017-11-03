## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Additive duals
################################################################################

AddDual <- function(path, subsim_, expected, Reward_, Scrap_, control_, basis,
                    basis_type, spline = FALSE, knots = matrix(NA, nrow = 1)) {
    .Call('_rlsm_AddDual', PACKAGE = 'rlsm', path, subsim_,
          expected, Reward_, Scrap_, control_, basis,
          basis_type, spline, knots)
}
