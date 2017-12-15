## Copyright 2017 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Least squares Monte Carlo
################################################################################

LSM <- function(path, Reward, Scrap, control, basis = matrix(c(1), nrow = 1),
                intercept = TRUE, basis_type = "power", spline = FALSE,
                knots = matrix(NA, nrow = 1), Basis = function(){}, n_rbasis = 0,
                Reg) {
    if (missing(Reg)) {
        Reg <- function(){}
        .Call('_rlsm_LSM', PACKAGE = 'rlsm', path, Reward, Scrap,
              control, basis, intercept, basis_type, spline, knots, Basis, n_rbasis,
              Reg, TRUE)
    } else {
        .Call('_rlsm_LSM', PACKAGE = 'rlsm', path, Reward, Scrap,
              control, basis, intercept, basis_type, spline, knots, Basis, n_rbasis,
              Reg, FALSE)
    }
}
