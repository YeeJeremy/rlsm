\name{GetBounds}
\alias{GetBounds}
\title{Confidence Bounds}
\description{Confidence bounds for the value.}
\usage{
GetBounds(duality, alpha, position)
}
\arguments{
  \item{duality}{Object returned by the Bounds function.}
  \item{alpha}{Specifies the (1-alpha) confidence bounds.}
  \item{position}{Natural number indicating the starting position.}
}
\value{
  Array representing the (1-alpha) confidence bounds for the value
  of the specified position.
}
\examples{
## Bermuda put option
step <- 0.02
mu <- 0.06 * step
vol <- 0.2 * sqrt(step)
n_dec <- 51
start <- 36
strike <- 40
## LSM
n_path <- 1000
path <- GBM(start, mu, vol, n_dec, n_path, TRUE)
control <- matrix(c(c(1, 1), c(2, 1)), nrow = 2, byrow = TRUE)
basis <- matrix(c(1, 1), nrow = 1)
knots <- matrix(c(30, 40, 50), nrow = 1)
Scrap <- function(state) {
    output <- matrix(data = 0, nrow = nrow(state), ncol = 2)
    output[, 2] <- exp(-mu * (n_dec - 1)) * pmax(strike - state, 0)
    return(output)
}
Reward <- function(state, time) {
    output <- array(data = 0, dim = c(nrow(state), 2, 2))
    output[, 2, 2] <- exp(-mu * (time - 1)) * pmax(strike - state, 0)
    return(output)
}
lsm <- LSM(path, Reward, Scrap, control, basis, TRUE, "power", TRUE, knots)
n_path2 <- 100
path2 <- GBM(start, mu, vol, n_dec, n_path2, TRUE)
policy <- PathPolicy(path2, lsm$expected, Reward, control, basis,
"power", TRUE, knots)
n_subsim <- 100
subsim <- NestedGBM(path2, mu, vol, n_subsim, TRUE)
mart <- AddDual(path2, subsim, lsm$expected, Reward, Scrap, control,
basis, "power", TRUE, knots)
bounds <- Bounds(path2, Reward, Scrap, control, mart, policy)
confidenceInterval <-GetBounds(bounds, 0.05, 2)
}
\author{Jeremy Yee}

