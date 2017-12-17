\name{PathPolicy}
\alias{PathPolicy}
\title{PathPolicy}
\description{Obtaining the prescribed policy for sample paths}
\usage{
PathPolicy(path, expected, Reward, control, basis = matrix(c(1), nrow =
1), basis_type = "power", spline = FALSE, knots = matrix(NA, nrow = 1),
Basis = function(){}, n_rbasis = 0)
}
\arguments{
  \item{path}{3-D array representing sample paths. Entry [i,j,k]
    represents the j-th component of the state at time k for sample path
    i.}
  \item{expected}{3-D array representing the fitted coefficients for the
    continuation value function. Array [,i,j] gives the fit for position
    i at time j.}
  \item{Reward}{User supplied function to represent the reward function.
    The function should take in the following arguments, in this order:
    \itemize{
      \item{\eqn{n \times d} matrix representing the \eqn{n}
	\eqn{d}-dimensional states. }
      \item{A natural number representing the decison time.}
    }
    The function should output the following:
    \itemize{
      \item{3-D array with dimensions \eqn{n \times (a \times p)}
	representing the rewards where \eqn{n} is the number of sample
	paths, \eqn{a} is the number of action, and \eqn{p} is the
	number of positions. The \eqn{[i, j, k]}-th entry corresponds to
	the reward from applying the \eqn{j}-th action to the \eqn{k}-th
	position for the \eqn{i}-th state.}
    }
    }
  \item{control}{Array representing the transition probabilities of the
    controlled Markov chain. Two possible inputs:
    \itemize{
      \item{Matrix of dimension \eqn{p \times a} where entry
        [i,j] describes the next position after selecting action j at
        position i.}
      \item{3-D array with dimensions \eqn{p \times a \times p} where
	entry [i,j,k] is the probability of moving to position k after
	applying action j to position i.}
    }
  }
  \item{basis}{Logical matrix describing some transformation of the
    components of the state. If \emph{btype==''power''} and if entry
    [i,j] is non-zero, then j-th power of the i-th component of the
    state is included in the regression basis. If
    \emph{btype==''laguerre''} and if entry $[i,j]$ is non-zero, then the
    j-th Laguerre polynomial of i-th component of the state is included
    in the regression basis. The object \eqn{basis} is processed
    row-wise.
  }
  \item{basis_type}{The type of tranformation to use for \emph{basis}:
  "power" and "laguerre".}
  \item{spline}{Logical value indicating whether linear splines should
    be used.}
  \item{knots}{Real valued matrix indicating the location of the knots
    for the linear splines. If entry [i,j] gives value \eqn{x}, then a
    knot at \eqn{x} is used for the j-th component of the state. If
    there is no knot, use \strong{NA} for matrix entry. For each row,
    the numbers should be placed before the \strong{NA} values.}
  \item{Basis}{User supplied function to represent other basis
    functions. The function should take in the following argument:
    \itemize{
      \item{\eqn{n \times d} matrix representing the \eqn{n}
	\eqn{d}-dimensional states.}
    }
    The function should output the following:
    \itemize{
      \item{Matrix with dimensions \eqn{n \times n_rbasis} representing
     the matrix to append to the design matrix horizontally on the
     right.}
    }   
  }
  \item{n_rbasis}{The number of basis functions added by the
    \emph{Basis} function above. Must be used if \emph{Basis} is given.}
}
\value{
  3-D array containing the prescribed policy. Entry [i,p,t] is
  for path i and position p at time t.
}
\examples{
library(StochasticProcess)
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
n_path2 <- 1000
path2 <- GBM(start, mu, vol, n_dec, n_path2, TRUE)
policy <- PathPolicy(path2, lsm$expected, Reward, control, basis, "power", TRUE, knots)
}
\author{Jeremy Yee}

