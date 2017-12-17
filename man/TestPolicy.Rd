\name{TestPolicy}
\alias{TestPolicy}
\title{TestPolicy}
\description{Testing prescribed policy for sample paths.}
\usage{
TestPolicy(start_position, path, control, Reward, Scrap, path_action)
}
\arguments{
  \item{start_position}{Starting position.}
  \item{path}{3-D array representing sample paths.  Entry [i,,j]
    represents the state at time j for sample path i.}
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
  \item{Scrap}{User supplied function to represent the scrap function.
    The function should take in the following argument:
    \itemize{
      \item{\eqn{n \times d} matrix representing the \eqn{n}
	\eqn{d}-dimensional states. }
    }
    The function should output the following:
    \itemize{
      \item{Matrix with dimensions \eqn{n \times p} representing the
	scrap where \eqn{n} is the number of sample paths and \eqn{p} is
	the number of positions. The \eqn{[i, j]}-th entry corresponds
	to the scrap at position \eqn{j} for the \eqn{i}-th path.}
    }   
  }
  \item{path_action}{3-D array containing the prescribed policy. Entry
    [i,j,k] is for path i and position j at time k.}
}
\value{
  Array containing the values for each path.
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
policy <- PathPolicy(path2, lsm$expected, Reward, control, basis,
"power", TRUE, knots)
test <- TestPolicy(2, path, control, Reward, Scrap, policy)
}
\author{Jeremy Yee}

