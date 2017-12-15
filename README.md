# *rlsm* - R Package

## Description

This package implements least squares Monte Carlo and duality methods.
 Please contact me by email (jeremyyee@outlook.com.au) or through my
 GitHub account to report any issues.

## Example: Bermuda put option

### Continuation value function approximation

Let us consider the valuation of a Bermuda put option with strike
price **40** and expiry date of **1** year. The underlying asset price
follows a geometric Brownian motion. We assume the option is
exercisable at 51 evenly spaced time points, including one at
beginning and one at the end of the year.
~~~
step <- 0.02
mu <- 0.06 * step
vol <- 0.2 * sqrt(step)
n_dec <- 51
start <- 36
strike <- 40
btype <- "power"  ## power, laguerre
~~~
Then we generate the paths.
~~~
n_path <- 100000
path <- GBM(start, mu, vol, n_dec, n_path, TRUE)
~~~
Define the position control.
~~~
control <- matrix(c(c(1, 1), c(2, 1)), nrow = 2, byrow = TRUE)
~~~
Set the regression basis. Here we use linear splines.
~~~
basis <- matrix(c(1, 1), nrow = 1)
knots <- matrix(c(30, 40, 50), nrow = 1)
~~~
Specify the reward and scrap functions.
~~~
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
~~~
Then perform least squares Monte Carlo.
~~~
lsm <- LSM(path, Reward, Scrap, control, basis, TRUE, btype, TRUE, knots)
~~~
Below prints the value of the put option,
~~~
print(mean(lsm$value[, 2, 1]))
print(sd(lsm$value[, 2, 1])/sqrt(n_path))
~~~
We can plot the continuation value function using the code below.
~~~
## Plot the value function
n_grid <- 451
states <- seq(15, 60, length = n_grid)
sBasis <- cbind(states, states^2, rep(1,n_grid))
for (i in 1:length(knots)) {
  sBasis <- cbind(sBasis, pmax(states - knots[i], 0))
}
vFunction <- sBasis %*% lsm$expected[,2,1]
plot(states,vFunction,type="l", main="LSM", xlab="z", ylab="value")
~~~

### Primal-dual bounds

Having computed the function approximations above, we can now
calculate the bounds on the value of the option. Suppose that the
current price of the underlying stock is **36**.

~~~
## Duality bounds 
n_path2 <- 1000
path2 <- GBM(start, mu, vol, n_dec, n_path2, TRUE)
n_subsim <- 1000
subsim <- NestedGBM(path2, mu, vol, n_subsim, TRUE)
time2 <- proc.time()
mart <- AddDual(path2, subsim, lsm$expected, Reward, Scrap, control, basis, btype, TRUE, knots)
time2 <- proc.time() - time2
policy <- PathPolicy(path2, lsm$expected, Reward, control, basis, btype, TRUE, knots)
bounds <- Bounds(path2, Reward, Scrap, control, mart, policy)
~~~
Print bound estimates as below
~~~
print(mean(bounds$primal[,2,1]))
print(sd(bounds$primal[,2,1])/sqrt(n_path2))
print(mean(bounds$dual[,2,1]))
print(sd(bounds$dual[,2,1])/sqrt(n_path2))
~~~
which gives
~~~
[1] 4.461242
[1] 0.008069322
[1] 4.535191
[1] 0.006654249
~~~
