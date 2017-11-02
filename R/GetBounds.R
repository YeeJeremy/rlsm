## Copyright 2015 <Jeremy Yee> <jeremyyee@outlook.com.au>
## Obtaining confidence intervals for primal-dual approach
################################################################################

GetBounds <- function(duality, alpha, position) {
    n_path <- dim(duality$primal)[1]
    primal <- mean(duality$primal[, position, 1]) 
    primal_error <- qnorm(1 - alpha/2) * sd(duality$primal[, position, 1])/sqrt(n_path)
    dual <- mean(duality$dual[, position, 1])
    dual_error <- qnorm(1 - alpha/2) * sd(duality$dual[, position, 1])/sqrt(n_path)
    return(c(primal - primal_error, dual + dual_error))
}
