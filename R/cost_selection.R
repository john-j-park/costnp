# =============================================================================
# cost_selection.R
# =============================================================================
# Depends on: classifiers.R

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

build_cost_grid <- function(cost_start, cost_end, cost_by) {
  step <- if (cost_start > cost_end) -abs(cost_by) else abs(cost_by)
  seq(cost_start, cost_end, by = step)
}

eval_population <- function(calib, population_data, chosen_cost, get_threshold, obj = NULL) {
  pop_obj   <- if (is.null(obj)) calib$fit_pop(chosen_cost) else obj
  threshold <- get_threshold(chosen_cost)

  pop_is_0  <- population_data$y == 0
  pop_prob_0 <- calib$predict_pop(pop_obj, population_data$x[pop_is_0, , drop = FALSE])
  pop_type1  <- mean(pop_prob_0 > threshold)

  pop_is_1  <- population_data$y == 1
  pop_prob_1 <- calib$predict_pop(pop_obj, population_data$x[pop_is_1, , drop = FALSE])
  pop_type2  <- mean(pop_prob_1 <= threshold)

  list(obj = pop_obj, type1 = pop_type1, type2 = pop_type2)
}

# =============================================================================
# COST SELECTION METHODS
# =============================================================================
# Internal helper: split data, fit classifier on train set, and return scoring
# functions used throughout the cost selection loop.
# Returns list(get_result, get_threshold, fit_pop, predict_pop, n_calib).
setup_calibration <- function(data, method = NULL, data_pct = 0.5,
                              classify_fun = classify_fun_stratified,
                              threshold_fun = function(cost) 0.5) {
  idx_0       <- which(data$y == 0)
  train_idx_0 <- sample(idx_0, floor(data_pct * length(idx_0)))
  train_idx   <- (data$y == 1) | (seq_along(data$y) %in% train_idx_0)
  xtrain      <- data$x[train_idx, , drop = FALSE]
  ytrain      <- data$y[train_idx]
  xcalib      <- data$x[!train_idx, , drop = FALSE]
  n_calib     <- nrow(xcalib)

  get_threshold <- threshold_fun
  fit_pop       <- function(cost) classify_fun(xtrain, ytrain, cost = cost,
                                               xnew = xtrain[1, , drop = FALSE],
                                               method = method)$obj
  get_result    <- function(cost) classify_fun(xtrain, ytrain, cost = cost,
                                               xnew = xcalib, method = method)
  predict_pop   <- function(obj, xnew) {
    if (is.null(method)) {
      stop(
        "Cannot score population data when method = NULL. ",
        "Either supply a method string, or override 'predict_pop' via a ",
        "custom classify_fun."
      )
    }
    score_base(obj, xnew, method)
  }

  list(get_result = get_result, get_threshold = get_threshold,
       fit_pop = fit_pop, predict_pop = predict_pop, n_calib = n_calib)
}

#' Cost selection with Neyman-Pearson guarantees
#'
#' @param data A list with elements \code{x} (feature matrix) and \code{y}
#'   (0/1 label vector). Used for calibration and training.
#' @param population_data A list with the same structure as \code{data}.
#'   Used to evaluate population type I and type II error rates.
#' @param alpha Target upper bound on the type I error rate.
#' @param delta Confidence level; the guarantee holds with probability
#'   \code{1 - delta}.
#' @param cost_start Starting value of the cost grid (highest cost).
#' @param cost_end Ending value of the cost grid (lowest cost).
#' @param cost_by Step size for the cost grid.
#' @param data_pct Fraction of class-0 observations used for training
#'   (remainder go to calibration).
#' @param method Classifier method string passed through to \code{classify_fun}.
#'   Required when using a built-in \code{classify_fun}; may be \code{NULL}
#'   only when supplying a fully custom \code{classify_fun} that ignores it.
#' @param verbose If \code{TRUE}, prints cost and alpha values at each step.
#' @param seed Integer seed passed to \code{set.seed()} for reproducibility.
#'   If \code{NULL} (default), no seed is set.
#' @param interpolate If \code{TRUE}, linearly interpolates the chosen cost
#'   between the last under-estimate and first over-estimate grid points.
#' @param safety If \code{TRUE}, returns a zero classifier (always predicts 0)
#'   when the first grid point already exceeds the alpha target, rather than
#'   warning and returning an invalid result.
#' @param classify_fun Function with signature
#'   \code{(xtrain, ytrain, cost, xnew, method, ...)} returning
#'   \code{list(prob, obj)}. Defaults to \code{\link{classify_fun_stratified}}.
#'   Use \code{\link{classify_fun_threshold}} paired with
#'   \code{threshold_fun = function(cost) cost} to sweep decision thresholds
#'   instead of resampling.
#' @param threshold_fun Function of \code{cost} returning the decision
#'   threshold. Defaults to \code{function(cost) 0.5}.
#' @return A list with elements \code{cost}, \code{population_type1},
#'   \code{population_type2}, \code{n_calib}, \code{m_errors_in_calib},
#'   and \code{classifier}. Returns \code{NULL} (with a warning) if the first
#'   cost grid point already exceeds the alpha target and \code{safety = FALSE}.
#' @examples
#' set.seed(1)
#' # Class 0: N(0, I);  Class 1: N(c(1.5, 0), I)
#' x0 <- matrix(rnorm(120 * 2),             120, 2)
#' x1 <- matrix(rnorm( 80 * 2, mean = 1.5),  80, 2)
#' data <- list(x = rbind(x0, x1), y = c(rep(0L, 120), rep(1L, 80)))
#'
#' x0p <- matrix(rnorm(5000 * 2),             5000, 2)
#' x1p <- matrix(rnorm(5000 * 2, mean = 1.5), 5000, 2)
#' pop_data <- list(x = rbind(x0p, x1p), y = c(rep(0L, 5000), rep(1L, 5000)))
#'
#' result <- select_np_cost(data, pop_data, method = "LR", alpha = 0.1, seed = 1)
#' result$cost
#' result$population_type1
#' @export
select_np_cost <- function(data, population_data, alpha = 0.1, delta = 0.1,
                                  cost_start = 0.99, cost_end = 0.2, cost_by = 0.01,
                                  data_pct = 0.5, method = NULL,
                                  verbose = FALSE, seed = NULL,
                                  interpolate = FALSE, safety = TRUE,
                                  classify_fun = classify_fun_stratified,
                                  threshold_fun = function(cost) 0.5) {
  builtin_funs <- list(classify_fun_stratified, classify_fun_threshold)
  if (is.null(method) && any(vapply(builtin_funs, identical, logical(1), classify_fun))) {
    stop(
      "'method' must be specified when using a built-in classify_fun ",
      "(e.g. method = \"LR\"). ",
      "Supply a method string, or provide a custom 'classify_fun' that ",
      "does not rely on 'method'."
    )
  }
  if (!is.null(seed)) set.seed(seed)
  calib <- setup_calibration(data, method = method, data_pct = data_pct,
                             classify_fun = classify_fun, threshold_fun = threshold_fun)
  get_result    <- calib$get_result
  get_threshold <- calib$get_threshold
  n_calib       <- calib$n_calib
  costvec       <- build_cost_grid(cost_start, cost_end, cost_by)
  chosen_cost   <- costvec[1]
  chosen_obj    <- NULL
  alpha_underestimate <- NA
  m <- NA

  for (i in seq_along(costvec)) {
    current_cost <- costvec[i]
    result       <- get_result(current_cost)
    yhat         <- as.numeric(result$prob > get_threshold(current_cost))
    m            <- sum(yhat == 1)
    alpha_upper  <- qbeta(1 - delta, m + 1, n_calib - m)

    if (current_cost == costvec[1] && alpha_upper >= alpha) {
      if (safety) {
        zero_clf <- structure(list(), class = "zero_classifier")
        pop      <- eval_population(calib, population_data, costvec[1], get_threshold,
                                    obj = zero_clf)
        return(list(
          cost               = costvec[1],
          population_type1   = pop$type1,
          n_calib            = n_calib,
          m_errors_in_calib  = 0L,
          classifier         = zero_clf,
          population_type2   = pop$type2
        ))
      }
      warning("First cost grid point already exceeds the alpha target; returning NULL. Use safety = TRUE for a zero-classifier fallback.")
      return(NULL)
    }

    if (alpha_upper <= alpha || current_cost == costvec[1]) {
      chosen_cost         <- current_cost
      chosen_obj          <- result$obj
      alpha_underestimate <- alpha_upper
      if (verbose) {
        cat("Alpha Underestimate:", alpha_underestimate, "\n")
        cat("Cost:", current_cost, "\n")
      }
    } else {
      alpha_overestimate <- alpha_upper
      if (verbose) {
        cat("Alpha Overestimate:", alpha_overestimate, "\n")
        cat("Cost:", current_cost, "\n")
      }
      if (alpha_overestimate <= alpha_underestimate) {
        chosen_cost <- current_cost
      } else if (interpolate) {
        percentile  <- (alpha_overestimate - alpha) / (alpha_overestimate - alpha_underestimate)
        chosen_cost <- costvec[i] + percentile * (costvec[i - 1] - costvec[i])
        if (verbose) {
          cat("Percentile:", percentile, "\n")
          cat("Current Cost:", costvec[i], "\n")
          cat("Previous Cost:", costvec[i - 1], "\n")
        }
      } else {
        chosen_cost <- costvec[i - 1]
        if (verbose) {
          cat("Current Cost:", costvec[i], "\n")
          cat("Previous Cost:", costvec[i - 1], "\n")
        }
      }
      result_final <- get_result(chosen_cost)
      yhat        <- as.numeric(result_final$prob > get_threshold(chosen_cost))
      m           <- sum(yhat == 1)
      alpha_upper <- qbeta(1 - delta, m + 1, n_calib - m)
      chosen_obj  <- result_final$obj
      break
    }
  }

  pop <- eval_population(calib, population_data, chosen_cost, get_threshold, obj = chosen_obj)

  if (verbose) {
    cat("Alpha Upper:", alpha_upper, "\n")
    cat("Pop Type I:", pop$type1, "\n")
    cat("Cost:", chosen_cost, "\n")
  }

  list(
    cost               = chosen_cost,
    population_type1   = pop$type1,
    n_calib            = n_calib,
    m_errors_in_calib  = m,
    classifier         = pop$obj,
    population_type2   = pop$type2
  )
}

#' Cost selection with interpolation between grid points (costnp+)
#'
#' A convenience wrapper around \code{\link{select_np_cost}} with
#' \code{interpolate = TRUE}. The chosen cost is linearly interpolated
#' between the last under-estimate and first over-estimate grid points,
#' making it slightly less conservative than \code{\link{costnp}}.
#'
#' @param ... Arguments passed to \code{\link{select_np_cost}}.
#' @inherit select_np_cost return
#' @examples
#' set.seed(1)
#' # Class 0: N(0, I);  Class 1: N(c(1.5, 0), I)
#' x0 <- matrix(rnorm(120 * 2),             120, 2)
#' x1 <- matrix(rnorm( 80 * 2, mean = 1.5),  80, 2)
#' data <- list(x = rbind(x0, x1), y = c(rep(0L, 120), rep(1L, 80)))
#'
#' x0p <- matrix(rnorm(5000 * 2),             5000, 2)
#' x1p <- matrix(rnorm(5000 * 2, mean = 1.5), 5000, 2)
#' pop_data <- list(x = rbind(x0p, x1p), y = c(rep(0L, 5000), rep(1L, 5000)))
#'
#' result <- costnp_plus(data, pop_data, method = "LR", seed = 1)
#' result$cost
#' @export
costnp_plus <- function(...) {
  select_np_cost(..., interpolate = TRUE)
}

#' Cost selection snapped to the grid (costnp)
#'
#' A convenience wrapper around \code{\link{select_np_cost}} with
#' \code{interpolate = FALSE}. The chosen cost is always a grid point,
#' making it the more conservative of the two named wrappers.
#'
#' @param ... Arguments passed to \code{\link{select_np_cost}}.
#' @inherit select_np_cost return
#' @examples
#' set.seed(1)
#' # Class 0: N(0, I);  Class 1: N(c(1.5, 0), I)
#' x0 <- matrix(rnorm(120 * 2),             120, 2)
#' x1 <- matrix(rnorm( 80 * 2, mean = 1.5),  80, 2)
#' data <- list(x = rbind(x0, x1), y = c(rep(0L, 120), rep(1L, 80)))
#'
#' x0p <- matrix(rnorm(5000 * 2),             5000, 2)
#' x1p <- matrix(rnorm(5000 * 2, mean = 1.5), 5000, 2)
#' pop_data <- list(x = rbind(x0p, x1p), y = c(rep(0L, 5000), rep(1L, 5000)))
#'
#' result <- costnp(data, pop_data, method = "LR", seed = 1)
#' result$cost
#' @export
costnp <- function(...) {
  select_np_cost(..., interpolate = FALSE)
}
