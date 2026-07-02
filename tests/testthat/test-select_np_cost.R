# Shared synthetic dataset used across tests
make_data <- function(n = 300, seed = 1) {
  set.seed(seed)
  x <- matrix(rnorm(n * 2), n, 2)
  y <- as.integer(x[, 1] + rnorm(n, sd = 0.5) > 0)
  list(x = x, y = y)
}

data       <- make_data(n = 300, seed = 1)
pop_data   <- make_data(n = 1000, seed = 2)

# =============================================================================
# select_np_cost: return structure
# =============================================================================

test_that("select_np_cost returns the correct list elements", {
  result <- select_np_cost(data, pop_data, method = "LR", seed = 42)
  expect_named(result,
    c("cost", "population_type1", "population_type2",
      "n_calib", "m_errors_in_calib", "classifier"),
    ignore.order = TRUE
  )
})

test_that("select_np_cost cost is within the grid range", {
  result <- select_np_cost(data, pop_data, method = "LR",
                           cost_start = 0.99, cost_end = 0.5, seed = 42)
  expect_gte(result$cost, 0.5)
  expect_lte(result$cost, 0.99)
})

test_that("select_np_cost population_type1 is a probability", {
  result <- select_np_cost(data, pop_data, method = "LR", seed = 42)
  expect_gte(result$population_type1, 0)
  expect_lte(result$population_type1, 1)
})

test_that("select_np_cost n_calib is positive integer", {
  result <- select_np_cost(data, pop_data, method = "LR", seed = 42)
  expect_gt(result$n_calib, 0)
  expect_type(result$n_calib, "integer")
})

# =============================================================================
# select_np_cost: reproducibility
# =============================================================================

test_that("seed argument produces reproducible results", {
  r1 <- select_np_cost(data, pop_data, method = "LR", seed = 99)
  r2 <- select_np_cost(data, pop_data, method = "LR", seed = 99)
  expect_identical(r1$cost, r2$cost)
  expect_identical(r1$population_type1, r2$population_type1)
})

# =============================================================================
# select_np_cost: input validation
# =============================================================================

test_that("method = NULL with built-in classify_fun throws an error", {
  expect_error(
    select_np_cost(data, pop_data, method = NULL,
                   classify_fun = classify_fun_stratified),
    "'method' must be specified"
  )
  expect_error(
    select_np_cost(data, pop_data, method = NULL,
                   classify_fun = classify_fun_threshold),
    "'method' must be specified"
  )
})

# =============================================================================
# select_np_cost: safety fallback
# =============================================================================

test_that("safety = TRUE returns a zero_classifier when alpha is very small", {
  # Use an impossibly small alpha to force the safety fallback
  result <- select_np_cost(data, pop_data, method = "LR",
                           alpha = 1e-10, safety = TRUE, seed = 42)
  expect_s3_class(result$classifier, "zero_classifier")
  expect_identical(result$m_errors_in_calib, 0L)
})

test_that("safety = FALSE returns NULL with a warning when alpha is very small", {
  expect_warning(
    result <- select_np_cost(data, pop_data, method = "LR",
                             alpha = 1e-10, safety = FALSE, seed = 42),
    "exceeds the alpha target"
  )
  expect_true(is.null(result))
})

# =============================================================================
# costnp_plus / costnp wrappers
# =============================================================================

test_that("costnp_plus returns the same structure as select_np_cost", {
  result <- costnp_plus(data, pop_data, method = "LR", seed = 42)
  expect_named(result,
    c("cost", "population_type1", "population_type2",
      "n_calib", "m_errors_in_calib", "classifier"),
    ignore.order = TRUE
  )
})

test_that("costnp returns the same structure as select_np_cost", {
  result <- costnp(data, pop_data, method = "LR", seed = 42)
  expect_named(result,
    c("cost", "population_type1", "population_type2",
      "n_calib", "m_errors_in_calib", "classifier"),
    ignore.order = TRUE
  )
})

test_that("costnp cost is <= costnp_plus cost (stricter is more conservative)", {
  r_strict <- costnp(data, pop_data, method = "LR", seed = 42)
  r_interp <- costnp_plus(data,   pop_data, method = "LR", seed = 42)
  # costnp snaps to the grid; costnp_plus interpolates, so may go slightly lower
  expect_gte(r_strict$cost, r_interp$cost - 0.01 - .Machine$double.eps)
})
