# =============================================================================
# classifiers.R - Base classifiers and cost-sensitive wrappers
# =============================================================================

# =============================================================================
# BASE CLASSIFIERS
# =============================================================================

# Train a classifier and return predictions.
# method: one of "LR", "penLR", "ridgeLR", "RF", "NB", "LDA", "QDA", "SVM"
# Returns list(prob, obj).
classify_base <- function(xtrain, ytrain, xnew, method, ...) {
  if (method == "LR") {
    dtrain <- data.frame(ytrain = ytrain, xtrain)
    obj <- glm(ytrain ~ ., family = binomial(link = 'logit'), data = dtrain, ...)
    if (is.null(xnew)) {
      prob <- predict(obj, type = "response")
    } else {
      prob <- predict(obj, data.frame(xnew), type = "response")
    }
  } else if (method == "penLR") {
    obj <- glmnet::cv.glmnet(x = xtrain, y = ytrain, family = "binomial", alpha = 1, ...)
    if (is.null(xnew)) {
      prob <- predict(obj, newx = as.matrix(xtrain), type = "response", s = "lambda.min")
    } else {
      prob <- predict(obj, newx = as.matrix(xnew), type = "response", s = "lambda.min")
    }
    prob <- as.numeric(prob[, 1])
  } else if (method == "ridgeLR") {
    obj <- glmnet::cv.glmnet(x = xtrain, y = ytrain, family = "binomial", alpha = 0, nlambda = 25, nfolds = 5, ...)
    if (is.null(xnew)) {
      prob <- predict(obj, newx = as.matrix(xtrain), type = "response", s = "lambda.min")
    } else {
      prob <- predict(obj, newx = as.matrix(xnew), type = "response", s = "lambda.min")
    }
    prob <- as.numeric(prob[, 1])
  } else if (method == "RF") {
    col_names <- paste0("V", 1:ncol(xtrain))
    colnames(xtrain) <- col_names
    obj <- randomForest::randomForest(x = xtrain, y = factor(ytrain), ...)
    if (is.null(xnew)) {
      prob <- predict(obj, type = "prob")[, "1"]
    } else {
      xnew <- as.matrix(xnew)
      colnames(xnew) <- col_names
      prob <- predict(obj, newdata = xnew, type = "prob")[, "1"]
    }
  } else if (method == "NB") {
    col_names <- paste0("V", 1:ncol(xtrain))
    colnames(xtrain) <- col_names
    obj <- naivebayes::naive_bayes(x = xtrain, y = as.character(ytrain))
    if (is.null(xnew)) {
      prob <- predict(obj, type = "prob")[, "1"]
    } else {
      colnames(xnew) <- col_names
      prob <- predict(obj, xnew, type = "prob")[, "1"]
    }
  } else if (method == "LDA") {
    col_names <- paste0("V", 1:ncol(xtrain))
    colnames(xtrain) <- col_names
    obj <- MASS::lda(x = xtrain, grouping = ytrain, ...)
    pos_class <- as.character(sort(unique(ytrain))[2])
    if (is.null(xnew)) {
      prob <- predict(obj, data.frame(xtrain))$posterior[, pos_class]
    } else {
      xnew <- as.data.frame(xnew)
      colnames(xnew) <- col_names
      prob <- predict(obj, xnew)$posterior[, pos_class]
    }
  } else if (method == "QDA") {
    col_names <- paste0("V", 1:ncol(xtrain))
    colnames(xtrain) <- col_names
    keep_cols <- which(apply(xtrain, 2, function(col)
      all(vapply(split(col, ytrain), function(v) length(unique(v)) > 1L, logical(1)))))
    xtrain_fit <- xtrain[, keep_cols, drop = FALSE]
    obj <- tryCatch(
      MASS::qda(x = xtrain_fit, grouping = ytrain, ...),
      error = function(e) {
        if (grepl("rank deficiency", conditionMessage(e), fixed = TRUE))
          MASS::lda(x = xtrain_fit, grouping = ytrain)
        else
          stop(e)
      }
    )
    attr(obj, "keep_cols") <- keep_cols
    pos_class <- as.character(sort(unique(ytrain))[2])
    if (is.null(xnew)) {
      prob <- predict(obj, data.frame(xtrain_fit))$posterior[, pos_class]
    } else {
      xnew_df <- as.data.frame(xnew)[, keep_cols, drop = FALSE]
      colnames(xnew_df) <- col_names[keep_cols]
      prob <- predict(obj, xnew_df)$posterior[, pos_class]
    }
  } else if (method == "SVM") {
    col_names <- paste0("V", 1:ncol(xtrain))
    colnames(xtrain) <- col_names
    obj <- e1071::svm(x = xtrain, y = factor(ytrain), probability = TRUE, ...)
    if (is.null(xnew)) {
      pred <- predict(obj, xtrain, probability = TRUE)
      prob <- attr(pred, "probabilities")[, "1"]
    } else {
      xnew <- as.matrix(xnew)
      colnames(xnew) <- col_names
      pred <- predict(obj, xnew, probability = TRUE)
      prob <- attr(pred, "probabilities")[, "1"]
    }
  } else {
    stop(sprintf("Unknown method: %s", method))
  }

  return(list(prob = prob, obj = obj))
}

score_base <- function(obj, xnew, method) {
  if (nrow(as.matrix(xnew)) == 0L) return(numeric(0))
  if (inherits(obj, "zero_classifier")) return(rep(0, nrow(as.matrix(xnew))))
  if (method == "LR") {
    return(predict(obj, data.frame(xnew), type = "response"))
  } else if (method == "penLR") {
    prob <- predict(obj, newx = as.matrix(xnew), type = "response", s = "lambda.min")
    return(as.numeric(prob[, 1]))
  } else if (method == "ridgeLR") {
    prob <- predict(obj, newx = as.matrix(xnew), type = "response", s = "lambda.min")
    return(as.numeric(prob[, 1]))
  } else if (method == "RF") {
    xnew <- as.matrix(xnew)
    colnames(xnew) <- paste0("V", 1:ncol(xnew))
    return(predict(obj, newdata = xnew, type = "prob")[, "1"])
  } else if (method == "NB") {
    xnew <- as.matrix(xnew)
    colnames(xnew) <- paste0("V", 1:ncol(xnew))
    return(predict(obj, xnew, type = "prob")[, "1"])
  } else if (method == "LDA") {
    xnew <- as.data.frame(xnew)
    colnames(xnew) <- paste0("V", 1:ncol(xnew))
    pos_class <- as.character(sort(rownames(obj$means))[2])
    return(predict(obj, xnew)$posterior[, pos_class])
  } else if (method == "QDA") {
    keep_cols <- attr(obj, "keep_cols")
    xnew <- as.data.frame(xnew)
    if (!is.null(keep_cols)) {
      xnew <- xnew[, keep_cols, drop = FALSE]
      colnames(xnew) <- paste0("V", keep_cols)
    } else {
      colnames(xnew) <- paste0("V", 1:ncol(xnew))
    }
    pos_class <- as.character(sort(rownames(obj$means))[2])
    return(predict(obj, xnew)$posterior[, pos_class])
  } else if (method == "SVM") {
    xnew <- as.matrix(xnew)
    colnames(xnew) <- paste0("V", 1:ncol(xnew))
    pred <- predict(obj, xnew, probability = TRUE)
    return(attr(pred, "probabilities")[, "1"])
  }
  stop(sprintf("Unknown method: %s", method))
}

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

# Bootstrap resampling to achieve the class ratio implied by wcost.
# Returns list(x, y).
stratification <- function(xtrain, ytrain, wcost) {
  stopifnot(wcost > 0, wcost < 1)
  ind0 <- which(ytrain == 0)
  ind1 <- which(ytrain == 1)
  if (wcost >= 0.5) {
    n1 <- sum(ytrain == 1)
    new_n0 <- round(n1 * wcost / (1 - wcost))
    bsind <- sample(ind0, size = new_n0, replace = TRUE)
    new_xtrain <- xtrain[c(bsind, ind1), ]
    new_ytrain <- ytrain[c(bsind, ind1)]
  } else {
    n0 <- sum(ytrain == 0)
    new_n1 <- round(n0 * (1 - wcost) / wcost)
    bsind <- sample(ind1, size = new_n1, replace = TRUE)
    new_xtrain <- xtrain[c(bsind, ind0), ]
    new_ytrain <- ytrain[c(bsind, ind0)]
  }

  return(list(x = new_xtrain, y = new_ytrain))
}

# =============================================================================
# CLASSIFY FUNCTION WRAPPERS
# =============================================================================

#' Cost-sensitive classification via stratified resampling
#'
#' A built-in \code{classify_fun} that resamples the training data to achieve
#' the class ratio implied by \code{cost} before fitting the classifier.
#'
#' @param xtrain Matrix of training features.
#' @param ytrain Integer vector of training labels (0/1).
#' @param cost Cost weight for class 0 (between 0 and 1).
#' @param xnew Matrix of new observations to score.
#' @param method Classifier method string passed to the base classifier.
#'   One of \code{"LR"}, \code{"penLR"}, \code{"ridgeLR"}, \code{"RF"},
#'   \code{"NB"}, \code{"LDA"}, \code{"QDA"}, \code{"SVM"}.
#' @param ... Additional arguments passed to the base classifier.
#' @return A list with elements \code{prob} (predicted probabilities) and
#'   \code{obj} (the fitted model object).
#' @examples
#' set.seed(1)
#' # Class 0: N(0, I);  Class 1: N(c(1.5, 0), I)
#' x0 <- matrix(rnorm(60 * 2),            60, 2)
#' x1 <- matrix(rnorm(40 * 2, mean = 1.5), 40, 2)
#' x  <- rbind(x0, x1)
#' y  <- c(rep(0L, 60), rep(1L, 40))
#' xnew <- matrix(rnorm(20 * 2), 20, 2)
#'
#' result <- classify_fun_stratified(x, y, cost = 0.8, xnew = xnew, method = "LR")
#' result$prob
#' @export
classify_fun_stratified <- function(xtrain, ytrain, cost, xnew, method = "LR", ...) {
  data_str <- stratification(xtrain, ytrain, cost)
  score <- classify_base(xtrain = data_str$x, ytrain = data_str$y, xnew, method, ...)
  return(score)
}

#' Cost-sensitive classification via decision threshold sweep
#'
#' A built-in \code{classify_fun} that fits a standard (unweighted) classifier
#' and varies the decision threshold rather than resampling. Intended to be
#' paired with \code{threshold_fun = function(cost) cost} in
#' \code{\link{select_np_cost}}.
#'
#' @inheritParams classify_fun_stratified
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(100 * 2), 100, 2)
#' y <- as.integer(x[, 1] + rnorm(100, sd = 2) > 0)
#' xnew <- matrix(rnorm(20 * 2), 20, 2)
#'
#' # Pair with threshold_fun = function(cost) cost in select_np_cost
#' result <- classify_fun_threshold(x, y, cost = 0.8, xnew = xnew, method = "LR")
#' result$prob
#' @export
classify_fun_threshold <- function(xtrain, ytrain, cost, xnew, method = "LR", ...) {
  classify_base(xtrain, ytrain, xnew = xnew, method = method, ...)
}

#' Cost-sensitive classification via SVM class weights
#'
#' A built-in \code{classify_fun} that fits an SVM with class weights derived
#' from \code{cost}, setting the weight for class 0 to \code{cost / (1 - cost)}.
#'
#' @inheritParams classify_fun_stratified
#' @export
classify_fun_svm_weighted <- function(xtrain, ytrain, cost, xnew, method = "SVM", ...) {
  classify_base(xtrain, ytrain, xnew, method = "SVM",
                class.weights = c("0" = cost / (1 - cost), "1" = 1), ...)
}
