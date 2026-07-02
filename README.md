# costnp

> Neyman-Pearson cost selection for cost-sensitive classification

`costnp` selects a cost parameter (and corresponding classifier) that satisfies
a **Neyman-Pearson (NP) guarantee**: it minimizes the type II error (false
negative rate) subject to the type I error (false positive rate) being bounded
above by a target `alpha`, with high probability `1 - delta`.

## Installation

```r
# install.packages("devtools")
devtools::install_github("john-j-park/costnp")
```

## Quick start

Data is supplied as a list with a feature matrix `x` and a 0/1 label vector `y`,
where class 0 is the class whose false positive rate you want to bound.

```r
library(costnp)
set.seed(1)

# Class 0: N(0, I);  Class 1: N(c(1.5, 0, 0), I)
x0 <- matrix(rnorm(200 * 3),             200, 3)
x1 <- matrix(rnorm(200 * 3, mean = 1.5), 200, 3)
data <- list(x = rbind(x0, x1), y = c(rep(0L, 200), rep(1L, 200)))

# Population data — same distribution, larger sample
x0p <- matrix(rnorm(5000 * 3),             5000, 3)
x1p <- matrix(rnorm(5000 * 3, mean = 1.5), 5000, 3)
pop_data <- list(x = rbind(x0p, x1p), y = c(rep(0L, 5000), rep(1L, 5000)))

result <- select_np_cost(
  data            = data,
  population_data = pop_data,
  alpha           = 0.1,
  delta           = 0.1,
  method          = "LR",
  seed            = 42
)

result$cost              # selected cost parameter
result$population_type1  # type I error on population_data
result$population_type2  # type II error on population_data
```

With probability at least `1 - delta` (here 0.9), the population type I error is
at most `alpha` (here 0.1). A tighter type I bound generally increases the
type II error.

## `costnp` vs `costnp_plus`

Two named wrappers differ in how the final cost is chosen at the boundary
between under- and over-estimating the target:

- **`costnp()`** snaps to the last grid point that still satisfies the bound —
  the more conservative choice.
- **`costnp_plus()`** linearly interpolates between the last under-estimate and
  first over-estimate — slightly less conservative, potentially lower type II
  error.

## Base classifiers

The `method` argument selects the base classifier:

| `method`    | Classifier                | Package        |
|-------------|---------------------------|----------------|
| `"LR"`      | Logistic regression       | base R         |
| `"penLR"`   | Lasso logistic            | glmnet         |
| `"ridgeLR"` | Ridge logistic            | glmnet         |
| `"RF"`      | Random forest             | randomForest   |
| `"NB"`      | Naive Bayes               | naivebayes     |
| `"LDA"`     | Linear discriminant       | MASS           |
| `"QDA"`     | Quadratic discriminant    | MASS           |
| `"SVM"`     | Support vector machine    | e1071          |

By default the cost-sensitivity is injected via stratified resampling. You can
instead sweep the decision threshold (`classify_fun_threshold`), use SVM class
weights (`classify_fun_svm_weighted`), or supply your own `classify_fun`. See
`vignette("introduction", package = "costnp")` for details.

## License

MIT © costnp authors
