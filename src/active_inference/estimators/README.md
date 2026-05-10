# `estimators/` — point estimates and parameter learning

Algorithms that operate on top of a `core/` model: analytic estimators,
gradient descent, conjugate Bayesian updates, and EM.

## Files

| File | What it defines |
|---|---|
| [`mle.py`](mle.py) | Univariate hidden-state MLE under linear-Gaussian: `mle_analytic_linear`, `mle_loss`, `mle_grad_x`. |
| [`map.py`](map.py) | Univariate MAP — same shape as MLE plus a Gaussian-prior penalty. |
| [`gradient_descent.py`](gradient_descent.py) | Generic 1-D minimizer with optional analytic gradient and history tracking. |
| [`linear_regression.py`](linear_regression.py) | OLS via the normal equation, vectorized GD, `BayesianLinearRegression` with `fit` / `fit_sequential` / `predictive`. |
| [`em.py`](em.py) | Linear factor-analysis E and M steps + `fit_factor_analysis` loop. |
| `__init__.py` | Re-exports the public surface. |

## Public API

```python
from active_inference.estimators import (
    mle_analytic_linear, mle_loss, mle_grad_x,
    map_analytic_linear, map_loss, map_grad_x,
    gradient_descent, GradientDescentResult,
    add_intercept, mle_linear_regression, gd_linear_regression,
    GDRegressionResult,
    BayesianLinearRegression, BLRPosterior,
    fit_factor_analysis, factor_analysis_e_step, factor_analysis_m_step,
    incomplete_log_likelihood, FactorAnalysisResult,
)
```

The top-level `active_inference` package re-exports the same names.

## Overview

| Estimator | Closed-form? | Gradient-based fallback? | Where used |
|---|---|---|---|
| `mle_analytic_linear` | Yes | `gradient_descent` + `mle_grad_x` | Examples 2.7, 2.8 |
| `map_analytic_linear` | Yes | `gradient_descent` + `map_grad_x` | Examples 2.2, 2.9 |
| `mle_linear_regression` | Yes (lstsq) | `gd_linear_regression` | Examples 3.1, 3.2, 3.3 |
| `BayesianLinearRegression` | Yes (conjugate) | — | Example 3.5 |
| `fit_factor_analysis` | Iterative (EM) | — | Example 3.7 |

## Design Decisions

- **Closed-form first.** Every linear-Gaussian recipe ships an analytic
  estimator; gradient descent is provided alongside so readers can verify
  the iterative answer matches the closed form.
- **Pure NumPy.** No `torch` / `jax`; reproducible across platforms.
- **Result dataclasses.** `GradientDescentResult`, `GDRegressionResult`,
  `BLRPosterior`, `FactorAnalysisResult` carry the full iteration history
  so downstream code can plot or animate convergence without re-running
  the algorithm.
- **Validation up front.** All estimators raise `ValueError` for negative
  variances, non-positive learning rates, mismatched shapes.

## Dependencies

`numpy` only.

## Testing

See `tests/estimators/test_mle.py`, `test_map.py`,
`test_gradient_descent.py`, `test_linear_regression.py`, `test_em.py`.
