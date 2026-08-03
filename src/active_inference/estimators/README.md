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
| [`variational.py`](variational.py) | **(Ch.4)** `coordinate_search_vfe`, `fixed_form_vi`, `free_form_cavi`, `MeanFieldConfig`, and their result traces (`CoordinateSearchResult`, `FixedFormResult`, `CAVIResult`). |
| [`predictive_coding.py`](predictive_coding.py) | **(Ch.5)** `predictive_coding_inference`, `multivariate_predictive_coding`, `hierarchical_predictive_coding`, `HierarchicalPCModel`, and the result traces `PredictiveCodingResult`/`MultivariatePCResult`/`HierarchicalPCResult`. |
| [`generalized_filtering.py`](generalized_filtering.py) | **(Ch.6)** online generalized filtering, generalized measurements, and vector generalized-coordinate filtering. |
| [`active_inference.py`](active_inference.py) | **(Ch.7)** univariate and multivariate active generalized-filtering loops. |
| [`continuous_learning.py`](continuous_learning.py) | **(Ch.8)** `simulate_learning_attention` and `LearningAttentionResult` for perception, learning, and attention traces. |
| [`pomdp.py`](pomdp.py) | **(Ch.9–10)** Grid World planning, policy enumeration, POMDP agent simulation; Dirichlet parameter learning, precision sweeps, two-armed bandit, and hierarchical-agent simulations. |
| [`pomdp_extensions.py`](pomdp_extensions.py) | **(Ch.11)** Part III demos: sophisticated planning, state-preference schedules, parameter forgetting, structure learning, preference/habit learning, and path-policy computation. |
| [`applications.py`](applications.py) | **(Ch.13)** Application demos: robot navigation, fault-tolerant control, social inference, and the robotics-theory landscape. |
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
    # Chapter 4 — variational inference
    coordinate_search_vfe, fixed_form_vi, free_form_cavi, MeanFieldConfig,
    CoordinateSearchResult, FixedFormResult, CAVIResult,
    # Chapter 5 — predictive coding
    predictive_coding_inference, multivariate_predictive_coding,
    hierarchical_predictive_coding, HierarchicalPCModel,
    PredictiveCodingResult, MultivariatePCResult, HierarchicalPCResult,
    # Chapter 6–7 — generalized filtering and action
    generalized_measurements_from_series, generalized_vector_filter,
    simulate_multivariate_active_inference, MultivariateActiveInferenceResult,
    # Chapter 8 — continuous learning and attention
    simulate_learning_attention, LearningAttentionResult,
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
| `coordinate_search_vfe` / `fixed_form_vi` / `free_form_cavi` | — | Iterative (VFE descent) | Chapter 4 (Examples 4.1, 4.6, 4.7) |
| `predictive_coding_inference` / `multivariate_*` / `hierarchical_*` | Linear fixed point closed-form | Iterative (PC descent) | Chapter 5 (Examples 5.1, 5.3, 5.4, 5.7) |
| `generalized_vector_filter` | — | Generalized-coordinate VFE descent | Chapter 6 (Example 6.7) |
| `simulate_multivariate_active_inference` | — | Coupled perception-action flow | Chapter 7 (§7.5) |
| `simulate_learning_attention` | — | Damped continuous learning / attention flow | Chapter 8 (Example 8.1) |

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

See `tests/estimators/`: `test_mle.py`, `test_map.py`,
`test_gradient_descent.py`, `test_linear_regression.py`, `test_em.py`,
`test_variational.py` (Ch.4), `test_predictive_coding.py` (Ch.5),
`test_continuous_learning.py` (Ch.8), and `test_recovery.py`
(cross-estimator parameter-recovery checks).
