# Chapter 3 — concept map

Chapter 3 stitches *learning* and *inference* together. The previous chapter
treated the agent's parameters as already known; here we look at how they get
estimated in the first place, and what happens when both the parameters and
the hidden state are unknown at the same time.

## The five recipes

Each is exposed as a configurable building block in `active_inference` and
demonstrated by a thin orchestrator in `chapters/chapter_03/`.

| Recipe | Key API | Orchestrator |
|--------|---------|--------------|
| Closed-form linear regression (MLE) | `mle_linear_regression(X, y)` | `example_3_1_linear_regression_mle.py` |
| Gradient descent over the squared loss | `gd_linear_regression(X, y, learning_rate=...)` | `example_3_2_linear_regression_gd.py` |
| Multiple regression (vectorized) | `mle_linear_regression(X, y)` (batched) | `example_3_3_multiple_regression.py` |
| Multivariate normal anatomy | `mvn_pdf`, `mvn_sample`, `confidence_ellipse` | `example_3_4_multivariate_gaussian.py` |
| Bayesian linear regression | `BayesianLinearRegression(...).fit(X, y)` | `example_3_5_bayesian_linear_regression.py` |
| Linear Gaussian System (sensor fusion) | `LinearGaussianSystem(...).posterior_batch(Y)` | `example_3_6_lgs_food_localization.py` |
| Factor analysis via EM | `fit_factor_analysis(Y, n_factors=...)` | `example_3_7_factor_analysis_em.py` |

## Reusable building blocks

* **`active_inference.core.distributions`** — `mvn_pdf`, `mvn_log_pdf`,
  `mvn_sample`, `mahalanobis_squared`, `isotropic_cov`, `diagonal_cov`. All
  multivariate helpers use Cholesky-based solves, never explicit matrix
  inverses.
* **`active_inference.core.lgs`** — `LinearGaussianSystem` with
  `posterior(y)` and `posterior_batch(Y)` methods returning a
  `LGSPosterior` (mean, covariance, std, precision).
* **`active_inference.estimators.linear_regression`** —
  `mle_linear_regression`, `gd_linear_regression`,
  `BayesianLinearRegression` (with `fit`, `fit_sequential`, and posterior
  predictive helpers).
* **`active_inference.estimators.em`** — `fit_factor_analysis`,
  plus the individual `factor_analysis_e_step` and
  `factor_analysis_m_step` helpers for didactic walk-throughs.
* **`active_inference.visualizations.animations`** — drop-in helpers for
  `animate_2d_posterior`, `animate_em_convergence`, etc.
  No FFmpeg dependency: GIFs are written via the bundled pillow writer.

## Where the book takes this next

Chapter 4 generalizes the closed-form posteriors here to settings where the
posterior is no longer Gaussian and replaces the EM machinery with
variational Bayesian inference. The classes here are designed so that those
upgrades plug in naturally — `LinearGaussianMVModel` already exposes
`log_likelihood` and `log_prior`, which is the interface a variational scheme
needs.
