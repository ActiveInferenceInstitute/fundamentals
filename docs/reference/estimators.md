# `active_inference.estimators` — module reference

Point-estimate and parameter-learning algorithms. Together with
`core.inference` (exact grid-based posterior) and `core.lgs` (closed-form
multivariate posterior) these cover every learning / inference path used in
Chapters 2 and 3.

## `estimators.mle`

Univariate maximum-likelihood estimation of the hidden state given a linear
generating function.

| Symbol | Role |
|---|---|
| `mle_analytic_linear(y_obs, beta0, beta1)` | Closed-form ``x_MLE = (mean(y) − β₀) / β₁``. |
| `mle_loss(x, y_obs, beta0, beta1, sigma2_y, psi=None)` | Sum of negative log-likelihoods at scalar / array `x`. |
| `mle_grad_x(x, y_obs, beta0, beta1, sigma2_y)` | Analytic gradient of the linear-Gaussian NLL. |

## `estimators.map`

The MAP estimator adds a Gaussian prior to the MLE objective.

| Symbol | Role |
|---|---|
| `map_analytic_linear(y_obs, beta0, beta1, sigma2_y, m_x, s2_x)` | Closed-form precision-weighted average of MLE and prior mean. |
| `map_loss(x, y_obs, beta0, beta1, sigma2_y, m_x, s2_x, psi=None)` | Negative log-posterior up to a constant. |
| `map_grad_x(x, y_obs, beta0, beta1, sigma2_y, m_x, s2_x)` | Gradient of the MAP loss. |

## `estimators.gradient_descent`

Generic 1-D minimizer with optional analytic gradient (falls back to
centered finite differences when none is provided).

```python
from active_inference import gradient_descent

result = gradient_descent(
    loss_fn=lambda x: (x - 3.0) ** 2,
    x0=0.0,
    learning_rate=0.1,
    max_iter=200,
)
print(result.x, result.converged, result.n_iterations)
```

`GradientDescentResult` carries the final iterate, full iterate / loss
history, iteration count, and a `converged` flag based on the iterate-step
threshold.

## `estimators.linear_regression`

Vectorized linear regression for the design-matrix convention
`X ∈ ℝ^{N×C}` (a column of ones is prepended automatically when
`intercept=True`).

| Symbol | Role |
|---|---|
| `add_intercept(X)` | Prepend a column of ones to the design matrix. |
| `mle_linear_regression(X, y, intercept=True)` | Closed-form normal equation via `np.linalg.lstsq` (Moore–Penrose). |
| `GDRegressionResult` | dataclass carrying `theta`, `history`, `losses`, `n_iterations`, `converged`. Returned by `gd_linear_regression`. |
| `gd_linear_regression(X, y, learning_rate, ..., l2=0.0)` | Vectorized gradient descent with optional L2 regularization. |
| `squared_loss(theta, X, y, ..., l2=0.0)` | Loss value at a parameter vector. |
| `squared_loss_grad(theta, X, y, ..., l2=0.0)` | Analytic gradient of the squared loss. |

### Bayesian linear regression

```python
from active_inference import BayesianLinearRegression
import numpy as np

blr = BayesianLinearRegression(
    prior_mean=np.zeros(C + 1),
    prior_cov=np.eye(C + 1) * 4.0,
    sigma2_y=0.25,
)
posterior = blr.fit(X, y)
print(posterior.mean, posterior.std())

mean_pred, var_pred = posterior.predictive(X_new, sigma2_y=blr.sigma2_y)
```

`BayesianLinearRegression.fit_sequential(X, y)` yields `(i, BLRPosterior)`
after each row is assimilated — this is what powers
`animation_blr_tightening.py`.

## `estimators.em`

Expectation–Maximization for linear factor analysis. Standard-normal prior
on the latent states, diagonal observation noise, zero-centered data.

| Symbol | Role |
|---|---|
| `factor_analysis_e_step(Y, Theta, cov_y)` | Posterior mean (per row) + (shared) covariance. |
| `factor_analysis_m_step(Y, mu, cov)` | Updated `Θ` and diagonal noise covariance. |
| `incomplete_log_likelihood(Y, Theta, cov_y)` | Marginal `log p(Y)` under the current parameters. |
| `fit_factor_analysis(Y, n_factors, ...)` | Full EM loop with convergence diagnostics. |

`FactorAnalysisResult` holds the final loadings, posteriors, log-likelihood
trace, iteration count, convergence flag, and per-iteration history of `Θ`
and the noise diagonal — used by `animation_em_convergence.py`.
