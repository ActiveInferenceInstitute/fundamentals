# Learning and inference

When the parameters that govern the likelihood are not known in advance,
the agent has to *learn* them from data — usually before, sometimes
jointly with, hidden-state inference. The codebase ships four
learning paths, each picked according to whether the parameters are
treated as deterministic (MLE / OLS / GD) or as random variables
(Bayesian linear regression, EM).

## In this codebase

- **MLE / MAP for a scalar hidden state:**
  `estimators.mle.mle_analytic_linear`,
  `estimators.map.map_analytic_linear`, plus matching `*_loss` and
  `*_grad_x` for gradient-based variants.
- **OLS / vectorized regression:**
  `estimators.linear_regression.mle_linear_regression` (closed form via
  `np.linalg.lstsq`), `gd_linear_regression` (gradient descent with
  optional L2).
- **Bayesian linear regression:**
  `estimators.linear_regression.BayesianLinearRegression` with `fit`,
  `fit_sequential`, and `predictive`. Returns a `BLRPosterior`
  carrying `mean`, `cov`, `predict`, `sample`, `summary`.
- **EM for factor analysis:**
  `estimators.em.fit_factor_analysis(Y, n_factors=...)` returns a
  `FactorAnalysisResult` with `posterior_means`, `Theta`, `cov_y`,
  `log_likelihoods`, `predict_latent`, `reconstruct`.
- **Generic 1-D minimizer used by everything else:**
  `estimators.gradient_descent.gradient_descent`.

## End-to-end snippet

```python
import numpy as np
from active_inference import (
    BayesianLinearRegression, fit_factor_analysis, mle_linear_regression,
)

rng = np.random.default_rng(0)
N, C = 200, 3

# Linear regression — closed form.
X = rng.normal(size=(N, C))
y = X @ np.array([1.0, -0.5, 0.7]) + 2.0 + rng.normal(scale=0.1, size=N)
theta = mle_linear_regression(X, y)

# Bayesian linear regression — full posterior over θ.
blr = BayesianLinearRegression(
    prior_mean=np.zeros(C + 1), prior_cov=np.eye(C + 1) * 4.0,
    sigma2_y=0.01,
)
posterior = blr.fit(X, y)
print(posterior.summary())

# Factor analysis — joint learning of loadings and noise via EM.
Y = rng.normal(size=(400, 5))
result = fit_factor_analysis(Y, n_factors=2, rng=rng)
print(result.summary())
```

## Pitfalls

- Gradient descent on linear regression diverges if
  `learning_rate * λ_max(XᵀX) ≥ 2`. The defaults in chapter scripts
  are conservative; tighten only if your design matrix is
  well-scaled.
- BLR predictive variance always includes the irreducible noise
  ``σ²_y``: predictive intervals therefore *cannot* be tighter than
  the noise level no matter how much data arrives.
- Factor analysis is identifiable only up to a rotation. Compare
  *subspaces* (via the QR singular values) rather than raw loadings.

## See also

- [`gradient_descent.md`](gradient_descent.md) — the underlying optimizer.
- [`generative_models.md`](generative_models.md) — what is being
  parameterized.
- [`../chapters/chapter_03.md`](../chapters/chapter_03.md) — Examples
  3.1–3.7 that exercise every recipe here.
- [`../reference/estimators.md`](../reference/estimators.md) — full API.
