# Generative models

A *generative model* is a joint distribution ``p(x, y) = p(y | x) p(x)``
that the agent uses to invert observations and recover the hidden state
that produced them. The package keeps a strict split between the
**generative process** (the environment, sample-only) and the
**generative model** (the agent's beliefs, density-aware).

## In this codebase

- **Process side** (samples observations from a chosen generator):
  - `core.generative_process.GenerativeProcess` — abstract base.
  - `LinearGaussianProcess` — univariate, optional nonlinear `psi`.
  - `LinearGaussianMVProcess` — multivariate `y = Θ x* + b + ω`.
- **Model side** (defines log-likelihood + log-prior):
  - `core.generative_model.GenerativeModel` — abstract base with
    `log_likelihood`, `log_prior`, `predict_mean`.
  - `LinearGaussianModel` — Gaussian likelihood, Gaussian or uniform
    prior, optional nonlinear `psi`.
  - `LinearGaussianMVModel` — multivariate Gaussian on both ends.

The model side never *samples* observations; the process side never
*evaluates* densities. This keeps the abstraction sharp and prevents
chapter scripts from accidentally computing posteriors against samples
they actually meant to be ground truth.

## End-to-end snippet

```python
import numpy as np
from active_inference import (
    LinearGaussianProcess, LinearGaussianModel,
    GridBayesianInference, make_grid,
)

# 1. Environment (sample-only)
process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                rng=np.random.default_rng(0))
y = process.sample(x_star=2.0, n=30).flatten()

# 2. Agent (density-aware)
model = LinearGaussianModel(
    beta0=3.0, beta1=2.0, sigma2_y=0.25,
    m_x=4.0, s2_x=0.25, prior_kind="gaussian",
)

# 3. Inference
result = GridBayesianInference(model, make_grid(0, 5, 500)).infer(y)
print(result.summary())
```

## Pitfalls

- The agent's parameters do **not** have to match the process's. The
  Example 2.6 script intentionally mismatches them to demonstrate
  systematic bias.
- `predict_mean(x)` returns the *noiseless* mean of the likelihood, not
  a sample.
- For multivariate models, validate at construction time — covariance
  matrices must be square, symmetric, and positive-definite.

## See also

- [`bayesian_inference.md`](bayesian_inference.md) — the inversion step.
- [`multivariate_gaussians.md`](multivariate_gaussians.md) — the
  multivariate machinery underpinning the MV classes.
- [`../chapters/chapter_02.md`](../chapters/chapter_02.md) — Examples
  2.1–2.10 that exercise these classes.
- [`../reference/core.md`](../reference/core.md) — full API.
