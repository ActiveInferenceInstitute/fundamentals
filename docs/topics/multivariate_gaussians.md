# Multivariate Gaussians

The multivariate normal (MVN) is the workhorse for vector-valued
hidden-state inference, sensor fusion, and parameter posteriors. Every
MVN routine in this codebase is implemented with **Cholesky-based
solves** — never explicit covariance inversion — so the math stays
numerically stable even when the covariance is ill-conditioned.

## In this codebase

- **Density evaluation:** `core.distributions.mvn_pdf` (linear),
  `mvn_log_pdf` (log space, batched).
- **Sampling:** `core.distributions.mvn_sample(mu, cov, n=...)` — uses
  the same Cholesky factor as the density routines.
- **Quadratic form:** `core.distributions.mahalanobis_squared(x, mu, cov)`
  computes ``(x − μ)ᵀ Σ⁻¹ (x − μ)`` via a Cholesky solve, returning a
  scalar for a vector input or a length-N array for a batch.
- **Covariance constructors:** `isotropic_cov(d, var)` (spherical) and
  `diagonal_cov(variances)` (axis-aligned).
- **Confidence ellipses for plotting:**
  `visualizations.plotting.confidence_ellipse(mean, cov, n_std=2.0)`.
- **MVN-flavored generative classes:** `LinearGaussianMVProcess`,
  `LinearGaussianMVModel`, `LinearGaussianSystem`, `LGSPosterior`.

## End-to-end snippet

```python
import numpy as np
from active_inference import (
    isotropic_cov, mvn_pdf, mvn_sample, mahalanobis_squared,
)

rng = np.random.default_rng(0)
mu = np.array([1.0, -1.0])
cov = np.array([[1.0, 0.4], [0.4, 0.5]])

samples = mvn_sample(mu, cov, n=20_000, rng=rng)
print("empirical mean :", samples.mean(axis=0))
print("empirical cov  :", np.cov(samples, rowvar=False))
print("density at mu  :", mvn_pdf(mu, mu, cov))
print("Mahalanobis(0):", mahalanobis_squared(np.zeros(2), mu, cov))
```

## Pitfalls

- Covariance matrices must be **symmetric and positive definite**.
  All constructors validate this; if you build one by hand and pass it
  in, expect a `ValueError` if it isn't.
- The MVN log-PDF works on either a single vector ``(D,)`` or a batch
  ``(N, D)``. The output shape mirrors the input.
- A Mahalanobis distance squared of ``c`` corresponds to a
  ``χ²_d``-quantile credible region — for ``d = 2`` and 95 % coverage
  use `chi2_95 = -2 * np.log(0.05) ≈ 5.991`.
- For diagonal covariances, prefer `diagonal_cov` over `np.diag(...)` —
  the helper validates positivity and is clearer at the call site.

## See also

- [`generative_models.md`](generative_models.md) — where MVN slots in.
- [`bayesian_inference.md`](bayesian_inference.md) — closed-form MVN
  posteriors.
- [`../statistics/divergences.md`](../statistics/divergences.md) — KL
  between MVNs.
- [`../reference/core.md`](../reference/core.md) — full API.
