# `core/` — mathematical primitives

Distributions, generative process / model classes, exact grid Bayesian
inference, and closed-form Linear Gaussian System posteriors. Everything in
the rest of the package builds on this layer.

## Files

| File | What it defines |
|---|---|
| [`distributions.py`](distributions.py) | Univariate (`gaussian_pdf`, `gaussian_log_pdf`, `uniform_pdf`, `dirac_like_pdf`, `normalize_density`) and multivariate (`mvn_pdf`, `mvn_log_pdf`, `mvn_sample`, `mahalanobis_squared`, `isotropic_cov`, `diagonal_cov`) helpers. |
| [`generative_process.py`](generative_process.py) | `GenerativeProcess` (abstract), `LinearGaussianProcess` (univariate), `LinearGaussianMVProcess` (multivariate). |
| [`generative_model.py`](generative_model.py) | `GenerativeModel` (abstract), `LinearGaussianModel` (univariate), `LinearGaussianMVModel` (multivariate). |
| [`inference.py`](inference.py) | `GridBayesianInference` + `InferenceResult` — exact posterior on a 1-D grid with trapezoid normalization. |
| [`lgs.py`](lgs.py) | `LinearGaussianSystem` + `LGSPosterior` — closed-form multivariate posterior. |
| `__init__.py` | Re-exports the public surface. |

## Public API

```python
from active_inference.core.distributions import (
    gaussian_pdf, gaussian_log_pdf, uniform_pdf, dirac_like_pdf,
    mvn_pdf, mvn_log_pdf, mvn_sample, mahalanobis_squared,
    isotropic_cov, diagonal_cov, normalize_density,
)
from active_inference.core.generative_process import (
    GenerativeProcess, LinearGaussianProcess, LinearGaussianMVProcess,
)
from active_inference.core.generative_model import (
    GenerativeModel, LinearGaussianModel, LinearGaussianMVModel,
)
from active_inference.core.inference import GridBayesianInference, InferenceResult
from active_inference.core.lgs import LinearGaussianSystem, LGSPosterior
```

## Design Decisions

- **Vectorized.** Shapes are designed so that any of `x`, `mu`, or `sigma2`
  may broadcast — crucial for grid-based inference.
- **Variances everywhere.** Code uses `sigma2`, `s2_x`, `cov_y` (variance /
  covariance), never standard deviations.
- **Cholesky for MVN.** Multivariate density and sampling use Cholesky-based
  solves rather than explicit matrix inverses.
- **Explicit RNG.** Random generators are passed via `rng: np.random.Generator`.
- **Log-space inference.** `GridBayesianInference` works in log-space and
  subtracts the max log-density before exponentiating to avoid under/overflow.
- **Result dataclasses.** `InferenceResult` and `LGSPosterior` provide
  computed properties (mode, mean, variance, credible interval, std,
  precision) so downstream code never reimplements them.

## Dependencies

`numpy` (everywhere) + `scipy.linalg.solve_triangular` (only inside
`mvn_log_pdf` for stability).

## Testing

See `tests/core/test_distributions.py`, `test_distributions_mvn.py`,
`test_generative_process.py`, `test_generative_model.py`,
`test_inference.py`, and `test_lgs.py`.
