# Bayesian inference

Bayes' theorem turns a prior belief about a hidden state into a posterior
belief after observing data. The four ingredients are the *prior*
``p(x)``, the *likelihood* ``p(y|x)``, the *posterior* ``p(x|y)``, and
the *evidence* ``p(y)`` that normalizes the product. In this codebase
the posterior is computed exactly on a 1-D grid (`GridBayesianInference`)
or in closed form for multivariate Gaussian setups (`LinearGaussianSystem`).

## In this codebase

- **Prior + likelihood definition:** any subclass of
  `core.generative_model.GenerativeModel`. The shipped implementations
  are `LinearGaussianModel` (univariate) and `LinearGaussianMVModel`
  (multivariate).
- **Exact 1-D inference:**
  `core.inference.GridBayesianInference(model, x_grid).infer(y)` →
  `InferenceResult` with `posterior`, `posterior_mode/mean/variance`,
  `credible_interval`, `cdf`, `quantile`, `entropy`, `kl_from_prior`,
  `log_evidence`.
- **Closed-form multivariate posterior:**
  `core.lgs.LinearGaussianSystem(...).posterior(y)` /
  `posterior_batch(Y)` → `LGSPosterior` with `mean`, `cov`, `std`,
  `precision`, `sample`.
- **Diagnostics on top of the result:**
  `core.diagnostics.grid_kl_divergence`, `grid_entropy`,
  `gaussian_kl_univariate`, `coverage_from_intervals`.

## End-to-end snippet

```python
from active_inference import (
    LinearGaussianModel, GridBayesianInference, make_grid,
)

model = LinearGaussianModel(
    beta0=3.0, beta1=2.0, sigma2_y=0.25,
    m_x=4.0, s2_x=0.25, prior_kind="gaussian",
)
result = GridBayesianInference(model, make_grid(0, 5, 500)).infer(y=7.0)
print(result.summary())
print("95% CI:", result.credible_interval(0.95))
print("KL[post||prior]:", result.kl_from_prior())
```

## Pitfalls

- Grid bounds matter. If the posterior has appreciable mass at the
  boundary, widen the grid or `infer` will warn / refuse.
- The likelihood used in the three-panel figure is *unnormalized* —
  it is a function of `x`, not a density. The posterior is the
  normalized product; the figure shows the likelihood on a credibility
  axis.
- Differential entropy is *not* bounded below by zero. Tight Gaussians
  on continuous domains have negative differential entropy — that's
  expected, not a bug.

## See also

- [`../chapters/chapter_01.md`](../chapters/chapter_01.md) — Bayes'
  theorem walk-through and the inverse problem.
- [`../chapters/chapter_02.md`](../chapters/chapter_02.md) — full
  recipe and the precision sweep.
- [`../statistics/divergences.md`](../statistics/divergences.md),
  [`../statistics/entropy.md`](../statistics/entropy.md) — diagnostics
  applied here.
- [`../reference/core.md`](../reference/core.md) — every public symbol
  in `core`.
