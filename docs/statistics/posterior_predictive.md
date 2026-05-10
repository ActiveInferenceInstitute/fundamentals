# Posterior predictive checks

A posterior predictive check (PPC) compares a chosen test statistic on
the observed data against the same statistic on data sets replicated
under the fitted posterior. If the model captures the data-generating
process, the observed value should look typical of the replicates.

## Definition

Given observations ``y`` and ``M`` replicate data sets
``y_rep^{(1)}, …, y_rep^{(M)}`` drawn from the posterior predictive,
the two-sided PPC p-value for a test statistic ``T`` is::

    p = 2 · min[ P(T(y_rep) ≥ T(y)),
                 P(T(y_rep) ≤ T(y)) ]

Small ``p`` means the observed statistic is in the tail of the
replicate distribution — evidence of model–data mismatch on the chosen
statistic.

## API

| Symbol | Signature | Notes |
|---|---|---|
| `posterior_predictive_check` | `(observed, replicated_samples, statistic) -> PosteriorPredictiveCheck` | `replicated_samples` is `(M, N)`; `statistic` is any callable on a 1-D array. |
| `PosteriorPredictiveCheck` | dataclass: `observed`, `replicated`, `p_value`, `summary()` |
| `plot_posterior_predictive_check` | `(check, label=..., save_path=...) -> Figure` | Histogram of replicates with the observed value marked and the in-figure p-value. |

`posterior_predictive_check` and the dataclass live in
`active_inference.core.diagnostics`; the figure helper is in
`active_inference.visualizations.diagnostics`.

## Tests that pin it down

`tests/core/test_diagnostics.py::TestPPC` covers:

- p-values lie in `[0, 2]` for any input (the two-sided definition).
- An observation shifted far outside the replicate distribution yields
  ``p < 0.05``.
- Replicate input must be ``(M, N)`` — anything else raises.

## End-to-end snippet

```python
from active_inference import (
    BayesianLinearRegression, posterior_predictive_check,
)
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
y = X @ np.array([1.0, -0.5]) + 0.5 + rng.normal(scale=0.3, size=200)

blr = BayesianLinearRegression(
    prior_mean=np.zeros(3), prior_cov=np.eye(3) * 4.0, sigma2_y=0.09,
)
posterior = blr.fit(X, y)

theta_samples = posterior.sample(n=500, rng=rng)
replicated = np.array([
    theta[0] + theta[1:] @ X.T + rng.normal(scale=0.3, size=200)
    for theta in theta_samples
])
check = posterior_predictive_check(y, replicated, statistic=np.std)
print(check.summary())
```

## Pitfalls

- The choice of test statistic matters enormously. Always check at
  least a *location* (mean), a *scale* (std / IQR), and a *tail*
  (range / max) statistic before declaring a model good.
- A two-sided p-value near ``1.0`` means the observed value sits at
  the median of the replicate distribution — that is *good*, not
  suspicious.
- Replicates must be drawn from the *posterior predictive*, not the
  *prior predictive*. Use parameter samples from `BLRPosterior.sample`.

## See also

- [`scoring_rules.md`](scoring_rules.md) — proper-scoring-rule
  alternative for ranking models.
- [`calibration.md`](calibration.md) — coverage of credible intervals,
  another sanity check.
- [`../chapters/chapter_03.md`](../chapters/chapter_03.md) — the
  `visualize_posterior_predictive.py` script.
- [`../reference/core.md`](../reference/core.md) — full API.
