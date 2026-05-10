# Scoring rules

Scoring rules score a probabilistic forecast against a realized outcome.
A *proper* scoring rule is maximized in expectation by the true
distribution, so the average score over a held-out set tells you
whether your model is well-calibrated *and* well-specified.

## Definitions

### Log score (logarithmic score)

For a Gaussian forecast ``N(μ, σ²)`` and a realized outcome ``y``::

    log_score(y; μ, σ²) = log p(y; μ, σ²)
                       = − ½ log(2π σ²) − (y − μ)² / (2 σ²)

Higher is better. Equal in spirit to negative-NLL.

### Continuous Ranked Probability Score (CRPS)

For a Gaussian forecast (Gneiting & Raftery, 2007 closed form)::

    z   = (y − μ) / σ
    CRPS = σ [ z (2 Φ(z) − 1) + 2 φ(z) − 1/√π ]

where ``Φ`` is the standard-normal CDF and ``φ`` its PDF. **Lower** is
better. CRPS is non-negative and reduces to mean absolute error when
``σ → 0``.

## API

| Function | Signature | Notes |
|---|---|---|
| `log_score_gaussian` | `(y, mu, sigma2) -> array` | Pointwise; sum / mean as appropriate. |
| `crps_gaussian` | `(y, mu, sigma2) -> array` | Pointwise; uses `math.erf` via `np.vectorize`. |

Both live in `active_inference.core.diagnostics`.

## Tests that pin them down

`tests/core/test_diagnostics.py::TestScoringRules` covers:

- Log score is higher when the forecast mean is at the truth than when
  it is shifted away.
- CRPS is **lower** for the correct-mean forecast (consistent with the
  "lower is better" convention).
- CRPS is non-negative for arbitrary forecasts.

## Pitfalls

- The two scoring rules disagree on direction: log score is *higher*
  is better; CRPS is *lower* is better. Don't mix them in a single
  ranking without a sign flip.
- Both rules penalize miscalibrated *variance*, not just miscalibrated
  mean. A forecast that is right on the mean but overconfident in the
  variance will score worse than a wider, well-calibrated one.
- The closed-form CRPS used here assumes a Gaussian forecast. For
  ensembles or mixtures, CRPS has to be evaluated by the empirical
  formulation — that is not in the package.

## See also

- [`calibration.md`](calibration.md) — coverage of credible intervals.
- [`../topics/learning_and_inference.md`](../topics/learning_and_inference.md)
  — using log score / CRPS to compare estimators.
- [`../reference/core.md`](../reference/core.md) — full API.
