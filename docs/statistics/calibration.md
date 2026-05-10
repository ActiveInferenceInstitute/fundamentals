# Calibration

A model is *calibrated* if a credible interval at nominal mass
``α`` actually contains the truth ``α``-of-the-time across many
trials. Calibration is the most basic sanity check on any
probabilistic prediction; failure usually points to either a bad
prior, an under-dispersed likelihood, or a model–data mismatch.

## Definitions

### Empirical coverage

For a fixed credible mass ``α`` and a batch of trials each with truth
``x_t*`` and credible interval ``[ℓ_t, h_t]``::

    coverage(α) = (1 / T) Σ_t  𝟙[ x_t* ∈ [ℓ_t, h_t] ]

If the model is calibrated, ``coverage(α) ≈ α``.

### Calibration curve

The function ``α → coverage(α)`` over a sweep of nominal levels. A
calibrated model produces the diagonal; deviations diagnose
over- / under-confidence.

### Expected calibration error (ECE)

L₁ deviation from the diagonal::

    ECE = (1 / L) Σ_ℓ |coverage(α_ℓ) − α_ℓ|

Smaller is better; ``ECE = 0`` means perfect calibration on the
sampled levels.

## API

| Symbol | Signature | Notes |
|---|---|---|
| `coverage_from_intervals` | `(truths, lows, highs) -> float` | Pointwise hit rate; raises on shape mismatch. |
| `calibration_curve` | `(truths, lower_fn, upper_fn, nominal_levels) -> CalibrationCurve` | Sweeps nominal levels; both `lower_fn` and `upper_fn` take a level and return a per-trial array. |
| `CalibrationCurve` | dataclass with `nominal`, `empirical`, `n_trials`, `calibration_error()` |
| `plot_calibration` | `(curve, ...) -> Figure` | Reliability diagram with the in-figure ECE. |
| `animate_calibration_growth` | `(nominal, empirical_history, ...) -> FuncAnimation` | Reliability diagram filling in trial-by-trial. |

`coverage_from_intervals`, `calibration_curve`, and `CalibrationCurve`
live in `active_inference.core.diagnostics`. The plotting helpers live
in `active_inference.visualizations.diagnostics` and `animations`.

## Tests that pin it down

- `tests/core/test_diagnostics.py::TestCalibration::test_perfect_calibration_for_oracle`
  — perfect coverage when the credible interval is centered on the truth.
- `tests/core/test_diagnostics.py::TestCalibration::test_well_calibrated_predictions`
  — ``ECE < 0.05`` when the model matches the data-generating process.
- `tests/estimators/test_recovery.py::TestBLRRecovery::test_posterior_predictive_calibration`
  — empirical 90 % predictive interval covers between 85 % and 95 % of
  the held-out data.

## Pitfalls

- An L₁ ECE near zero is necessary but not sufficient — a model can be
  marginally calibrated yet locally biased. Plot the per-level
  empirical points before trusting the summary.
- Coverage is a frequency claim and needs many trials. ``T = 200`` is
  the minimum for stable curves at the 0.05 levels.
- Predictive variance is an *additive* combination of parameter
  uncertainty and irreducible noise. A predictive interval can never be
  tighter than ``σ_y``.

## See also

- [`scoring_rules.md`](scoring_rules.md) — log-score / CRPS penalize
  miscalibrated forecasts, complementary to coverage.
- [`../topics/learning_and_inference.md`](../topics/learning_and_inference.md)
  — how Bayesian linear regression produces calibrated intervals.
- [`../chapters/chapter_03.md`](../chapters/chapter_03.md) — the
  `visualize_calibration.py` script.
- [`../reference/core.md`](../reference/core.md) — full API.
