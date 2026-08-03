# `tests/core/` — tests for `src/active_inference/core/`

One test file per source module, plus a dedicated split for the
multivariate distributions because they are large enough to deserve their
own file.

| Source file | Test file |
|---|---|
| `core/distributions.py` (univariate helpers) | [`test_distributions.py`](test_distributions.py) |
| `core/distributions.py` (multivariate helpers) | [`test_distributions_mvn.py`](test_distributions_mvn.py) |
| `core/generative_process.py` | [`test_generative_process.py`](test_generative_process.py) |
| `core/generative_model.py` | [`test_generative_model.py`](test_generative_model.py) |
| `core/inference.py` | [`test_inference.py`](test_inference.py) |
| `core/lgs.py` | [`test_lgs.py`](test_lgs.py) |
| `core/variational.py` (Ch.4) | [`test_variational.py`](test_variational.py) |
| `core/thermodynamics.py` | [`test_thermodynamics.py`](test_thermodynamics.py) |
| `core/predictive_coding.py` (Ch.5) | [`test_predictive_coding.py`](test_predictive_coding.py) |
| `core/generalized_filtering.py` (Ch.6) | [`test_generalized_filtering.py`](test_generalized_filtering.py) |
| `core/active_inference.py` (Ch.7) | [`test_active_inference.py`](test_active_inference.py) |
| `core/continuous_learning.py` (Ch.8) | [`test_continuous_learning.py`](test_continuous_learning.py) |
| `core/pomdp.py` (Ch.9–10) | [`test_pomdp.py`](test_pomdp.py) |
| `core/pomdp_extensions.py` (Ch.11) | [`test_pomdp_extensions.py`](test_pomdp_extensions.py) |
| `core/diagnostics.py` | [`test_diagnostics.py`](test_diagnostics.py) |
| `core/free_energy_forms.py` | [`test_free_energy_forms.py`](test_free_energy_forms.py) |
| `core/free_energy_forms.py` (Appendix D decomposition forms) | [`test_free_energy_appendix_forms.py`](test_free_energy_appendix_forms.py) |
| `core/factor_graph.py` | [`test_factor_graph.py`](test_factor_graph.py) |
| `core/ergodic.py` | [`test_ergodic.py`](test_ergodic.py) |
| `core/bayesian_mechanics.py` (Ch.14) | [`test_bayesian_mechanics.py`](test_bayesian_mechanics.py) |
| `core/appendix_math.py` (Appendix B/C) | [`test_appendix_math.py`](test_appendix_math.py) |
| `core/model_comparison.py` (Appendix C.11) | [`test_model_comparison.py`](test_model_comparison.py) |
| `core/noise.py` (Appendix C.9) | [`test_noise.py`](test_noise.py) |
| `core/compose.py` | [`test_compose.py`](test_compose.py) |
| `core/posterior.py` | [`test_posterior.py`](test_posterior.py) |
| `core/validators.py` | [`test_validators.py`](test_validators.py) |
| `core/types.py` | [`test_types.py`](test_types.py) |
| Appendix B–D + Part III helpers (cross-cutting) | [`test_appendix_part_iii_negative_controls.py`](test_appendix_part_iii_negative_controls.py) |

## Running

```bash
# all core tests
pytest tests/core -v

# a single module
pytest tests/core/test_lgs.py -v
```

## What's covered

- Density helpers normalize correctly via trapezoid integration.
- Log-PDF matches the exponential of the PDF (numerical stability).
- Generative processes recover known means and covariances from samples.
- Generative models validate input shapes and raise on bad inputs.
- `GridBayesianInference` posterior matches sequential Bayesian updating.
- `LinearGaussianSystem.posterior_batch` recovers the truth as N grows.
- Variational free energy's five forms agree and the bound `𝓕 ≥ −log p(y)`
  holds, tight at the posterior (Ch.4).
- Thermodynamic bridge helpers validate canonical probabilities, entropy,
  free-energy potentials, and the `T=1,pV=0` equality with VFE.
- Predictive coding's analytic gradient matches finite differences and its
  linear fixed point equals the grid posterior mean (Ch.5).
- Free-energy form helpers, factor-graph message updates, and ergodic-density
  helpers validate algebraic decompositions, normalization, and error handling.
- Diagnostics (calibration, CRPS, KL/entropy, `gradient_check`,
  `convergence_report`, `oracle_agreement`), compose pipelines, the posterior
  protocol, validators, and type asserts.
