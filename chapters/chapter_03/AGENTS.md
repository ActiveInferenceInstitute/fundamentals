# `chapters/chapter_03/` — Combining Learning and Inference

Chapter 3 scripts cover the transition from "parameters are given" to
"parameters must be learned". Three threads run through them: linear
regression (MLE / MAP / Bayesian), multivariate hidden-state inference
via the Linear Gaussian System, and Expectation–Maximization for factor
analysis.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_3_1_linear_regression_mle.py`](example_3_1_linear_regression_mle.py) | ~110 | Closed-form linear regression at low N: many plausible θ hypotheses. |
| [`example_3_2_linear_regression_gd.py`](example_3_2_linear_regression_gd.py) | ~120 | Gradient descent on the (β₀, β₁) loss surface, verifies match to analytic. |
| [`example_3_3_multiple_regression.py`](example_3_3_multiple_regression.py) | ~95 | Vectorized multiple regression; estimation error vs N. |
| [`example_3_4_multivariate_gaussian.py`](example_3_4_multivariate_gaussian.py) | ~80 | Anatomy of the MVN: spherical / anisotropic / correlated covariances. |
| [`example_3_5_bayesian_linear_regression.py`](example_3_5_bayesian_linear_regression.py) | ~50 | Thin wrapper over the `orchestrator_workflows` BLR builder: posterior over (β₀, β₁) tightens with N; predictive band. |
| [`example_3_6_lgs_food_localization.py`](example_3_6_lgs_food_localization.py) | ~110 | Multivariate hidden-state inference for a 2-D location. |
| [`example_3_7_factor_analysis_em.py`](example_3_7_factor_analysis_em.py) | ~55 | Thin wrapper over the `orchestrator_workflows` FA builder: EM on synthetic data; loadings + noise variances recovered. |
| [`animation_blr_tightening.py`](animation_blr_tightening.py) | ~85 | GIF: 2-D BLR posterior tightening as observations arrive. |
| [`animation_blr_predictive_band.py`](animation_blr_predictive_band.py) | ~90 | GIF: predictive band shrinking with each new observation. |
| [`animation_em_convergence.py`](animation_em_convergence.py) | ~90 | GIF: marginal LL + loadings matrix evolving per EM iteration. |
| [`animation_em_steps.py`](animation_em_steps.py) | ~100 | GIF: E-step / M-step alternation of factor-analysis EM. |
| [`animation_lgs_online.py`](animation_lgs_online.py) | ~85 | GIF: online 2-D LGS posterior collapsing per sample. |
| [`animation_precision_sweep.py`](animation_precision_sweep.py) | ~80 | GIF: posterior shape as prior/likelihood precision varies. |
| [`animation_bimodal_emergence.py`](animation_bimodal_emergence.py) | ~80 | GIF: bi-modal posterior from a non-injective generator. |
| [`animation_sufficient_statistics.py`](animation_sufficient_statistics.py) | ~105 | GIF: running sufficient statistics over a Gaussian stream. |
| [`visualize_calibration.py`](visualize_calibration.py) | ~80 | Empirical-vs-nominal coverage curve for a BLR forecast. |
| [`visualize_coverage.py`](visualize_coverage.py) | ~80 | Empirical coverage of a fixed 95% LGS credible region across a sample-size (N) sweep. |
| [`visualize_posterior_predictive.py`](visualize_posterior_predictive.py) | ~85 | Posterior predictive check on regression residuals. |
| [`interactive_bayesian_regression.py`](interactive_bayesian_regression.py) | ~30 | **Interactive** (GUI / web-launchable): `N` / prior-precision sliders tighten the ±2σ posterior-predictive band; readout reports recovered `β0`/`β1` ± posterior std. |
| [`interactive_lgs_localization.py`](interactive_lgs_localization.py) | ~35 | **Interactive** (GUI / web-launchable): `(y1, y2)` observation sliders slide the posterior mean ellipse toward the fixed prior; readout reports posterior mean/std and distance from prior/observation. |

## Running

```bash
# Run a single script (headless — saves to output/figures/chapter_03/)
python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save

# Run every Chapter 3 script at once
python scripts/run_all_figures.py --chapters 3
./scripts/run_all_chapter_03.sh
```

Each script accepts `--save` for headless rendering; stochastic scripts also
accept `--seed` for reproducibility.

## Library Usage

Common imports across this chapter:

```python
from active_inference import (
    LinearGaussianProcess, LinearGaussianMVProcess,
    LinearGaussianSystem, BayesianLinearRegression,
    fit_factor_analysis,
    mle_linear_regression, gd_linear_regression,
    mvn_pdf, mvn_sample, isotropic_cov, diagonal_cov,
)
```

`interactive_bayesian_regression.py` is a thin wrapper around
`active_inference.visualizations.interactive_bayesian_regression`.
`interactive_lgs_localization.py` is a thin wrapper around
`active_inference.visualizations.interactive_lgs_localization`.

## Smoke Tests

`tests/chapters/test_smoke.py` runs each script via `subprocess` and
asserts exit code 0 (the single parametrized test
`test_chapter_script_runs_and_exports_raw_data`, run over every discovered
chapter script). `interactive_bayesian_regression.py` and `interactive_lgs_localization.py`
are skipped there (filename match on `interactive`) and exercised
separately by `tests/visualizations/test_interactive.py`.

## Key Concepts

- **Closed-form vs iterative.** Examples 3.1–3.3 show the analytic
  normal-equation solution; 3.2 verifies the gradient-descent path
  reaches the same point.
- **Bayesian linear regression** turns the parameters into random
  variables and produces a *distribution* over θ rather than a point
  estimate, naturally tightening as more data arrives.
- **Linear Gaussian System** generalizes the univariate Bayesian update
  to vector-valued states with a closed-form posterior.
- **EM for factor analysis** alternates an E-step (Bayesian update over
  the latent state) with an M-step (MLE for the loadings and noise
  variances), with a marginal log-likelihood that increases monotonically.
