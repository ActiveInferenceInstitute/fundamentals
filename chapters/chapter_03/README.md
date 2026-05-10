# Chapter 3 — Combining Learning and Inference

Chapter 3 connects parameter learning to hidden-state inference. The book
walks from supervised linear regression (deterministic θ) → Bayesian linear
regression (probabilistic θ) → multivariate hidden-state inference via the
Linear Gaussian System → factor analysis with EM, where θ and x are jointly
unknown.

This folder contains a thin orchestrator per concept, eight bonus
animations, and three diagnostic visualizations. Every script imports
configurable building blocks from `active_inference`.

## Scripts

### Numbered examples

| Script | Mirrors | What it shows |
|--------|---------|---------------|
| `example_3_1_linear_regression_mle.py`      | Example 3.1 | Closed-form linear regression at low N: many plausible θ hypotheses. |
| `example_3_2_linear_regression_gd.py`       | Example 3.2 | Gradient descent over the (β₀, β₁) loss surface. |
| `example_3_3_multiple_regression.py`        | Example 3.3 | Vectorized multiple regression via the normal equation. |
| `example_3_4_multivariate_gaussian.py`      | Example 3.4 | Anatomy of the MVN: covariance shapes, sampling, contours. |
| `example_3_5_bayesian_linear_regression.py` | Example 3.5 | Posterior over θ tightens with N; predictive bands shown. |
| `example_3_6_lgs_food_localization.py`      | Example 3.6 | Multivariate hidden-state inference for a 2-D food source. |
| `example_3_7_factor_analysis_em.py`         | §3.5        | EM loop on synthetic factor-analysis data with reconstruction. |

### Animations (GIFs)

| Script | What it shows |
|--------|---------------|
| `animation_blr_tightening.py`        | 2-D posterior over (β₀, β₁) tightening as data arrives. |
| `animation_blr_predictive_band.py`   | Predictive band shrinking with each new observation. |
| `animation_em_convergence.py`        | EM log-likelihood and loadings matrix evolving per iteration. |
| `animation_em_steps.py`              | Detailed E-step / M-step alternation of factor-analysis EM. |
| `animation_lgs_online.py`            | Online 2-D LGS posterior collapsing with each new sample. |
| `animation_precision_sweep.py`       | Posterior shape as prior / likelihood precision varies. |
| `animation_bimodal_emergence.py`     | Bi-modal posterior emerging from a non-injective generator. |
| `animation_sufficient_statistics.py` | Running sufficient statistics over a Gaussian stream. |

### Diagnostic visualizations

| Script | What it shows |
|--------|---------------|
| `visualize_calibration.py`          | Empirical-vs-nominal coverage curve for a BLR forecast. |
| `visualize_coverage.py`             | Coverage sweep across credible levels. |
| `visualize_posterior_predictive.py` | Posterior predictive check on regression residuals. |

## Running

```bash
# from the repo root
uv sync                                                        # one-time setup
uv run python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save

# or via the top-level menu
./run.sh --chapter 3
```

Pass `--save` to dump figures (and animations) into
`output/figures/chapter_03/`. Without it the scripts open interactive
windows.

## Programmatic usage

```python
from active_inference import (
    BayesianLinearRegression,
    LinearGaussianMVProcess,
    LinearGaussianSystem,
    fit_factor_analysis,
    isotropic_cov,
)
import numpy as np

rng = np.random.default_rng(0)
N, C = 100, 3
X = rng.normal(size=(N, C))
y = X @ np.array([1.0, -0.5, 0.7]) + 2.0 + rng.normal(scale=0.1, size=N)

blr = BayesianLinearRegression(
    prior_mean=np.zeros(C + 1),
    prior_cov=np.eye(C + 1) * 4.0,
    sigma2_y=0.01,
)
posterior = blr.fit(X, y)
print(posterior.mean, posterior.std())
```
