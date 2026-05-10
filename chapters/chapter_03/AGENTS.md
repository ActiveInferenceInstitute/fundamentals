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
| [`example_3_5_bayesian_linear_regression.py`](example_3_5_bayesian_linear_regression.py) | ~130 | Posterior over (β₀, β₁) tightens with N; predictive band. |
| [`example_3_6_lgs_food_localization.py`](example_3_6_lgs_food_localization.py) | ~110 | Multivariate hidden-state inference for a 2-D location. |
| [`example_3_7_factor_analysis_em.py`](example_3_7_factor_analysis_em.py) | ~125 | EM on synthetic FA data; loadings + noise variances recovered. |
| [`animation_blr_tightening.py`](animation_blr_tightening.py) | ~75 | GIF: 2-D BLR posterior tightening as observations arrive. |
| [`animation_em_convergence.py`](animation_em_convergence.py) | ~75 | GIF: marginal LL + loadings matrix evolving per EM iteration. |

## Running

```bash
# Run a single script (headless — saves to output/figures/chapter_03/)
python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save

# Run every Chapter 3 script at once
python scripts/run_all_figures.py --chapters 3
./scripts/run_all_chapter_03.sh
```

Each script accepts `--save` for headless rendering and `--seed` for
reproducibility.

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

## Smoke Tests

`tests/chapters/test_smoke.py` runs each script via `subprocess` and
asserts exit code 0 (see `test_chapter_3_scripts_run` and
`test_chapter_3_animations`).

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
