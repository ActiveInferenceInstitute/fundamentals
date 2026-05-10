# `active_inference.core` — module reference

The `core` subpackage holds the mathematical primitives every other layer
builds on: probability densities, generative-process and generative-model
classes, exact grid-based inference, and closed-form Linear Gaussian System
posteriors.

## `core.distributions`

Numerically stable density helpers. Univariate functions broadcast over
NumPy arrays; multivariate functions accept either a single vector or a
batch of row-vectors and use Cholesky-based solves rather than explicit
matrix inverses.

| Symbol | Signature | Purpose |
|---|---|---|
| `gaussian_pdf(x, mu, sigma2)` | scalar/array → array | Univariate normal density. |
| `gaussian_log_pdf(x, mu, sigma2)` | scalar/array → array | Log of `gaussian_pdf`, computed without exponentiating first. |
| `uniform_pdf(x, low, high)` | scalar/array → array | Box density on `[low, high]`. |
| `dirac_like_pdf(x, location, epsilon)` | array → array | Narrow Gaussian standing in for a delta. |
| `normalize_density(values, grid)` | array → array | Trapezoid-rule normalization on a 1-D grid. |
| `mvn_pdf(x, mu, cov)` | (D,) or (N, D) → scalar/array | MVN density. |
| `mvn_log_pdf(x, mu, cov)` | (D,) or (N, D) → scalar/array | Log MVN. |
| `mvn_sample(mu, cov, n, rng)` | (D,) → (N, D) | Cholesky-based sampler. |
| `mahalanobis_squared(x, mu, cov)` | (D,) or (N, D) → scalar/array | Squared Mahalanobis distance. |
| `isotropic_cov(d, var)` | int, float → (d, d) | Spherical covariance `var · I`. |
| `diagonal_cov(variances)` | (d,) → (d, d) | Build a diagonal covariance from per-channel variances. |

## `core.diagnostics`

Statistical diagnostics. The full statistical-tool reference lives in
[`../statistics/`](../statistics/); this section is the API table.

| Symbol | Signature | Purpose |
|---|---|---|
| `logsumexp(a, axis=None)` | array → scalar/array | Numerically stable `log(sum(exp(a)))`. |
| `effective_sample_size(log_weights)` | (N,) → float | Kish ESS in log space. |
| `grid_entropy(p, x_grid)` | (G,), (G,) → float | Trapezoid differential entropy. |
| `grid_kl_divergence(p, q, x_grid)` | (G,), (G,), (G,) → float | Trapezoid `KL(p ‖ q)`. |
| `gaussian_entropy_univariate(sigma2)` | float → float | Closed form Gaussian H. |
| `gaussian_entropy_mvn(cov)` | (d, d) → float | MVN entropy via Cholesky log-det. |
| `gaussian_kl_univariate(mu_p, s2_p, mu_q, s2_q)` | floats → float | Closed form Gaussian KL. |
| `gaussian_kl_mvn(mu_p, cov_p, mu_q, cov_q)` | vectors + matrices → float | Closed form MVN KL. |
| `log_score_gaussian(y, mu, sigma2)` | arrays → array | Pointwise log score (higher is better). |
| `crps_gaussian(y, mu, sigma2)` | arrays → array | Pointwise CRPS (lower is better). |
| `coverage_from_intervals(truths, lows, highs)` | arrays → float | Empirical coverage of a fixed-mass CI. |
| `calibration_curve(truths, lower_fn, upper_fn, levels)` | arrays + 2 callables → `CalibrationCurve` | Reliability sweep. |
| `CalibrationCurve` | dataclass | `nominal`, `empirical`, `n_trials`, `calibration_error()`. |
| `posterior_predictive_check(observed, replicates, statistic)` | arrays + callable → `PosteriorPredictiveCheck` | Two-sided p-value. |
| `PosteriorPredictiveCheck` | dataclass | `observed`, `replicated`, `p_value`, `summary()`. |
| `normal_ci(mean, sigma2, level)` | floats → (lo, hi) | Equal-tailed Gaussian CI via `scipy.special.erfinv`. |
| `standardize(samples)` | (N, D) → (N, D) | Columnwise z-score with ddof=1. |

## `core.generative_process`

Sample-only generative processes — the *environment*. A process exposes
`mean(x_star)` and `sample(x_star, n)` only; it does not assess probabilities.

| Symbol | Role |
|---|---|
| `GenerativeProcess` | Generic scalar process: pass an arbitrary callable + parameters. |
| `LinearGaussianProcess` | `y = β₀ + β₁ ψ(x*) + ω` with optional nonlinear `psi`. |
| `LinearGaussianMVProcess` | Multivariate `y = Θ x* + b + ω` with full covariance. |

## `core.generative_model`

Agent-side models. Each exposes `log_likelihood`, `log_prior`, and (where
applicable) `predict_mean` so that downstream inference can run on a grid
or in closed form.

| Symbol | Role |
|---|---|
| `GenerativeModel` | Abstract base — subclasses override `log_likelihood` / `log_prior`. |
| `LinearGaussianModel` | Univariate Gaussian likelihood + Gaussian or uniform prior; optional `psi`. |
| `LinearGaussianMVModel` | Multivariate Gaussian likelihood + Gaussian prior. |

The univariate model also offers `log_likelihood_batch` (sum of per-sample
log-likelihoods, with broadcasting) and `likelihood_deterministic` (the
Dirac-like proxy used for the Example 2.1 demonstration).

## `core.inference`

Exact 1-D Bayesian inference via grid + trapezoid integration.

```python
from active_inference import LinearGaussianModel, GridBayesianInference, make_grid

model  = LinearGaussianModel(beta0=3, beta1=2, sigma2_y=0.25, m_x=4, s2_x=0.25)
grid   = make_grid(0, 5, 500)
result = GridBayesianInference(model, grid).infer(7.0)

print(result.posterior_mode)
print(result.credible_interval(0.95))
```

`InferenceResult` exposes:

- `x_grid`, `prior`, `likelihood`, `posterior`, `log_evidence`
- `posterior_mode`, `posterior_mean`, `posterior_variance`
- `credible_interval(mass)`

`GridBayesianInference.joint_density(y_grid)` returns `(x, y, p(x, y))` for
heat-map / 3-D visualizations of the joint distribution.

## `core.posterior` — cross-cutting Posterior protocol

A structural protocol implemented by every posterior dataclass in the
package, plus uniform accessors that dispatch on which interface the
posterior actually exposes.

| Symbol | Signature | Purpose |
|---|---|---|
| `Posterior` | `Protocol` (runtime-checkable) | Anything with `summary(ndigits)` matches. |
| `has_credible_interval(p)` | object → bool | True for 1-D grid posteriors. |
| `has_mean_cov(p)` | object → bool | True for Gaussian posteriors. |
| `posterior_mean(p)` | object → scalar / vector | Uniform mean accessor. |
| `posterior_std(p)` | object → scalar / vector | Uniform std accessor. |
| `summarize_posterior(p, ndigits=4)` | object → str | One-line readout. |

## `core.types` — shape aliases + safe-cast helpers

| Symbol | Purpose |
|---|---|
| `Vector`, `Matrix`, `Grid1D`, `DesignMatrix`, `CovMatrix`, `Probabilities`, `LogProb` | NumPy-array aliases for documenting shapes. |
| `assert_cov(cov, dim)` | Validate square + symmetric + positive-definite. |
| `assert_probabilities(p, tol)` | Validate non-negative + sums to 1. |

## `core.validators` — defensive runtime checks

Consolidated input validators used at module / class boundaries. Every
validator returns the (coerced) input on success so it can be chained
inline.

| Symbol | Purpose |
|---|---|
| `require_positive_scalar(x, name)` | ``x > 0`` and finite. |
| `require_non_negative_scalar(x, name)` | ``x ≥ 0`` and finite. |
| `require_in_unit_interval(x, inclusive)` | ``x ∈ (0, 1)`` (or ``[0, 1]``). |
| `require_int_at_least(x, minimum)` | Integer ≥ minimum. |
| `require_finite_array(arr, name)` | All entries finite. |
| `require_1d(arr, length)` | 1-D, optional length. |
| `require_2d(arr, shape)` | 2-D, optional shape. |
| `require_same_length(*arrays, names)` | Matching leading dimensions. |
| `require_monotone(arr, increasing, strict)` | Sorted (strict or non-strict). |
| `require_design_matrix(X, n_features, n_samples)` | Regression / FA design matrix shape + finiteness. |

## `core.compose` — pipelines and running statistics

| Symbol | Signature | Purpose |
|---|---|---|
| `Pipeline` | dataclass `(process, model, x_grid)` | Bundles a generative process, generative model, and inference grid; exposes `sample`, `infer`, `run`. |
| `Pipeline.linear_gaussian(...)` | classmethod | Pre-configured Pipeline for the canonical linear-Gaussian setup; defaults match Chapters 1–3. |
| `running_stats(model, x_grid, samples)` | `(model, (G,), (N,)) → RunningPosteriorStats` | One-pass per-step posterior moments + KL + cumulative log-evidence. |
| `RunningPosteriorStats` | dataclass | `n_axis`, `means`, `stds`, `kl_from_prior`, `log_evidences`, `posteriors`, `summary()`. |

```python
from active_inference import Pipeline, running_stats

pipe = Pipeline.linear_gaussian(
    beta0=3.0, beta1=2.0, sigma2_y=0.4, m_x=4.0, s2_x=1.0,
)
ys = pipe.sample(x_star=2.0, n=80).flatten()
stats = running_stats(pipe.model, pipe.x_grid, ys)
print(stats.summary())
```

## `core.lgs` — Linear Gaussian System

When prior and likelihood are both Gaussian, the posterior is Gaussian and
its mean / covariance have closed forms.

```python
from active_inference import LinearGaussianSystem, isotropic_cov
import numpy as np

lgs = LinearGaussianSystem(
    Theta=np.eye(2),
    cov_y=isotropic_cov(2, 0.1),
    mx=np.array([0.5, 0.5]),
    cov_x=isotropic_cov(2, 1.0),
)
posterior = lgs.posterior_batch(Y)   # Y shape (N, D)
print(posterior.mean, posterior.std())
```

`LGSPosterior` holds `mean`, `cov`, plus computed `std()` / `precision`.

## Conventions

- Variances and covariances are *variances*, never standard deviations.
- 1-D inputs default to scalar return; 2-D inputs broadcast row-wise.
- Random number generators are passed in explicitly (`numpy.random.Generator`).
- Internal log-space arithmetic is used wherever it improves numerical
  stability (e.g., subtracting the max log-density before exponentiation).
