# Cookbook — copy-paste recipes

Short, runnable snippets for the most-used workflows in the package. Every
recipe assumes you've created the environment with::

    uv sync                  # recommended; reads uv.lock
    # — or —
    pip install -e ".[dev]"  # plain-pip fallback

and works with no further setup. Wrap any command with ``uv run`` to
invoke it inside the project's virtualenv without activating it manually.

> **Learning by hand:** this repo has no LLM stage — the understanding comes
> from running these recipes and changing a parameter in each, then re-running
> and watching the figure/data change. For the full deliberate-practice loop
> (read the orchestrator → read the `src/` logic → run → modify → map to the
> book symbol → lock in with a test), see [`learn_by_hand.md`](learn_by_hand.md).

## 1. End-to-end Bayesian inference in 6 lines

```python
from active_inference import Pipeline

pipe = Pipeline.linear_gaussian(
    beta0=3.0, beta1=2.0, sigma2_y=0.4,
    m_x=4.0, s2_x=1.0,
)
result = pipe.run(x_star=2.0, n=30)
print(result.summary())                 # mode, mean, std, H, KL, log evidence
```

`result.credible_interval(0.95)` and `result.kl_from_prior()` are right
there on the same object. See [`topics/bayesian_inference.md`](topics/bayesian_inference.md).

## 2. Sequential update with running statistics

```python
import numpy as np
from active_inference import Pipeline, running_stats

pipe = Pipeline.linear_gaussian(beta0=3, beta1=2, sigma2_y=0.4,
                                m_x=4, s2_x=1.0,
                                rng=np.random.default_rng(0))
ys = pipe.sample(x_star=2.0, n=80).flatten()
stats = running_stats(pipe.model, pipe.x_grid, ys)
print(stats.summary())                  # final mean, std, KL, log p(y)
# stats.posteriors → (N, G) full density per step (good for animations)
```

See [`topics/active_inference.md`](topics/active_inference.md) for how
this trace maps onto the agent loop, and
[`statistics/effective_sample_size.md`](statistics/effective_sample_size.md)
for the log-space stability primitives the trace relies on.

## 3. Bayesian linear regression with predictive bands

```python
import numpy as np
from active_inference import BayesianLinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
y = X @ np.array([1.0, -0.5]) + 0.5 + rng.normal(scale=0.3, size=200)

blr = BayesianLinearRegression(
    prior_mean=np.zeros(3), prior_cov=np.eye(3) * 4.0, sigma2_y=0.09,
)
posterior = blr.fit(X, y)
mean_pred, var_pred = posterior.predictive(X, sigma2_y=blr.sigma2_y)
print(posterior.summary())              # parameter posterior in one line
```

`posterior.sample(n=...)` draws parameter samples for posterior-predictive
checks. See [`topics/learning_and_inference.md`](topics/learning_and_inference.md).

## 4. Multivariate hidden-state inference (LGS sensor fusion)

```python
import numpy as np
from active_inference import LinearGaussianSystem, isotropic_cov

lgs = LinearGaussianSystem(
    Theta=np.eye(2),
    cov_y=isotropic_cov(2, 0.05),
    mx=np.array([0.5, 0.5]),
    cov_x=isotropic_cov(2, 1.0),
)
Y = np.array([[0.4, 0.6], [0.42, 0.58], [0.38, 0.61]])
post = lgs.posterior_batch(Y)
print(post.summary())                   # mean + std vectors
print(post.cov)
```

See [`topics/multivariate_gaussians.md`](topics/multivariate_gaussians.md).

## 5. Factor analysis via EM

```python
import numpy as np
from active_inference import diagonal_cov, fit_factor_analysis, mvn_sample

rng = np.random.default_rng(0)
true_Theta = np.array([[1.0, 0.5], [0.7, -0.2], [-0.3, 1.0],
                       [0.4, 0.4], [0.0, 0.9]])
X_latent = mvn_sample(np.zeros(2), np.eye(2), n=400, rng=rng)
noise = mvn_sample(np.zeros(5), diagonal_cov([0.10, 0.20, 0.05, 0.30, 0.15]),
                   n=400, rng=rng)
Y = X_latent @ true_Theta.T + noise

fit = fit_factor_analysis(Y, n_factors=2, max_iter=200, tol=1e-6, rng=rng)
print(fit.summary())                    # iter count, final LL, converged?
print(fit.predict_latent(Y[:5]))        # E-step posterior means for new rows
```

## 6. Calibration sweep on a Bayesian forecast

```python
import numpy as np
from scipy.special import erfinv
from active_inference import (
    BayesianLinearRegression, LinearGaussianProcess, calibration_curve,
)

sigma2_y = 0.25
process = LinearGaussianProcess(beta0=3, beta1=2, sigma2_y=sigma2_y,
                                rng=np.random.default_rng(0))

# A small Monte Carlo over fresh datasets.
truths, lows_by_lvl, highs_by_lvl = [], {}, {}
levels = np.array([0.5, 0.8, 0.95])
for lvl in levels:
    lows_by_lvl[float(lvl)], highs_by_lvl[float(lvl)] = [], []
for trial in range(200):
    x_train = np.random.uniform(0, 5, size=80)
    y_train = np.array([process.sample(float(x), n=1)[0] for x in x_train])
    x_test = np.random.uniform(0, 5, size=50)
    y_test = np.array([process.sample(float(x), n=1)[0] for x in x_test])
    blr = BayesianLinearRegression(np.zeros(2), np.eye(2) * 4.0, sigma2_y)
    post = blr.fit(x_train, y_train)
    mean_pred, var_pred = post.predictive(x_test, sigma2_y=sigma2_y)
    std_pred = np.sqrt(var_pred)
    truths.extend(y_test.tolist())
    for lvl in levels:
        half = float(np.sqrt(2.0) * erfinv(float(lvl))) * std_pred
        lows_by_lvl[float(lvl)].extend((mean_pred - half).tolist())
        highs_by_lvl[float(lvl)].extend((mean_pred + half).tolist())

curve = calibration_curve(
    np.asarray(truths),
    lower_fn=lambda lvl: np.asarray(lows_by_lvl[float(lvl)]),
    upper_fn=lambda lvl: np.asarray(highs_by_lvl[float(lvl)]),
    nominal_levels=levels,
)
print(curve.calibration_error())        # ECE — should be ≤ 0.05 if calibrated
```

See [`statistics/calibration.md`](statistics/calibration.md).

## 7. Posterior predictive check in 8 lines

```python
import numpy as np
from active_inference import (
    BayesianLinearRegression, posterior_predictive_check,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(150, 2))
y = X @ np.array([0.5, -0.5]) + 1.0 + rng.normal(scale=0.4, size=150)
post = BayesianLinearRegression(np.zeros(3), np.eye(3) * 4.0, 0.16).fit(X, y)
theta_samples = post.sample(n=400, rng=rng)
replicated = np.array([
    t[0] + t[1:] @ X.T + rng.normal(scale=0.4, size=150) for t in theta_samples
])
print(posterior_predictive_check(y, replicated, statistic=np.std).summary())
```

## 8. Render a chapter figure with bigger fonts (slide-deck style)

```python
import matplotlib.pyplot as plt
from active_inference.visualizations import figure_style
from active_inference import Pipeline
from active_inference.visualizations import plot_prior_likelihood_posterior

with figure_style({"font.size": 16, "axes.titlesize": 18}):
    pipe = Pipeline.linear_gaussian(beta0=3, beta1=2, sigma2_y=0.4,
                                    m_x=4, s2_x=0.25)
    fig = plot_prior_likelihood_posterior(pipe.run(x_star=2.0, n=30),
                                          truth=2.0)
    fig.savefig("posterior_slide.png", dpi=200)
```

`figure_style(...)` is a context manager — defaults are restored on exit
so you can mix slide-style and print-style figures in one script.

## 9. Generic posterior code via the `Posterior` protocol

```python
from active_inference import (
    Pipeline, BayesianLinearRegression, LinearGaussianSystem,
    posterior_mean, posterior_std, summarize_posterior,
)

posteriors = [
    Pipeline.linear_gaussian(beta0=3, beta1=2, sigma2_y=0.4,
                             m_x=4, s2_x=1.0).run(x_star=2.0, n=20),
    LinearGaussianSystem(...).posterior_batch(Y),
    BayesianLinearRegression(...).fit(X, y),
]
for p in posteriors:
    print(summarize_posterior(p), posterior_mean(p), posterior_std(p))
```

A single loop works for grid / LGS / BLR posteriors. See
[`reference/core.md`](reference/core.md#coreposterior--cross-cutting-posterior-protocol).

## 10. Defensive validation

```python
from active_inference import (
    require_positive_scalar, require_design_matrix, require_in_unit_interval,
)

sigma2 = require_positive_scalar(0.4, name="sigma2_y")
X = require_design_matrix(my_array, n_features=3)
level = require_in_unit_interval(0.95, name="credible level")
```

Each validator returns the (coerced) value on success and raises a
``ValueError`` with a clear message on failure.

## See also

- [`reading_order.md`](reading_order.md) — what to read in what order.
- [`reference/core.md`](reference/core.md) — every public symbol catalogued.
- [`topics/`](topics/) — concept-by-concept walkthroughs.
- [`statistics/`](statistics/) — narrower pages on each statistical tool.
