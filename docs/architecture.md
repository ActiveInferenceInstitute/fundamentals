# Architecture overview

The package is organized around the *agent / environment* split that the book
emphasizes throughout Part I. Every concept gets its own small, documented
class so chapter scripts can mix and match them.

## Layered design

```
┌──────────────────────────────────────────────────────────────┐
│ chapters/<chapter>/                                          │  thin orchestrators
│  └── 30–120 LOC scripts importing from `active_inference`    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ src/active_inference/menu/      src/active_inference/web/    │  text + browser UIs
│   runner.py, tui.py               server.py, templates.py    │  stdlib-only
│   (run.sh)                        (run.sh --web)             │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ src/active_inference/visualizations/                         │  static + interactive
│   plotting.py, interactive.py, animations.py,                │  matplotlib helpers,
│   diagnostics.py, style.py                                   │  GIF / slider widgets,
│                                                              │  diagnostic figures
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ src/active_inference/estimators/                             │  point estimates
│   mle.py, map.py, gradient_descent.py,                       │  (analytic + iterative)
│   linear_regression.py, em.py                                │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ src/active_inference/core/                                   │  exact Bayesian
│   distributions.py, generative_process.py,                   │  inference + diagnostics
│   generative_model.py, inference.py, lgs.py,                 │
│   compose.py, diagnostics.py, posterior.py,                  │
│   types.py, validators.py                                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ src/active_inference/utils/                                  │  grids, logging, paths
│   grids.py, logging.py, io.py                                │
└──────────────────────────────────────────────────────────────┘
```

Higher layers depend only on lower layers; layers never import one another
sideways. This keeps the dependency graph acyclic and lets you swap out, say,
the inference engine without touching the chapter scripts.

## Key types

| Class                        | File                            | Role |
|------------------------------|---------------------------------|------|
| `GenerativeProcess`          | `core/generative_process.py`    | Samples scalar observations from a chosen generating function. |
| `LinearGaussianProcess`      | same                            | Univariate linear-Gaussian specialization (optional nonlinear `psi`). |
| `LinearGaussianMVProcess`    | same                            | Multivariate ``y = Θ x* + b + ω`` with arbitrary covariance. |
| `GenerativeModel`            | `core/generative_model.py`      | Abstract ``log_likelihood`` / ``log_prior`` interface. |
| `LinearGaussianModel`        | same                            | Univariate agent (Gaussian or uniform prior). |
| `LinearGaussianMVModel`      | same                            | Multivariate agent with full-covariance prior and likelihood. |
| `GridBayesianInference`      | `core/inference.py`             | Exact posterior via grid + trapezoid integration. |
| `InferenceResult`            | same                            | Posterior diagnostic helpers (mode, mean, variance, CI, entropy, KL). |
| `LinearGaussianSystem`       | `core/lgs.py`                   | Closed-form MVN ↔ MVN posterior; single + batched updates. |
| `LGSPosterior`               | same                            | Posterior mean + covariance with `std()`/`precision`. |
| `Pipeline`                   | `core/compose.py`               | One-shot wiring of process + model + grid inference. |
| `RunningPosteriorStats`      | `core/compose.py`               | Online posterior trace for animations / diagnostics. |
| `Posterior` (protocol)       | `core/posterior.py`             | Structural type satisfied by `InferenceResult`, `LGSPosterior`, `BLRPosterior`. |
| `CalibrationCurve`           | `core/diagnostics.py`           | Empirical vs nominal coverage at chosen credible levels. |
| `PosteriorPredictiveCheck`   | `core/diagnostics.py`           | Replicated-statistic Bayesian p-value. |
| `BayesianLinearRegression`   | `estimators/linear_regression.py`| Conjugate Gaussian update with `fit`, `fit_sequential`, predictive bands. |
| `BLRPosterior` / `GDRegressionResult` | same                   | Posterior over θ; iterate / loss history. |
| `FactorAnalysisResult`       | `estimators/em.py`              | Final loadings, posteriors, log-likelihood trace. |
| `GradientDescentResult`      | `estimators/gradient_descent.py`| Iterate / loss history of a 1-D descent run. |
| `ChapterEntry` / `ScriptEntry` | `menu/runner.py`              | Discovered chapter folder + runnable script descriptor. |

See [`reference/`](reference/) for a full per-subpackage API catalogue.

## Conventions

- All variances are *variances*, not standard deviations.
- All densities are evaluated on a 1-D NumPy grid and integrated with the
  trapezoid rule (`np.trapezoid`); this is sufficient for the scalar Part-I
  examples.
- Every chapter script accepts `--save` for headless rendering and `--seed`
  for reproducibility.
- Random number generators are passed in explicitly via `numpy.random.Generator`
  — no global state.
- The `menu/` subpackage is stdlib-only — no `numpy`/`matplotlib` imports.

## Environment / tooling

- **uv** is the recommended package manager. `uv sync` installs the
  environment from `uv.lock`; `uv run <cmd>` invokes commands inside it
  without manual activation. The top-level [`run.sh`](../run.sh) prefers
  `uv run` and falls back to `python3`/`python` if `uv` is unavailable.
- `pyproject.toml` is the single source of truth for dependencies. The
  `[dependency-groups]` table holds the `dev` group consumed by uv; the
  `[project.optional-dependencies]` table mirrors `dev` so `pip install -e
  ".[dev]"` still works.
- `requirements.txt` is kept for legacy `pip install -r` users; the
  authoritative source is `pyproject.toml`.

## Adding a new example

1. Add a method or class to the appropriate `core/` or `estimators/` module
   (with tests in `tests/<sub>/`).
2. Drop a thin orchestrator into `chapters/chapter_<N>/` that imports your
   new helper. The script should be ≤ ~120 lines.
3. Document it in `chapters/chapter_<N>/README.md`. The
   [`run.sh`](../run.sh) menu and `tests/chapters/test_smoke.py` will pick
   it up automatically as long as it follows the
   `example_*.py` / `animation_*.py` / `visualize_*.py` naming convention
   and accepts `--save`.
