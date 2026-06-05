# src/active_inference/ — Python Package

The `active_inference` package provides a clean, well-tested Python
implementation of the core algorithms described in the book
*Fundamentals of Active Inference* (Namjoshi, MIT Press 2026).

Core, estimator, utility, menu, and web entry points below are available via
`from active_inference import <name>`. Visualization helpers and the extras
registry are public subpackage APIs imported from
`active_inference.visualizations` or `active_inference.extra_topics`. The
canonical top-level list is in [`__init__.py`](__init__.py)'s `__all__`; this
README is the human-readable package catalogue.

## Import surface

### Core — distributions, generative process / model, inference

| Name | Type | Module | Role |
|---|---|---|---|
| `GenerativeProcess` | Class | `core.generative_process` | Abstract base; samples observations from `g_E(x*; theta*) + noise`. |
| `LinearGaussianProcess` | Class | same | Linear (optionally nonlinear via `psi`) Gaussian process. |
| `LinearGaussianMVProcess` | Class | same | Multivariate ``y = Θ x* + b + ω``. |
| `GenerativeModel` | Class | `core.generative_model` | Abstract `log_likelihood` / `log_prior` / `predict_mean` interface. |
| `LinearGaussianModel` | Class | same | Concrete univariate model (Gaussian or uniform prior). |
| `LinearGaussianMVModel` | Class | same | Multivariate model with full-covariance prior. |
| `GridBayesianInference` | Class | `core.inference` | Exact posterior inference on a 1-D grid. |
| `InferenceResult` | Dataclass | same | Posterior diagnostics — mode, mean, variance, CI, entropy, KL from prior. |
| `LinearGaussianSystem` | Class | `core.lgs` | Closed-form multivariate hidden-state inference. |
| `LGSPosterior` | Dataclass | same | Posterior mean / cov / precision / std. |
| `Pipeline` | Class | `core.compose` | One-line process + model + grid wiring with `run`, `sample`, `infer`. |
| `RunningPosteriorStats` | Dataclass | `core.compose` | Per-step posterior trace (means, stds, KLs, log-evidences). |
| `running_stats` | Function | `core.compose` | Compute a `RunningPosteriorStats` for an observation stream. |
| `LearningAttentionModel` / `LearningAttentionState` | Dataclass | `core.continuous_learning` | Chapter 8 continuous-state model for hidden state, parameter, and log-precision inference. |
| `learning_attention_free_energy` / `learning_attention_grad` | Function | same | VFE and analytic gradient for Chapter 8 learning/attention. |
| `HierarchicalContinuousModel` / `hierarchical_message_terms` | Dataclass/function | same | Two-layer continuous hierarchy and forward/backward message decomposition. |

### Core — distributions

| Name | Module | Role |
|---|---|---|
| `gaussian_pdf` / `gaussian_log_pdf` | `core.distributions` | Univariate Gaussian density and its log. |
| `uniform_pdf` | same | Uniform density on `[low, high]`. |
| `dirac_like_pdf` | same | Narrow Gaussian proxy for a Dirac delta. |
| `normalize_density` | same | Normalize an unnormalized density via trapezoid integration. |
| `mvn_pdf` / `mvn_log_pdf` | same | Multivariate Gaussian densities (Cholesky-stable). |
| `mvn_sample` | same | Cholesky-based MVN sampler. |
| `mahalanobis_squared` | same | Vectorized Mahalanobis distances. |
| `diagonal_cov` / `isotropic_cov` | same | Diagonal / scalar covariance constructors. |

### Core — diagnostics

| Name | Module | Role |
|---|---|---|
| `CalibrationCurve` | `core.diagnostics` | Empirical vs nominal coverage at chosen levels. |
| `PosteriorPredictiveCheck` | same | Replicated-statistic Bayesian p-value container. |
| `calibration_curve` | same | Build a `CalibrationCurve` from a forecaster and ground truth. |
| `coverage_from_intervals` | same | Single-level coverage helper. |
| `crps_gaussian` | same | Continuous ranked probability score for a Gaussian forecast. |
| `effective_sample_size` | same | Kish ESS computed in log-space. |
| `gaussian_entropy_univariate` / `gaussian_entropy_mvn` | same | Differential entropy in nats. |
| `gaussian_kl_univariate` / `gaussian_kl_mvn` | same | KL divergences between Gaussians. |
| `grid_entropy` / `grid_kl_divergence` | same | Entropy / KL for tabulated densities. |
| `log_score_gaussian` | same | Log-score for a Gaussian forecast (higher is better). |
| `logsumexp` | same | Numerically stable log-sum-exp. |
| `normal_ci` | same | Symmetric Gaussian credible / confidence interval. |
| `posterior_predictive_check` | same | Bayesian p-value with replicated samples. |
| `standardize` | same | Mean-zero / unit-variance helper. |

### Core — free-energy variants, factor graphs, ergodicity

| Name | Module | Role |
|---|---|---|
| `FreeEnergyForm` | `core.free_energy_forms` | Named scalar free-energy total plus additive term dictionary. |
| `expected_free_energy_form` / `free_energy_of_future` / `generalized_free_energy_form` | same | Pedagogical EFE, FEF, and GFE decompositions. |
| `observed_predicted_free_energy` / `bethe_free_energy_form` | same | Observed/predicted blend and Bethe-style energy/entropy form. |
| `renyi_bound` / `renyi_limit_energy` | same | Renyi-style certainty-equivalent energy and its alpha-to-one expected-energy limit. |
| `free_energy_variant_table` | same | Policy-indexed comparison arrays for EFE/FEF/GFE sweeps. |
| `normalize_message` / `categorical_factor_message` | `core.factor_graph` | Normalized categorical messages and factor-to-variable sum-product messages. |
| `sum_product_chain` / `variational_message_update` | same | Forward messages for a categorical chain and VMP-style softmax updates. |
| `EntropyBound` | `core.ergodic` | Entropy, upper bound, and non-negative residual gap. |
| `ergodic_density` / `density_entropy` | same | Histogram-based ergodic density and differential entropy on a grid. |
| `entropy_upper_bound_from_vfe` / `ergodic_ou_trajectory` | same | Entropy-bound residual helper and deterministic OU teaching trajectory. |

### Core — posterior protocol, validators, types

| Name | Module | Role |
|---|---|---|
| `Posterior` | `core.posterior` | Structural protocol satisfied by `InferenceResult`, `LGSPosterior`, `BLRPosterior`. |
| `posterior_mean` / `posterior_std` / `summarize_posterior` | same | Protocol-driven accessors. |
| `has_credible_interval` / `has_mean_cov` | same | Capability checks for generic code. |
| `assert_cov` / `assert_probabilities` | `core.types` | Shape + PSD / unit-simplex checks. |
| `require_1d`, `require_2d`, `require_design_matrix` | `core.validators` | Shape validation with coercion. |
| `require_finite_array`, `require_monotone`, `require_same_length` | same | Array-content validation. |
| `require_positive_scalar`, `require_non_negative_scalar`, `require_int_at_least`, `require_in_unit_interval` | same | Scalar validation. |

### Core — Chapters 6–7 continuous active inference

| Name | Module | Role |
|---|---|---|
| `gaussian_temporal_covariance` / `correlated_embedding_precision` | `core.generalized_filtering` | Chapter 6 §6.6 `S(γ)` and full generalized precision matrices. |
| `GeneralizedVectorModel` / `generalized_vector_free_energy_grad` | same | Vector generalized-coordinate model and FD-verified VFE gradient. |
| `MultivariateActiveInferenceAgent` / `multivariate_action_gradient` | `core.active_inference` | Chapter 7 §7.5 vector action through the generalized sensory channel. |

### Estimators

| Name | Type | Module | Role |
|---|---|---|---|
| `mle_analytic_linear` / `mle_loss` / `mle_grad_x` | Function | `estimators.mle` | Closed-form + gradient MLE for the linear-Gaussian model. |
| `map_analytic_linear` / `map_loss` / `map_grad_x` | Function | `estimators.map` | Closed-form + gradient MAP. |
| `gradient_descent` | Function | `estimators.gradient_descent` | General 1-D minimizer with optional analytic gradient. |
| `GradientDescentResult` | Dataclass | same | `x`, `history`, `losses`, `n_iterations`, `converged`. |
| `add_intercept` | Function | `estimators.linear_regression` | Prepend a constant column for the intercept. |
| `mle_linear_regression` | Function | same | Normal-equation OLS solver. |
| `squared_loss` / `squared_loss_grad` | Function | same | OLS loss and its gradient. |
| `gd_linear_regression` | Function | same | Gradient-descent OLS. |
| `GDRegressionResult` | Dataclass | same | Iterates and final coefficients. |
| `BayesianLinearRegression` | Class | same | Conjugate Gaussian update; `fit`, `fit_sequential`, predictive bands. |
| `BLRPosterior` | Dataclass | same | Posterior over θ with `mean`, `cov`, `std`, `predictive`. |
| `fit_factor_analysis` | Function | `estimators.em` | EM loop with monotonicity guarantee. |
| `factor_analysis_e_step` / `factor_analysis_m_step` | Function | same | Single-step EM helpers. |
| `incomplete_log_likelihood` | Function | same | Marginal log-likelihood of the observation data. |
| `FactorAnalysisResult` | Dataclass | same | Loadings, log-likelihood trace, convergence flag. |
| `simulate_learning_attention` | Function | `estimators.continuous_learning` | Chapter 8 perception, first-order parameter learning, and second-order log-precision attention. |
| `LearningAttentionResult` | Dataclass | same | Hidden-state, parameter, precision, prediction-error, and VFE traces. |
| `generalized_measurements_from_series` / `generalized_vector_filter` | Function | `estimators.generalized_filtering` | Chapter 6 §6.6 generalized measurement tensors and vector filter. |
| `simulate_multivariate_active_inference` | Function | `estimators.active_inference` | Chapter 7 §7.5 vector action-perception loop. |
| `MultivariateActiveInferenceResult` | Dataclass | same | Vector state, belief, action, generalized-measurement, error, and VFE traces. |

### Utilities

| Name | Module | Role |
|---|---|---|
| `make_grid` / `make_2d_grid` | `utils.grids` | Evenly-spaced 1-D and 2-D grids. |
| `get_logger` | `utils.logging` | Factory returning a configured stdlib logger. |
| `default_figure_dir` | `utils.io` | Returns `output/figures/`. |
| `default_data_dir` | `utils.io` | Returns `output/data/`. |
| `ensure_dir` | `utils.io` | Create directory if missing; return `Path`. |
| `chapter_data_dir` / `extra_data_dir` | `utils.export` | Resolve chapter and extras raw-data directories. |
| `infer_chapter_from_path` / `infer_extra_topic_from_path` | `utils.export` | Infer chapter numbers or extras topic slugs from artifact paths. |
| `save_chapter_data` / `save_extra_data` / `save_figure_data` / `save_animation_data` | `utils.export` | Write paired NPZ+JSON sidecars for saved chapter/extras arrays, figures, and animations. |
| `extract_figure_data` / `extract_animation_data` / `data_paths_for_figure` / `data_paths_for_extra_figure` | `utils.export` | Extract reconstructable Matplotlib data and map figure paths to raw-data sidecars. |

### Visualizations

| Name | Module | Role |
|---|---|---|
| `plot_prior_likelihood_posterior` | `visualizations.plotting` | Three-panel prior / likelihood / posterior figure. |
| `plot_generating_function` | same | Plot `y = g(x)` with optional sample scatter. |
| `plot_likelihood_ridge` | same | Stacked ridge plot of per-sample likelihoods. |
| `plot_joint_heatmap` | same | Heatmap of joint density `p(x, y)`. |
| `plot_gradient_descent` | same | Side-by-side loss curve + iterate trajectory. |
| `plot_precision_comparison` | same | Overlay multiple posteriors for a precision sweep. |
| `plot_2d_gaussian` | same | 2-D MVN contour figure. |
| `confidence_ellipse` | same | Draw a Gaussian confidence ellipse. |
| `save_or_show` | same | Save figure or call `plt.show`. |
| `interactive_inference` | `visualizations.interactive` | 4-slider widget for real-time Bayesian update. |
| `interactive_precision` | same | 1-slider widget for precision-ratio sweep. |
| `interactive_topic_demo` | same | Registry-driven extras topic slider backed by `extra_topics.build_topic_demo`. |
| `animate_sequential_posterior` | `visualizations.animations` | Posterior tightening as N grows. |
| `animate_gradient_descent` | same | Iterate rolling down NLL. |
| `animate_2d_posterior` | same | 2-D MVN posterior collapse. |
| `animate_em_convergence` / `animate_em_steps` | same | EM log-likelihood + per-step trajectory. |
| `animate_sufficient_statistics` | same | Running sufficient statistics over a Gaussian stream. |
| `animate_calibration_growth` | same | Calibration curve filling in as data arrives. |
| `animate_precision_sweep` / `animate_bimodal_emergence` / `animate_lgs_online` / `animate_blr_predictive_band` | same | Chapter-3 animation drivers. |
| `animate_learning_attention` | same | Chapter 8 learning/attention convergence GIF. |
| `animate_multivariate_active_inference` | same | Chapter 7 §7.5 vector action-perception GIF. |
| `save_animation` | same | Persist a `FuncAnimation` to a GIF / MP4. |
| `plot_correlated_embedding_precision` / `plot_generalized_vector_filter` / `plot_multivariate_active_inference` | `visualizations.unified` | Chapter 6 §6.6 and Chapter 7 §7.5 educational figures. |
| `plot_learning_attention` / `plot_hierarchical_message_passing` | `visualizations.unified` | Chapter 8 learning/attention traces and hierarchy message-passing diagram. |
| `plot_calibration` / `plot_cdf_comparison` / `plot_coverage_curve` | `visualizations.diagnostics` | Calibration & coverage diagnostics. |
| `plot_kl_trace` / `plot_score_trace` / `plot_running_statistics` | same | Running posterior diagnostic plots. |
| `plot_qq` / `plot_posterior_predictive_check` | same | Distributional diagnostics. |
| `COLORS` / `DEFAULT_RC` / `figure_style` / `set_default_style` / `annotate_stat_box` / `stat_box_bbox` | `visualizations.style` | Repo-wide matplotlib style + helpers. |

### Menu

| Name | Module | Role |
|---|---|---|
| `run_menu` | `active_inference` (top-level) | Equivalent to `python -m active_inference.menu`; used by the PEP 621 console script `active-inference-menu`. |
| `main` | `menu.tui` | Entry point with argparse-driven non-interactive flags. |
| `discover_chapters` / `discover_extras` / `discover_scripts` / `discover_extra_scripts` | `menu.runner` | Folder-driven chapter and extras discovery. |
| `run_chapter` / `run_extra_topic` / `run_all_chapters` / `run_all_extras` / `run_script` | same | Headless execution helpers (set `MPLBACKEND=Agg`). |
| `ChapterEntry` / `ExtraTopicEntry` / `ScriptEntry` | same | Frozen dataclasses describing discovered chapters, extras topics, and scripts. |

### Extras Registry

The top-level module `active_inference.extra_topics` powers the repo-root
`extras/` wrappers and is imported by the menu and web discovery layers. It is
not re-exported through the top-level package `__all__`; wrapper scripts import
its runner helpers directly through `active_inference.extra_topics`.

| Name | Module | Role |
|---|---|---|
| `ExtraTopicSpec` / `TopicDemo` | `extra_topics` | Frozen metadata and numerical-demo containers for one extras topic. |
| `EXTRA_TOPICS` / `extra_topic_slugs` / `extra_topic_spec` / `extra_topics_by_family` | same | Registry and lookup helpers shared by docs, tests, menu, and web UI. |
| `build_topic_demo` / `render_topic_figure` / `build_topic_animation` | same | Deterministic data and artifact builders for extras wrappers. |
| `main_visualize` / `main_simulate` / `main_animation` / `main_interactive` / `topic_artifact_path` | same | Thin CLI entry points used by `extras/<topic>/` scripts. |

### Web

| Name | Module | Role |
|---|---|---|
| `run_web` | `active_inference` (top-level) | Equivalent to `python -m active_inference.web`; used by the PEP 621 console script `active-inference-web`. |
| `run_server` / `launch` | `web.server` | Start the local HTTP server. `block=False` returns the `ThreadingHTTPServer` for in-process tests. |
| `DEFAULT_HOST` / `DEFAULT_PORT` | `web` | Default bind address (`127.0.0.1:8765`). |
| `main` | `web.server` | CLI entry point (`--host`, `--port`, `--no-browser`, `--verbose`). |

### Version

```python
import active_inference
active_inference.__version__  # "0.1.0"
```

## Subpackage Index

| Subpackage | Path | README |
|---|---|---|
| `core` | [`src/active_inference/core/`](core/) | [README](core/README.md) |
| `estimators` | [`src/active_inference/estimators/`](estimators/) | [README](estimators/README.md) |
| `utils` | [`src/active_inference/utils/`](utils/) | [README](utils/README.md) |
| `visualizations` | [`src/active_inference/visualizations/`](visualizations/) | [README](visualizations/README.md) |
| `menu` | [`src/active_inference/menu/`](menu/) | [README](menu/README.md) |
| `web` | [`src/active_inference/web/`](web/) | [README](web/README.md) |

## Quick Start

```python
from active_inference import (
    LinearGaussianProcess,
    LinearGaussianModel,
    GridBayesianInference,
    Pipeline,
    make_grid,
    map_analytic_linear,
)

# Idiomatic — Pipeline does the wiring in one line:
pipe = Pipeline.linear_gaussian(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
result = pipe.run(x_star=2.0, n=30)
print(result.posterior_mode, result.credible_interval(0.95))

# Or assemble by hand:
process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.25)
model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                             m_x=4.0, s2_x=0.25)
y = process.sample(x_star=2.0, n=30).flatten()
result = GridBayesianInference(model, make_grid(0, 5, 500)).infer(y)

# Closed-form point estimate:
print(map_analytic_linear(y, 3.0, 2.0, 0.25, 4.0, 0.25))
```

## Dependencies

Required: `numpy >= 1.23`, `scipy >= 1.10`, `matplotlib >= 3.6`,
`pillow >= 10.0` (for GIF rendering).

Optional extras (see [`../../pyproject.toml`](../../pyproject.toml)):

- `interactive`: `ipywidgets >= 8.0`, `jupyter >= 1.0`
- `dev`: `pytest >= 7.0`, `pytest-cov >= 4.0`, `ruff >= 0.1.0`

Install with `uv sync` (recommended) or `pip install -e ".[dev]"`.
