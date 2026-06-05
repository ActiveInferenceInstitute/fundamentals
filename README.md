# Fundamentals of Active Inference — Python Companion

Open-source Python code that follows along with
[Sanjeev V. Namjoshi's *Fundamentals of Active Inference*](https://mitpress.mit.edu/9780262050951/fundamentals-of-active-inference/)
(MIT Press, 2026). The book itself is **not** open source; this repository
provides a clean, well-tested Python implementation of the algorithms it
describes and a set of thin orchestrator scripts that reproduce its figures
and numerical examples.

> Maintained by the [Active Inference Institute](https://activeinference.institute/).
> Live cohort-based reading groups for the book run continuously — register at
> [textbook-group.activeinference.institute](https://textbook-group.activeinference.institute/).

```
fundamentals/
├── run.sh                     ← top-level chapter runner (text menu)
├── pyproject.toml             ← PEP 621 metadata, uv & pip both honored
├── uv.lock                    ← pinned environment for reproducibility
├── src/active_inference/      ← reusable, documented, tested library
│   ├── core/                  ← distributions, generative process/model, inference, diagnostics, composition, validators
│   ├── estimators/            ← MLE, MAP, gradient descent, linear regression, EM / factor analysis
│   ├── visualizations/        ← static plots, slider explorers, animations, diagnostic figures, style
│   ├── utils/                 ← grids, logging, paths, NPZ+JSON data exports
│   ├── menu/                  ← stdlib text menu used by run.sh
│   └── web/                   ← stdlib local web UI launched by run.sh --web
├── chapters/
│   ├── chapter_01/            ← 4 concept orchestrators
│   ├── chapter_02/            ← examples 2.1–2.10 + auxiliary + 2 animations
│   ├── chapter_03/            ← examples 3.1–3.7 + 8 animations + 3 diagnostic visualizations
│   ├── chapter_04/            ← variational inference: 5 examples + 1 animation + 3 visualizations + interactive
│   ├── chapter_05/            ← predictive coding: 4 examples + 2 animations (univariate / multivariate / hierarchical)
│   ├── chapter_06/            ← generalized filtering for perception (Part II): 4 examples + 1 visualization (§6.1–§6.6)
│   ├── chapter_07/            ← active generalized filtering (Part II): 2 examples + 1 animation (§7.1–§7.5)
│   ├── chapter_08/            ← learning, attention, and hierarchy (Part II): 2 examples + 1 visualization + 1 animation (§8.1–8.6)
│   ├── chapter_09/            ← active inference in POMDPs (Part II): 5 examples + 2 animations (§9.1–9.6, discrete)
│   └── chapter_10/            ← learning & extensions in POMDPs (Part II): 8 examples + 1 visualization + 3 animations (§10.1 Dirichlet A/B/D + novelty, §10.2 habit + precision, §10.3 factorial / two-armed bandit, §10.4 hierarchical)
├── extras/                    ← cross-cutting topic orchestrators beyond the chapter spine
├── docs/                      ← architecture, notation, chapter prose, topic walkthroughs, reference, statistics
├── scripts/                   ← batch runners, figure pipeline
├── tests/                     ← pytest suite (unit + chapter smoke tests)
├── output/figures/            ← regenerated PNGs / GIFs per chapter and extras topic
└── output/data/               ← regenerated NPZ arrays + JSON metadata per chapter and extras topic
```

## Install

This project is uv-first; plain `pip` still works as a fallback.

### Option A — uv (recommended)

```bash
# Install uv once: https://docs.astral.sh/uv/getting-started/installation/
git clone https://github.com/ActiveInferenceInstitute/fundamentals
cd fundamentals
uv sync                       # creates .venv, installs runtime + dev deps from uv.lock
uv sync --extra interactive   # optional: ipywidgets / jupyter for notebooks
```

Then either activate the venv (`source .venv/bin/activate`) or prefix every
command with `uv run` (e.g. `uv run pytest`). The `run.sh` script in the
repo root detects `uv` automatically.

### Option B — pip

```bash
git clone https://github.com/ActiveInferenceInstitute/fundamentals
cd fundamentals
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # or `pip install -e .` for runtime only
```

The package needs only `numpy`, `scipy`, `matplotlib`, and `pillow`
(for GIF rendering). The `interactive` extra adds `ipywidgets` + `jupyter`;
`dev` brings `pytest` + `pytest-cov` + `ruff`.

## Running everything

A text menu at the repo root drives every chapter script, and a local
web UI sits behind `--web`:

```bash
./run.sh                        # interactive text menu
./run.sh --all                  # render every chapter to output/figures/
./run.sh --chapter 2            # one chapter
./run.sh --script example_2_2   # one orchestrator by filename fragment
./run.sh --list                 # print the discovered menu and exit
./run.sh --no-animations        # skip slow GIF renderers
./run.sh --keep-going           # continue past failing scripts
uv run python -m active_inference.menu --extras          # render all extras
uv run python -m active_inference.menu --extra entropy   # one extras topic

# Local browser interface — one tab per chapter, gallery + render buttons:
./run.sh --web                  # default http://127.0.0.1:8765/
./run.sh --web --no-browser     # without auto-opening the browser
./run.sh --web --port 8080      # custom port (ephemeral fallback if taken)
```

Both surfaces are also reachable as Python modules / console scripts:

```bash
uv run python -m active_inference.menu
uv run python -m active_inference.web
uv run active-inference-menu        # PEP 621 console script (after install)
uv run active-inference-web         # ditto, for the browser UI
```

See [`docs/web.md`](docs/web.md) for the full HTTP surface and design
notes.

The older batch pipeline still works and is exercised by CI:

```bash
# render every figure from Chapters 1–10
uv run python scripts/run_all_figures.py
uv run python scripts/run_all_figures.py --chapters 1
uv run python scripts/run_all_figures.py --chapters 2 --no-animations
uv run python scripts/run_all_figures.py --chapters 4 --clean
uv run python scripts/run_all_figures.py --no-chapters --extras entropy expected_free_energy
uv run python scripts/run_all_figures.py --no-chapters --extras
uv run python scripts/validate_rendered_figures.py --root output/figures
uv run python scripts/validate_raw_data_exports.py --root output/data --chapters 1 2 3 4 5 6 7 8 9 10
uv run python scripts/validate_raw_data_exports.py --root output/data
uv run python scripts/validate_book_topic_coverage.py
uv run python scripts/validate_book_topic_coverage.py --require-rendered
uv run python scripts/validate_orchestrator_provenance.py

# unit tests
uv run pytest

# chapter smoke tests only
uv run pytest tests/chapters -v

# extras smoke tests only
uv run pytest tests/extras -v
```

Each chapter script runs standalone too:

```bash
uv run python chapters/chapter_01/01_box_scenario.py --save
uv run python chapters/chapter_02/example_2_2_linear_probabilistic.py --save
uv run python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save
uv run python chapters/chapter_02/interactive_explorer.py            # GUI window
uv run python extras/temperature/visualize_temperature.py --save
uv run python extras/temperature/interactive_temperature.py          # GUI window
```

When a non-interactive script is run with `--save`, the saved visual is paired
with raw data in `output/data/chapter_NN/` or `output/data/extras/<topic>/`: a
compressed `NPZ` file for numeric arrays and a `JSON` manifest for script
provenance, CLI args, seed when present, figure paths, array shapes/dtypes, and
summary statistics. The shared
`save_or_show` and `save_animation` helpers export figure/animation data
automatically; scripts with additional numerical traces can call
`save_chapter_data` or `save_extra_data` directly. Validate a render with
`scripts/validate_raw_data_exports.py`. Validate the thin-wrapper source-method
boundary with `scripts/validate_orchestrator_provenance.py`.

## What's inside

### `src/active_inference/` — the library

| Subpackage | What it provides |
|---|---|
| `core.distributions` | `gaussian_pdf`, `gaussian_log_pdf`, `uniform_pdf`, `dirac_like_pdf`, `normalize_density`, plus the multivariate set `mvn_pdf`, `mvn_log_pdf`, `mvn_sample`, `mahalanobis_squared`, `diagonal_cov`, `isotropic_cov` — numerically stable, fully vectorized |
| `core.generative_process` | `GenerativeProcess` (abstract), `LinearGaussianProcess`, `LinearGaussianMVProcess` — samples observations from a chosen generating function |
| `core.generative_model` | `GenerativeModel` (abstract), `LinearGaussianModel`, `LinearGaussianMVModel` — agent-side model with Gaussian or uniform prior, optional nonlinear `psi(x)` |
| `core.inference` | `GridBayesianInference` — exact posterior via grid + trapezoid integration; `InferenceResult` with mode, mean, variance, credible interval, entropy, KL from prior |
| `core.lgs` | `LinearGaussianSystem` — closed-form multivariate hidden-state inference; `LGSPosterior` with mean, cov, precision, std |
| `core.compose` | `Pipeline` (one-line process + model wiring), `running_stats` / `RunningPosteriorStats` |
| `core.diagnostics` | `calibration_curve`, `coverage_from_intervals`, `crps_gaussian`, `effective_sample_size`, `gaussian_entropy_*`, `gaussian_kl_*`, `grid_entropy`, `grid_kl_divergence`, `log_score_gaussian`, `logsumexp`, `normal_ci`, `posterior_predictive_check`, `standardize` |
| `core.posterior` | `Posterior` protocol + `summarize_posterior`, `posterior_mean`, `posterior_std`, `has_*` helpers — works across grid / LGS / BLR posteriors |
| `core.free_energy_forms` | `FreeEnergyForm`, EFE/FEF/GFE/Bethe/Renyi teaching decompositions, and `free_energy_variant_table` for policy-indexed comparisons |
| `core.factor_graph` | `normalize_message`, `categorical_factor_message`, `sum_product_chain`, and `variational_message_update` for categorical factor-graph demos |
| `core.ergodic` | `EntropyBound`, `ergodic_density`, `density_entropy`, `entropy_upper_bound_from_vfe`, and `ergodic_ou_trajectory` for FEP/ergodicity extras |
| `core.thermodynamics` | `ThermodynamicState`, `canonical_probabilities`, `expected_energy`, `boltzmann_entropy`, `helmholtz_free_energy`, `enthalpy`, `gibbs_free_energy`, `vfe_thermodynamic_state` — explicit FEP thermodynamic bridge |
| `core.types` | `assert_cov`, `assert_probabilities` — shape / PSD checks |
| `core.validators` | `require_1d`, `require_2d`, `require_design_matrix`, `require_finite_array`, `require_in_unit_interval`, `require_int_at_least`, `require_monotone`, `require_non_negative_scalar`, `require_positive_scalar`, `require_same_length` |
| `core.variational` | `GaussianBelief`, `variational_free_energy` → `VFEComponents` (the five VFE forms), `vfe_g_form`/`vfe_d_form`/`vfe_c_form`/`vfe_e_form`/`vfe_map_form`/`vfe_mle_form`, `log_model_evidence`, `surprisal` — Chapter 4 |
| `core.predictive_coding` | `GenerativeFunction`/`LinearFunction`/`QuadraticFunction`/`GenericFunction`, `PredictiveCodingModel`, `predictive_coding_free_energy` → `PCFreeEnergy`, `pc_free_energy_grad`(`_fd`), `pc_linear_fixed_point`, `pc_curvature_linear`, `sensory_prediction_error`, `state_prediction_error` — Chapter 5 |
| `core.generalized_filtering` | §6.1: `DynamicStateSpaceModel`, `gf_free_energy`, `gf_free_energy_grad`(`_fd`), `gf_fixed_point_linear`; §6.2: `MultivariateDynamicModel`, `LinearVectorFunction`/`GenericVectorFunction`, `mv_gf_*`; §6.3–§6.5: `shift_operator`, `embed_flow`, `GeneralizedModel`, `generalized_state_error`/`generalized_sensory_error`, `generalized_free_energy`(`_grad`/`_grad_fd`); §6.6: `gaussian_temporal_covariance`, `correlated_embedding_precision`, `GeneralizedVectorModel`, `generalized_vector_free_energy`(`_grad`/`_grad_fd`) — Chapter 6 |
| `core.continuous_learning` | `LearningAttentionModel`, `LearningAttentionState`, `LearningAttentionComponents`, `learning_attention_free_energy`, `learning_attention_grad`(`_fd`), `log_precision_to_precision`, `log_precision_to_variance`; `ContinuousHierarchyLayer`, `HierarchicalContinuousModel`, `hierarchical_continuous_free_energy`(`_grad`/`_grad_fd`), `hierarchical_message_terms` — Chapter 8 |
| `core.diagnostics` (Ch.5 additions) | `gradient_check`, `convergence_report` → `ConvergenceReport`, `oracle_agreement` → `OracleAgreement` |
| `estimators.mle` | `mle_analytic_linear`, `mle_loss`, `mle_grad_x` |
| `estimators.map` | `map_analytic_linear`, `map_loss`, `map_grad_x` |
| `estimators.gradient_descent` | `gradient_descent`, `GradientDescentResult` |
| `estimators.linear_regression` | `mle_linear_regression`, `gd_linear_regression`, `BayesianLinearRegression`, `BLRPosterior`, `GDRegressionResult`, `add_intercept`, `squared_loss`, `squared_loss_grad` |
| `estimators.em` | `fit_factor_analysis`, `factor_analysis_e_step`, `factor_analysis_m_step`, `incomplete_log_likelihood`, `FactorAnalysisResult` |
| `estimators.variational` | `coordinate_search_vfe`, `fixed_form_vi`, `free_form_cavi`, `MeanFieldConfig`, and their result traces (`CoordinateSearchResult`, `FixedFormResult`, `CAVIResult`) — Chapter 4 VBI |
| `estimators.predictive_coding` | `predictive_coding_inference`, `multivariate_predictive_coding`, `hierarchical_predictive_coding`, `HierarchicalPCModel`, `PredictiveCodingResult`, `MultivariatePCResult`, `HierarchicalPCResult` — Chapter 5 |
| `estimators.generalized_filtering` | `generalized_filter` (§6.1), `multivariate_generalized_filter` (§6.2), `generalized_filter_gc` (§6.3–§6.5), `generalized_measurements_from_series`, `generalized_vector_filter`, and their process simulators / result types — Chapter 6 |
| `core.active_inference` / `estimators.active_inference` | `ActiveInferenceAgent`, `action_gradient`, `perception_gradient`; `ActiveEnvironment`, `simulate_active_inference`, `ActiveInferenceResult`; §7.5 `MultivariateActiveInferenceAgent`, `multivariate_action_gradient`, `MultivariateActiveEnvironment`, `simulate_multivariate_active_inference`, `MultivariateActiveInferenceResult` — Chapter 7 (continuous-state AIF) |
| `estimators.continuous_learning` | `simulate_learning_attention`, `LearningAttentionResult` — Chapter 8 perception, learning, and attention |
| `core.pomdp` | `POMDPModel`, `infer_states`, `predict_state`, `efe_components`, `EFEComponents`, `evaluate_policy_components`, `PolicyEFETrace`, `efe_risk`, `efe_ambiguity`, `expected_free_energy`, `evaluate_policy`, `policy_posterior`, `action_posterior`, `forward_filter`, `predict_beliefs`, `discrete_vfe`, `softmax`, `one_hot`, `is_stochastic_matrix` — Chapter 9; `dirichlet_expected_value`, `expected_A/B/D`, `accumulate_a/b/d_counts`, `novelty_matrix`, `parameter_novelty`, `efe_with_novelty` — Ch.10 §10.1; `policy_posterior_full`, `precision_gradient`, `learn_precision` — Ch.10 §10.2; `FactorialPOMDP`, `factorial_infer_states`, `factorial_efe`, `factorial_expected_observation`, `factorial_likelihood_message`, `factorial_modality_ambiguity`, `factorial_predict_states` — Ch.10 §10.3 (factorial); `HierarchicalPOMDP`, `hierarchical_layer_vfe`, `hierarchical_layer_efe`, `hierarchical_policy_posterior` — Ch.10 §10.4 (hierarchical) |
| `estimators.pomdp` | `make_gridworld`, `enumerate_policies`, `simulate_pomdp_agent`, `GridWorldResult` — Chapter 9 §9.5 (Grid World planning); `DirichletParameters`, `LearningResult`, `learn_D_vector`, `simulate_array_learning`, `simulate_learning_agent` — Ch.10 §10.1; `precision_policy_sweep` — Ch.10 §10.2; `make_two_armed_bandit`, `simulate_two_armed_bandit`, `TwoArmedBanditResult` — Ch.10 §10.3 (two-armed bandit); `simulate_hierarchical_agent`, `HierarchicalResult` — Ch.10 §10.4 |
| `utils.grids` | `make_grid`, `make_2d_grid` |
| `utils.logging` | `get_logger` — lightweight, consistent logger factory |
| `utils.io` | `default_figure_dir`, `default_data_dir`, `ensure_dir` |
| `utils.export` | `save_chapter_data`, `save_extra_data`, `extract_figure_data`, `extract_animation_data`, `data_paths_for_figure`, `data_paths_for_extra_figure`; the NPZ+JSON raw-data contract used by `--save` |
| `extra_topics` | `EXTRA_TOPICS`, `extra_topic_spec`, `extra_topics_by_family`, `build_topic_demo`, `main_visualize`, `main_simulate`, `main_animation`, and `main_interactive` — registry-driven extras curriculum runners |
| `visualizations.plotting` | `plot_prior_likelihood_posterior`, `plot_generating_function`, `plot_likelihood_ridge`, `plot_joint_heatmap`, `plot_gradient_descent`, `plot_precision_comparison`, `plot_2d_gaussian`, `confidence_ellipse`, `save_or_show` |
| `visualizations.interactive` | `interactive_inference`, `interactive_precision`, `interactive_topic_demo` — matplotlib slider widgets, no `ipywidgets` dependency |
| `visualizations.variational` | `vfe_surface`, `plot_vfe_contour`, `plot_density_evolution`, `plot_vfe_decomposition`, `plot_surprisal_relationship` — Chapter 4 figures |
| `visualizations.unified` | `plot_recognition_dynamics` (Ch.4 `FixedFormResult` or Ch.5 `PredictiveCodingResult`), `plot_prediction_errors`, `plot_hierarchical_pc`, `plot_correlated_embedding_precision`, `plot_generalized_vector_filter`, `plot_multivariate_active_inference`, `plot_learning_attention`, `plot_hierarchical_message_passing`, `panel_grid`, `finalize`, `layer_colors` — composable Ch.4–10 layer |
| `visualizations.animations` | `animate_sequential_posterior`, `animate_gradient_descent`, `animate_2d_posterior`, `animate_em_convergence`, `animate_em_steps`, `animate_sufficient_statistics`, `animate_calibration_growth`, `animate_precision_sweep`, `animate_bimodal_emergence`, `animate_lgs_online`, `animate_blr_predictive_band`, `animate_vfe_descent`, `animate_recognition_dynamics`, `animate_hierarchical_pc`, `animate_multivariate_active_inference` (Ch.7 §7.5), `animate_learning_attention` (Ch.8), `animate_parameter_learning` (Ch.10 §10.1), `animate_policy_precision` (Ch.10 §10.2), `animate_two_armed_bandit` (Ch.10 §10.3), `save_animation` |
| `visualizations.diagnostics` | `plot_calibration`, `plot_cdf_comparison`, `plot_coverage_curve`, `plot_kl_trace`, `plot_posterior_predictive_check`, `plot_qq`, `plot_running_statistics`, `plot_score_trace` |
| `visualizations.style` | `COLORS` (Okabe-Ito colourblind-safe), `DEFAULT_RC`, `figure_style`, `set_default_style`, `annotate_stat_box`, `annotate_point`, `stat_box_bbox` |
| `menu` | `discover_chapters`, `discover_extras`, `discover_scripts`, `discover_extra_scripts`, `run_chapter`, `run_extra_topic`, `run_all_chapters`, `run_all_extras`, `run_script`, `main` — stdlib text menu (used by `run.sh`) |
| `web` | `run_server`, `launch`, `main` — stdlib HTTP server (used by `run.sh --web`); tab-per-chapter and tab-per-extra UI with figure galleries, render buttons, and inline docs |

Every public function/class is imported at `src/active_inference/__init__.py`
and listed in `__all__`. See [`docs/reference/`](docs/reference/) for a
subpackage-by-subpackage API catalogue.

### `chapters/` — thin orchestrators

Each script is ≤ ~120 lines, imports only from `active_inference` and
standard library, and follows the same pattern: parse args, build process
+ model, infer, plot, optionally save.

### `extras/` — cross-cutting topic orchestrators

Extras are deterministic topic demos beyond the Chapter 1-10 spine. The
registry in `active_inference.extra_topics` covers 58 concept-level topics
across Chapters 1-14 plus the math and free-energy appendices: foundations,
estimation, information theory, variational inference, predictive coding,
continuous active inference, discrete POMDPs, Part III extensions, factor
graphs, applications, ergodic density, and the thermodynamic/FEP bridge.

Each folder has a `README.md` and a `visualize_<topic>.py` static orchestrator;
57 topics also have `simulate_<topic>.py` and `interactive_<topic>.py`, and 23
topics have `animation_<topic>.py` for iterative trajectories. Saved extras
JSON manifests include the registry's `source_apis` references so artifacts can
be traced back to tested `active_inference` methods. The source-spine audit
lives in
[`docs/reference/book_topic_matrix.md`](docs/reference/book_topic_matrix.md)
and is checked by `scripts/validate_book_topic_coverage.py`. After rendering,
`scripts/validate_book_topic_coverage.py --require-rendered` additionally
requires every declared extras artifact to have matching PNG/GIF output and
NPZ+JSON raw-data sidecars.

**Chapter 1 — The Hypothesis-Testing Brain** (4 scripts)

| Script | What it shows |
|---|---|
| `01_box_scenario.py` | The "agent in a box" thought experiment as a stream of noisy sensor readings |
| `02_three_perspectives.py` | Side-by-side simulation of the *scientific*, *hypothesis-testing*, and *statistical* views |
| `03_bayes_intuition.py` | Bayes' theorem step-by-step on a single-state, single-observation toy |
| `04_inverse_problem.py` | Non-injective generator → bi-modal posterior |

**Chapter 2 — Hidden State Estimation** (10 examples + 2 auxiliary + 2 animations)

| Script | Mirrors | What it adds |
|---|---|---|
| `example_2_1_linear_deterministic.py` | Example 2.1 | Bayesian inversion of a noiseless linear sensor |
| `example_2_2_linear_probabilistic.py` | Example 2.2 | Standard Gaussian likelihood × Gaussian prior |
| `example_2_3_precision.py` | Example 2.3 | Sweep prior vs likelihood precision; plot the trade-off |
| `example_2_4_nonlinear_deterministic.py` | Example 2.4 | Quadratic generator → bi-modal posterior |
| `example_2_5_nonlinear_probabilistic.py` | Example 2.5 | Nonlinear generator with Gaussian noise |
| `example_2_6_imperfect_model.py` | Example 2.6 | Mismatch between generative process and model |
| `example_2_7_multiple_samples.py` | Example 2.7 | Sequential vs batch inference over N i.i.d. samples |
| `example_2_8_mle_analytic.py` | Example 2.8 | Closed-form MLE compared to grid-Bayesian mode |
| `example_2_9_map_analytic.py` | Example 2.9 | Closed-form MAP compared to grid-Bayesian mode |
| `example_2_10_gradient_descent.py` | §2.5.2 | Iterative MLE / MAP via gradient descent |
| `visualize_generative_model.py` | §2.4 | Heatmap and 3-D surface of `p(x, y)` |
| `interactive_explorer.py` | bonus | Slider-driven exploration of the canonical model |
| `animation_sequential.py` | bonus | Animated posterior tightening as N grows (GIF) |
| `animation_gradient_descent.py` | bonus | Animated iterate rolling down the NLL (GIF) |

**Chapter 3 — Combining Learning and Inference** (7 examples + 8 animations + 3 diagnostic visualizations)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_3_1_linear_regression_mle.py` | Example 3.1 | Closed-form linear regression at low N: many plausible θ hypotheses |
| `example_3_2_linear_regression_gd.py` | Example 3.2 | Gradient descent over the (β₀, β₁) loss surface |
| `example_3_3_multiple_regression.py` | Example 3.3 | Vectorized multiple regression via the normal equation |
| `example_3_4_multivariate_gaussian.py` | Example 3.4 | Anatomy of the MVN: covariance shapes, sampling, contours |
| `example_3_5_bayesian_linear_regression.py` | Example 3.5 | Posterior over θ tightens with N; predictive bands shown |
| `example_3_6_lgs_food_localization.py` | Example 3.6 | Multivariate hidden-state inference for a 2-D food source |
| `example_3_7_factor_analysis_em.py` | §3.5 | EM loop on synthetic factor-analysis data with reconstruction |
| `animation_blr_tightening.py` | bonus | 2-D posterior over (β₀, β₁) tightening as data arrives |
| `animation_blr_predictive_band.py` | bonus | Predictive band shrinking as new observations arrive |
| `animation_em_convergence.py` | bonus | EM log-likelihood and loadings matrix evolving per iteration |
| `animation_em_steps.py` | bonus | Detailed E / M step alternation of factor-analysis EM |
| `animation_lgs_online.py` | bonus | Online 2-D LGS posterior collapsing with each sample |
| `animation_precision_sweep.py` | bonus | Posterior shape as prior/likelihood precision varies |
| `animation_bimodal_emergence.py` | bonus | Bi-modal posterior emerging from a non-injective generator |
| `animation_sufficient_statistics.py` | bonus | Running sufficient statistics over a Gaussian stream |
| `visualize_calibration.py` | diagnostic | Empirical-vs-nominal coverage curve for a BLR forecast |
| `visualize_coverage.py` | diagnostic | Coverage sweep across credible levels |
| `visualize_posterior_predictive.py` | diagnostic | Posterior predictive check on regression residuals |

**Chapter 4 — Variational Bayesian Inference** (5 examples + 1 animation + 3 visualizations + 1 interactive)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_4_1_coordinate_search.py` | Example 4.1 / Alg. 4.2.1 | Zero-order coordinate search descends the VFE surface (Fig. 4.2.2) |
| `example_4_2_surprisal.py` | §4.3 | Surprisal `−log p(y)` vs `y` and vs `p(y)` (Fig. 4.3.1) |
| `example_4_3_vfe_forms.py` | §4.4 | All five VFE forms agree; G/C/E decompositions over a descent (Fig. 4.4.1) |
| `example_4_6_free_form_cavi.py` | Example 4.6 / Alg. 4.5.1 | Free-form mean-field CAVI on `(x, β₀, β₁)` converges, VFE monotone |
| `example_4_7_fixed_form.py` | Example 4.7 / Alg. 4.6.1 | Fixed-form gradient VI converges to the exact posterior `N(2.4, 0.05)` (Fig. 4.6.1/4.6.2) |
| `animation_vfe_descent.py` | bonus | `q(x)` tightening onto the posterior as VFE falls to the surprisal bound (GIF) |
| `visualize_kl_loss.py` | §4.1 | The KL loss surface vs the VFE surface (same minimum, no posterior needed) |
| `visualize_vfe_intuition.py` | §4.2 | G-form intuition: `q(x)`, `p(x, y)`, and the posterior (Fig. 4.2.3) |
| `visualize_model_comparison.py` | §4.3 | Model evidence of a good vs bad model against the true input (Fig. 4.3.2/4.3.3) |
| `interactive_vfe_explorer.py` | bonus | Slider-driven `(μ, σ²)` exploration of the live VFE decomposition |

**Chapter 5 — Predictive Coding** (4 examples + 2 animations; univariate / multivariate / hierarchical)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_5_1_prediction_errors.py` | §5.1 / Fig. 5.1.2 | Flat-prior MLE and the MAP free energy as two precision-weighted prediction errors |
| `example_5_3_multivariate.py` | §5.3 | Multivariate predictive coding (vector state, Jacobian `g`) converges |
| `example_5_4_recognition_dynamics.py` | Alg. 5.2.1 | Recognition dynamics (Eq. 16); `--linear` lands on the Ch.4 grid posterior mean `2.4` |
| `example_5_7_hierarchical.py` | Example 5.7 / §5.4 | Hierarchical PC converges to `[2,1,0]`, all layer errors → 0, `Σ F = 0` (Fig. 5.4.4) |
| `animation_recognition_descent.py` | Alg. 5.2.1 | GIF: `μ_x` descending onto the oracle, errors decaying, `𝓕` falling (`--nonlinear` for Fig. 5.2.3) |
| `animation_hierarchical.py` | Example 5.7 / Fig. 5.4.4 | GIF: layer beliefs settling to `[2,1,0]`, errors → 0, `Σ F → 0` |

**Chapter 6 — Generalized Filtering for Perception** (Part II; §6.1–§6.6)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_6_1_generalized_filter.py` | Example 6.1 / Alg. 6.1.1 | An agent filters a noisy observation stream online; belief `μ_x` tracks a moving hidden state `x*` (Fig. 6.1.3) |
| `example_6_2_multivariate_filter.py` | Example 6.2 / §6.2 | Multivariate filter on a Hooke's-law oscillator; vector belief tracks the orbit with perception lag (Fig. 6.2.3) |
| `example_6_6_generalized_coordinates.py` | Example 6.6 / §6.5 | Generalized filtering in generalized coordinates; belief recovers position *and* velocity (Fig. 6.5) |
| `visualize_6_6_correlated_embedding_orders.py` | §6.6 / Fig. 6.6.2 | Heatmaps of the generalized precision `Π̃(γ)` show how smoothness couples embedding orders |
| `example_6_7_multivariate_generalized_coordinates.py` | Example 6.7 / §6.6 | Vector generalized-coordinate filtering with correlated precision improves oscillator tracking over the ordinary filter |

**Chapter 7 — Active Generalized Filtering** (Part II; §7.1–§7.5)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_7_2_active_inference.py` | Example 7.2 / Alg. 7.2.1 | Active inference: the agent's action drives the true state to its preferred set-point against an exogenous force (`a → −v*`); the action-perception cycle (Fig. 7.4.5) |
| `example_7_5_multivariate_active_inference.py` | §7.5 / Alg. 7.5.1 | A 2-D vector agent uses generalized measurements and vector action to counter an exogenous attractor |
| `animation_7_5_multivariate_active_inference.py` | §7.5 animated | GIF of state, belief, action, sensory error, and free-energy evolution in the vector action-perception loop |

**Chapter 8 — Learning, Attention, and Hierarchical Models** (Part II; §8.1–§8.6)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_8_1_learning_attention.py` | §8.1 / Alg. 8.1.1 | Triple estimation: hidden state, first-order parameter, and second-order log precision all descend one VFE objective |
| `example_8_2_hierarchical_continuous.py` | §8.2–§8.4 | Two-layer continuous hierarchy: an upper context supplies a top-down prior for the lower state |
| `visualize_message_passing.py` | §8.5 | Forward prediction-error messages and backward prediction messages in the hierarchy |
| `animation_learning_attention.py` | §8.1 | GIF: state, parameter, log precision, variance, and free energy converge |

**Chapter 9 — Active Inference in POMDPs** (Part II; §9.1–§9.6, discrete)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_9_1_state_inference.py` | Example 9.1 / §9.1 | Discrete categorical Bayesian state inference (`s = σ(log Aᵀô + log D)`); a weather agent infers the state from a one-hot observation (Fig. 9.1.3) |
| `example_9_2_dynamic_filtering.py` | §9.2–§9.3 | Dynamic categorical filtering with `B`-propagated priors and per-step discrete VFE |
| `example_9_3_discrete_vfe.py` | §9.3 | Two-state simplex sweep: discrete VFE is minimized at the exact posterior and touches surprisal |
| `example_9_4_gridworld.py` | §9.4–9.5 / Alg. 9.5.1 | Grid World planning by **expected free energy** (`G = risk + ambiguity`); the agent navigates a 3×3 grid to the goal in the minimum 4 steps |
| `example_9_6_exploration_exploitation.py` | §9.6 | EFE decomposition (`risk + ambiguity`) makes exploration/exploitation visible |
| `animation_belief_filtering.py` | §9.2 | GIF: categorical beliefs update over observations |
| `animation_efe_tradeoff.py` | §9.6 | GIF: sharpening preferences shifts policy posterior from ambiguity-driven exploration toward risk-driven exploitation |

**Chapter 10 — Learning & extensions in POMDPs** (Part II; §10.1 Dirichlet `A`/`B`/`D` learning, §10.2 habit + precision, §10.3 factorial, §10.4 hierarchical)

| Script | Mirrors | What it shows |
|---|---|---|
| `example_10_1_learn_D.py` | Example 10.1 / Fig 10.1.2 | Learn the state prior `D` by accumulating initial-state beliefs; reproduces `d=[45.1,5.9]`, `D≈[0.884,0.116]` |
| `example_10_2_learn_A.py` | Example 10.2 / Fig 10.1.3 | Learn the likelihood `A` by counting (observation, state) pairs; entries converge as Dirichlet confidence grows |
| `example_10_3_learn_B.py` | Example 10.3 / Fig 10.1.4 | Learn the transition `B` by counting (next-state, current-state) pairs |
| `example_10_4_novelty.py` | Example 10.4 / §10.1 | Parameter-novelty `o·(Ws)=0.483` (information gain) + a novelty-seeking agent that learns `A` while acting (Alg. 10.1.1) |
| `example_10_5_precision.py` | Example 10.5 / Fig 10.2.3 | §10.2 policy posterior `σ(log E − γG)` swept over precision `γ` (uniform vs strong habits); reproduces Fig 10.2.3 exactly |
| `example_10_6_precision_learning.py` | Example 10.6 / Fig 10.2.4 | §10.2 learning `γ` from a Gamma prior (Eq. 23–25); `F` close to `G` ⇒ high confidence, far ⇒ low |
| `example_10_7_two_armed_bandit.py` | Example 10.7 / Figs 10.3.6–7 | §10.3 factorial two-armed bandit; the agent infers the hidden context and exploits the better machine (`--explore` = less risk-averse) |
| `example_10_8_hierarchical.py` | §10.4 / Fig 10.4.1 | §10.4 two-layer hierarchical POMDP; a slow top regime contextualizes a fast bottom layer (nested time scales) |
| `visualize_factorial_structure.py` | §10.3 / Fig 10.3.4 | Heatmaps of the two-armed bandit's factorial likelihood `A` set |
| `animation_learning.py` | §10.1 | Animated Dirichlet learning of `A`/`B` over trials |
| `animation_precision.py` | §10.2 | Animated policy-precision sweep (Fig 10.2.2) |
| `animation_bandit.py` | §10.3 | Animated two-armed bandit (context belief + policy posterior over time) |

### `docs/` — reference documentation

| Subfolder / file | Contents |
|---|---|
| `architecture.md` | Layered design diagram, key types table, conventions, guide for adding examples |
| `notation.md` | Symbol-to-identifier mapping for variables, parameters, densities, algorithms |
| `cookbook.md` | Copy-paste recipes for the 10 most-used workflows |
| `reading_order.md` | Reader-path guide (book follower, library user, contributor) |
| `chapters/` | Per-book-chapter concept maps (`chapter_01.md` … `chapter_10.md`) |
| `topics/` | Concept walkthroughs (Bayesian inference, generative models, learning, FEP, …) |
| `statistics/` | Statistical-tool references (KL, entropy, scoring rules, calibration, …) |
| `reference/` | Per-subpackage API catalogues (`core.md`, `estimators.md`, `utils.md`, `visualizations.md`) |
| `uv.md` | Quick reference for the uv workflow |

### `tests/` — pytest suite

The directory mirrors `src/active_inference/` one-for-one (see
[`tests/README.md`](tests/README.md) and [`tests/AGENTS.md`](tests/AGENTS.md)).

| Folder | Mirrors | What's covered |
|---|---|---|
| `tests/core/` | `core/` | Univariate + multivariate densities, generative process/model, grid Bayesian inference, LGS, diagnostics, posterior protocol, validators, type asserts |
| `tests/estimators/` | `estimators/` | MLE / MAP closed-form, gradient descent, OLS / BLR, factor-analysis EM monotonicity & subspace recovery |
| `tests/utils/` | `utils/` | Grid validation, path/export conventions, logger factory |
| `tests/visualizations/` | `visualizations/` | Figures save correctly, animations are valid `FuncAnimation` objects, diagnostic plots, interactive slider callbacks |
| `tests/chapters/` | `chapters/` | Subprocess smoke tests running every orchestrator with `--save` and checking raw-data sidecars |
| `tests/extras/` | `extras/` | Smoke tests for every non-interactive extras wrapper, registry/source API invariants, animation raw data, and README coverage |

## Roadmap

- [x] **Part I, Ch. 1** — hypothesis-testing brain, inverse problem demos
- [x] **Part I, Ch. 2** — hidden state estimation, MLE, MAP, gradient descent
- [x] **Part I, Ch. 3** — combining learning and inference (regression, BLR, LGS, factor-analysis EM)
- [x] **Part I, Ch. 4** — variational Bayesian inference (VFE & its 5 forms, coordinate search, fixed-form VI, mean-field CAVI)
- [x] **Part I, Ch. 5** — predictive coding (prediction-error form of VFE, recognition dynamics, univariate / multivariate / hierarchical PC, unified Ch.4–5 visualizations)
- [x] **Part II, Ch. 6** — generalized filtering for perception (§6.1: online generalized filter; §6.2: multivariate filter, Hooke's-law oscillator; §6.3–6.5: generalized coordinates of motion and the `D` shift operator; §6.6: correlated embedding-order precision `S(γ)^-1 ⊗ Π` plus Example 6.7 multivariate generalized coordinates).
- [x] **Part II, Ch. 7** — active generalized filtering (§7.1–7.4: action + preference prior, the action-perception cycle, action through the forward model, homeostatic set-point regulation; §7.5: multivariate AIF in generalized coordinates with vector action and animation).
- [~] **Part II, Ch. 8** — learning, attention, and hierarchical continuous models. §8.1: triple estimation over hidden state, first-order parameter, and second-order log precision; §8.2–§8.6: two-layer continuous hierarchy and forward/backward message passing. Further nonlinear hierarchy examples remain.
- [~] **Part II, Ch. 9** — active inference in POMDPs. §9.1: the `A`/`B`/`C`/`D` categorical generative model + exact discrete hidden-state inference (verified vs the book's Eq. 15). §9.2–9.3: dynamic filtering (HMM forward pass `forward_filter`, prediction rollout, discrete VFE) with static and animated belief-sequence figures. §9.4–9.6: **expected free energy** (risk + ambiguity, verified vs Eq. 63/68), first-class EFE component traces, policy/action selection, a Grid World planning agent (reaches goal in the minimum 4 steps), and exploration/exploitation figures/animation. 
- [x] **Part II, Ch. 10** — learning & extensions in POMDPs. §10.1: **Dirichlet** learning of `A`/`B`/`D` (Eq. 4–6, verified vs Examples 10.1–10.3), the **parameter-novelty** term (Eq. 12–15, verified vs Example 10.4 `o·(Ws)=0.483`), and the learning agent (Alg. 10.1.1). §10.2: **habit** (`E`) + **policy precision** `γ` (Eq. 22, reproduces Fig 10.2.3 exactly; γ-learning Eq. 23–25, verified vs Example 10.6). §10.3: **factorial depth** (state factors + observation modalities) and the two-armed bandit (Example 10.7) — factorial inference/EFE reduce exactly to Ch.9; the agent learns the hidden context and exploits the better arm. §10.4: **hierarchical depth** (nested POMDP layers, Eq. 39–50) — per-layer VFE/EFE reduce to the flat case; a slow top regime contextualizes a fast bottom layer.
- [ ] Part III — applications and extensions

## Contributing

Pull requests welcome. Please run `uv run pytest` (or `pytest`) before
submitting and follow the existing structure: configurable, documented
building blocks in `src/`, thin orchestrators in `chapters/`. See
[`docs/architecture.md`](docs/architecture.md) for the layer design and
conventions, and [`AGENTS.md`](AGENTS.md) for the contributor checklist.

## Citation

If you use this code in your work, please cite both this repository and the
book it follows:

```bibtex
@book{namjoshi2026fundamentals,
  title     = {Fundamentals of Active Inference: Principles, Algorithms, and Applications of the Free Energy Principle for Engineers},
  author    = {Namjoshi, Sanjeev V.},
  publisher = {MIT Press},
  year      = {2026},
  isbn      = {9780262050951}
}
```

## Community

This companion is maintained by the
[Active Inference Institute](https://activeinference.institute/).
The institute runs ongoing, free-to-join textbook reading groups in
cohorts. To take part, register at
[textbook-group.activeinference.institute](https://textbook-group.activeinference.institute/).
Pull requests and issues from group participants (and everyone else)
are welcome.

## License

MIT License. See [`LICENSE`](LICENSE).
