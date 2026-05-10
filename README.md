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
│   ├── utils/                 ← grids, logging, paths
│   ├── menu/                  ← stdlib text menu used by run.sh
│   └── web/                   ← stdlib local web UI launched by run.sh --web
├── chapters/
│   ├── chapter_01/            ← 4 concept orchestrators
│   ├── chapter_02/            ← examples 2.1–2.10 + auxiliary + 2 animations
│   └── chapter_03/            ← examples 3.1–3.7 + 8 animations + 3 diagnostic visualizations
├── docs/                      ← architecture, notation, chapter prose, topic walkthroughs, reference, statistics
├── scripts/                   ← batch runners, figure pipeline
├── tests/                     ← pytest suite (unit + chapter smoke tests)
└── output/figures/            ← regenerated PNGs / GIFs per chapter
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
# render every figure from Chapters 1–3
uv run python scripts/run_all_figures.py
uv run python scripts/run_all_figures.py --chapters 1
uv run python scripts/run_all_figures.py --chapters 2 --no-animations
uv run python scripts/run_all_figures.py --chapters 3 --clean

# unit tests
uv run pytest

# chapter smoke tests only
uv run pytest tests/chapters -v
```

Each chapter script runs standalone too:

```bash
uv run python chapters/chapter_01/01_box_scenario.py --save
uv run python chapters/chapter_02/example_2_2_linear_probabilistic.py --save
uv run python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save
uv run python chapters/chapter_02/interactive_explorer.py            # GUI window
```

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
| `core.types` | `assert_cov`, `assert_probabilities` — shape / PSD checks |
| `core.validators` | `require_1d`, `require_2d`, `require_design_matrix`, `require_finite_array`, `require_in_unit_interval`, `require_int_at_least`, `require_monotone`, `require_non_negative_scalar`, `require_positive_scalar`, `require_same_length` |
| `estimators.mle` | `mle_analytic_linear`, `mle_loss`, `mle_grad_x` |
| `estimators.map` | `map_analytic_linear`, `map_loss`, `map_grad_x` |
| `estimators.gradient_descent` | `gradient_descent`, `GradientDescentResult` |
| `estimators.linear_regression` | `mle_linear_regression`, `gd_linear_regression`, `BayesianLinearRegression`, `BLRPosterior`, `GDRegressionResult`, `add_intercept`, `squared_loss`, `squared_loss_grad` |
| `estimators.em` | `fit_factor_analysis`, `factor_analysis_e_step`, `factor_analysis_m_step`, `incomplete_log_likelihood`, `FactorAnalysisResult` |
| `utils.grids` | `make_grid`, `make_2d_grid` |
| `utils.logging` | `get_logger` — lightweight, consistent logger factory |
| `utils.io` | `default_figure_dir`, `default_data_dir`, `ensure_dir` |
| `visualizations.plotting` | `plot_prior_likelihood_posterior`, `plot_generating_function`, `plot_likelihood_ridge`, `plot_joint_heatmap`, `plot_gradient_descent`, `plot_precision_comparison`, `plot_2d_gaussian`, `confidence_ellipse`, `save_or_show` |
| `visualizations.interactive` | `interactive_inference`, `interactive_precision` — matplotlib slider widgets, no `ipywidgets` dependency |
| `visualizations.animations` | `animate_sequential_posterior`, `animate_gradient_descent`, `animate_2d_posterior`, `animate_em_convergence`, `animate_em_steps`, `animate_sufficient_statistics`, `animate_calibration_growth`, `animate_precision_sweep`, `animate_bimodal_emergence`, `animate_lgs_online`, `animate_blr_predictive_band`, `save_animation` |
| `visualizations.diagnostics` | `plot_calibration`, `plot_cdf_comparison`, `plot_coverage_curve`, `plot_kl_trace`, `plot_posterior_predictive_check`, `plot_qq`, `plot_running_statistics`, `plot_score_trace` |
| `visualizations.style` | `COLORS`, `DEFAULT_RC`, `figure_style`, `set_default_style`, `annotate_stat_box`, `stat_box_bbox` |
| `menu` | `discover_chapters`, `discover_scripts`, `run_chapter`, `run_all_chapters`, `run_script`, `main` — stdlib text menu (used by `run.sh`) |
| `web` | `run_server`, `launch`, `main` — stdlib HTTP server (used by `run.sh --web`); tab-per-chapter UI with figure galleries, render buttons, and inline docs |

Every public function/class is imported at `src/active_inference/__init__.py`
and listed in `__all__`. See [`docs/reference/`](docs/reference/) for a
subpackage-by-subpackage API catalogue.

### `chapters/` — thin orchestrators

Each script is ≤ ~120 lines, imports only from `active_inference` and
standard library, and follows the same pattern: parse args, build process
+ model, infer, plot, optionally save.

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

### `docs/` — reference documentation

| Subfolder / file | Contents |
|---|---|
| `architecture.md` | Layered design diagram, key types table, conventions, guide for adding examples |
| `notation.md` | Symbol-to-identifier mapping for variables, parameters, densities, algorithms |
| `cookbook.md` | Copy-paste recipes for the 10 most-used workflows |
| `reading_order.md` | Reader-path guide (book follower, library user, contributor) |
| `chapters/` | Per-book-chapter concept maps (`chapter_01.md`, `chapter_02.md`, `chapter_03.md`) |
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
| `tests/utils/` | `utils/` | Grid validation, path conventions, logger factory |
| `tests/visualizations/` | `visualizations/` | Figures save correctly, animations are valid `FuncAnimation` objects, diagnostic plots |
| `tests/chapters/` | `chapters/` | Subprocess smoke tests running every orchestrator with `--save` |

## Roadmap

- [x] **Part I, Ch. 1** — hypothesis-testing brain, inverse problem demos
- [x] **Part I, Ch. 2** — hidden state estimation, MLE, MAP, gradient descent
- [x] **Part I, Ch. 3** — combining learning and inference (regression, BLR, LGS, factor-analysis EM)
- [ ] Part I, Ch. 4 — variational Bayesian inference
- [ ] Part I, Ch. 5 — predictive coding
- [ ] Part II — generalized filtering, active generalized filtering, POMDP
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
