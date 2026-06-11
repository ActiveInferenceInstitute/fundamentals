# Notation cheatsheet

Quick reference for the symbols used throughout the codebase. The notation
follows the conventions established in the book and Appendix B; this file is
descriptive — it explains how the symbols map to identifiers in the package.

## Variables

| Symbol     | Identifier        | Meaning |
|------------|-------------------|---------|
| `x*`       | `x_star`, `x_true`| External (true) state of the environment. |
| `x`        | `x`               | Agent's *hidden state* — its belief about `x*`. |
| `y`        | `y`, `y_obs`      | Observation produced by the generative process. |
| `Y`        | `Y`               | Stacked observations as an `(N, D)` data matrix. |
| `X`        | `X`               | Design matrix `(N, C)` (or `(N, C+1)` with intercept). |
| `Θ` / `θ`  | `Theta` / `theta` | Mixing matrix (LGS / FA) or parameter vector (regression). |
| `ζ`        | `mu_zeta`         | Log precision for Chapter 8 attention / second-order parameter learning. |
| `μ̃_x`      | `mu_tilde`, `mus` | Generalized-coordinate hidden-state belief; public result tensors use `(time, embedding_order, variable)`. |
| `ỹ`        | `y_tilde`, `y_tildes` | Generalized measurements built from observations and their finite-difference derivatives. |
| `a`        | `action`, `actions` | Continuous action/control signal in Chapter 7 active generalized filtering. |
| `b`        | `b`               | Optional offset added to the linear generator. |
| `g_E`      | `process.mean()`  | Generating function inside the environment. |
| `g_M`      | `model.predict_mean()` | Agent's generating function. |
| `psi(x)`   | `psi`             | Optional nonlinear transform of `x`. |
| `omega_y`  | (sampled noise)   | Zero-mean Gaussian observation noise with variance `sigma2_y` (uni) or `cov_y` (multi). |

## Parameters

| Symbol     | Identifier            | Meaning |
|------------|------------------------|---------|
| `beta0`    | `beta0`                | Linear intercept of the generating function. |
| `beta1`    | `beta1`                | Linear slope. |
| `sigma2_y` | `sigma2_y`             | Variance of the likelihood (observation noise). |
| `m_x`      | `m_x`                  | Prior mean over the hidden state. |
| `s2_x`     | `s2_x`                 | Prior variance over the hidden state. |
| `theta*`   | `process.theta`        | Parameters of the generative process. |
| `theta`    | model attributes       | Parameters of the generative model. |
| `λ` / `Π`  | `lambda_*`, `precision_*` | Precision (inverse variance/covariance); Chapter 8 uses `exp(mu_zeta)`. |
| `Π̃(γ)`    | `correlated_embedding_precision(...)` | Generalized precision with correlated embedding orders. |
| `S(γ)`     | `gaussian_temporal_covariance(...)` | Gaussian temporal covariance over embedding orders. |
| `γ`        | `gamma`                 | Smoothness/roughness parameter controlling embedding-order correlations. |
| `A`        | `A`                     | Discrete likelihood matrix in POMDP and factor-graph examples. |
| `B`        | `B`, `transition`       | Discrete transition matrix or transition-factor stack. |
| `C`        | `C`, `preferences`      | Observation preferences, either raw log preferences or normalized probabilities depending on context. |
| `D`        | `D`, `prior`            | Initial categorical state prior. |
| `π`        | `policy`, `policies`    | Candidate action sequence in POMDP planning and policy-tree search. |
| `α`        | `alpha`, `counts`       | Dirichlet/Gamma concentration or pseudocount parameters, depending on context. |

## Densities

| Symbol            | Identifier in code             | Notes |
|-------------------|--------------------------------|-------|
| `p(y \| x)`       | `model.log_likelihood(...)`    | Likelihood / observation model. |
| `p(x)`            | `model.log_prior(...)`         | Hidden-state prior. |
| `p(x, y)`         | `inferer.joint_density(...)`   | Joint distribution. |
| `log p(y)`        | `result.log_evidence`          | Log marginal likelihood (model evidence); `surprisal = −log p(y)`. |
| `p(x \| y)`       | `result.posterior`             | Posterior on the grid. |

## Algorithms

| Acronym | Identifier | Where |
|---------|------------|-------|
| MLE     | `mle_analytic_linear`, `mle_loss`, `mle_grad_x`     | `estimators/mle.py` |
| MAP     | `map_analytic_linear`, `map_loss`, `map_grad_x`     | `estimators/map.py` |
| GD      | `gradient_descent`                                  | `estimators/gradient_descent.py` |
| OLS / multiple regression | `mle_linear_regression`           | `estimators/linear_regression.py` |
| BLR     | `BayesianLinearRegression`                          | `estimators/linear_regression.py` |
| LGS     | `LinearGaussianSystem`                              | `core/lgs.py` |
| EM (FA) | `fit_factor_analysis`, `factor_analysis_e_step`/`_m_step` | `estimators/em.py` |
| MVN     | `mvn_pdf`, `mvn_log_pdf`, `mvn_sample`, `mahalanobis_squared` | `core/distributions.py` |
| Pipeline | `Pipeline`, `Pipeline.linear_gaussian`             | `core/compose.py` |
| Running stats | `running_stats`, `RunningPosteriorStats`       | `core/compose.py` |
| VFE (Ch.4) | `variational_free_energy`, `coordinate_search_vfe`, `fixed_form_vi`, `free_form_cavi` | `core/variational.py`, `estimators/variational.py` |
| PC (Ch.5) | `predictive_coding_inference`, `multivariate_predictive_coding`, `hierarchical_predictive_coding`, `pc_free_energy_grad`, `pc_linear_fixed_point` | `core/predictive_coding.py`, `estimators/predictive_coding.py` |
| GF (Ch.6) | `generalized_filter`, `generalized_filter_gc`, `generalized_vector_filter`, `correlated_embedding_precision` | `core/generalized_filtering.py`, `estimators/generalized_filtering.py` |
| AIF (Ch.7) | `simulate_active_inference`, `simulate_multivariate_active_inference`, `multivariate_action_gradient` | `core/active_inference.py`, `estimators/active_inference.py` |
| AGF learning/attention (Ch.8) | `learning_attention_free_energy`, `simulate_learning_attention`, `hierarchical_message_terms` | `core/continuous_learning.py`, `estimators/continuous_learning.py` |
| POMDP planning (Ch.9–11) | `tree_policy_search`, `sophisticated_policy_trace`, `simulate_sophisticated_planning` | `core/pomdp.py`, `core/pomdp_extensions.py`, `estimators/pomdp_extensions.py` |
| Appendix probability/math (App. B-C) | `normalize_categorical`, `joint_from_likelihood_prior`, `gamma_pdf`, `dirichlet_mean`, `mutual_information`, `euler_integrate` | `core/appendix_math.py` |
| Colored noise (App. C.9) | `squared_exponential_covariance`, `colored_noise_precision`, `sample_colored_noise` | `core/noise.py` |
| Model comparison (App. C.11) | `model_posterior`, `bayes_factor`, `bayesian_model_average`, `bayesian_model_reduction` | `core/model_comparison.py` |
| Factor graph messages (Ch.12) | `sum_product_chain`, `forward_backward_smoothing`, `variational_message_update`, `marginal_message_passing`, `active_inference_factor_messages` | `core/factor_graph.py` |
| Applications (Ch.13) | `simulate_robot_navigation`, `simulate_fault_tolerant_control`, `simulate_social_inference`, `robotics_theory_landscape` | `estimators/applications.py` |
| Bayesian mechanics (Ch.14) | `bayesian_mechanics_summary`, `simulate_markov_blanket_flow`, `blanket_coupling_matrix`, `viability_indicator`, `entropy_vfe_bound_curve` | `core/bayesian_mechanics.py` |

## Information-theoretic / FEP quantities

| Symbol      | Identifier | Notes |
|-------------|-----------|-------|
| ``H[p]``    | `grid_entropy`, `gaussian_entropy_univariate/_mvn`, `InferenceResult.entropy` | Differential entropy in nats. |
| ``KL[p‖q]`` | `grid_kl_divergence`, `gaussian_kl_univariate/_mvn`, `InferenceResult.kl_from_prior` | Non-negative; asymmetric. |
| ``F[q]``    | derived: `−result.log_evidence + result.kl_from_prior()` | Variational free energy. |
| ``G(π)``    | `expected_free_energy`, `evaluate_policy`, `tree_policy_search.expected_free_energies` | Expected free energy of a future state or action sequence. |
| static/dynamic/expected free energy | `static_vfe_decomposition`, `dynamic_vfe_decomposition`, `expected_free_energy_decomposition` | Appendix D executable free-energy forms. |
| ``H[p] <= F`` | `entropy_upper_bound_from_vfe`, `bayesian_mechanics_summary` | Chapter 14 entropy/VFE-style bound summary. |
| log-score   | `log_score_gaussian` | Higher is better. |
| CRPS        | `crps_gaussian` | Lower is better. |
| ESS         | `effective_sample_size` | Kish ESS in log space. |

## Conventions in the code

- A trailing ``_grid`` denotes a 1-D NumPy array of points.
- Functions that work in log-space are suffixed ``_log`` (or named
  ``log_likelihood`` / ``log_prior``).
- Parameters that the book denotes with a star (``*``) — i.e. quantities of
  the *generative process* — are spelled out in code as ``x_star``,
  ``beta0_true``, etc., to avoid confusion with Python's unpacking operator.
- The inspected PDF source spine is explicit in `active_inference.source_spine`:
  Chapters 1-14 and Appendices A-D are present; Chapter 15 is absent.
