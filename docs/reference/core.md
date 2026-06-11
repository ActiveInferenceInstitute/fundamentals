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
| `calibration_curve(truths, lower_fn, upper_fn, nominal_levels)` | arrays + 2 callables → `CalibrationCurve` | Reliability sweep. |
| `CalibrationCurve` | dataclass | `nominal`, `empirical`, `n_trials`, `calibration_error()`. |
| `posterior_predictive_check(observed, replicates, statistic)` | arrays + callable → `PosteriorPredictiveCheck` | Two-sided p-value. |
| `PosteriorPredictiveCheck` | dataclass | `observed`, `replicated`, `p_value`, `summary()`. |
| `normal_ci(mean, sigma2, level)` | floats → (lo, hi) | Equal-tailed Gaussian CI via `scipy.special.erfinv`. |
| `standardize(samples)` | (N, D) → (N, D) | Columnwise z-score with ddof=1. |
| `gradient_check(f, grad, x0, *, eps=1e-5)` | callables + point → float | Max abs error between an analytic gradient and a central finite difference (used to validate the Ch.5 PC gradient). |
| `convergence_report(trace, *, tol=1e-9)` | (T,) → `ConvergenceReport` | Monotonicity, total decrease, max increase, and empirical linear-convergence rate of a descent trace. |
| `ConvergenceReport` | dataclass | `monotone`, `final`, `total_decrease`, `n_steps`, `rate`, `max_increase`. |
| `oracle_agreement(estimate, oracle, *, tol=1e-2)` | floats → `OracleAgreement` | Compares an estimate to an independent oracle (e.g. PC fixed point vs grid posterior mean). |
| `OracleAgreement` | dataclass | `estimate`, `oracle`, `abs_error`, `agree`. |

## `core.appendix_math`

Executable Appendix B/C probability, information, and dynamical-system helpers.

| Symbol | Role |
|---|---|
| `normalize_categorical(values, axis=None)` | Validate and normalize non-negative categorical weights. |
| `joint_from_likelihood_prior(likelihood, prior)` | Build `p(y, x) = p(y | x) p(x)` from an Appendix B/C likelihood orientation. |
| `marginalize(joint, axis)` | Sum out one tensor axis. |
| `bayes_posterior_from_likelihood(likelihood, prior, observation_index)` | Categorical Bayes update for one observed outcome. |
| `gamma_pdf(x, shape, rate)` | Gamma density in shape-rate parameterization. |
| `dirichlet_mean(alpha)` | Dirichlet mean vector. |
| `dirichlet_variance(alpha)` | Per-component Dirichlet variance. |
| `mutual_information(joint)` | Discrete mutual information from a joint probability table. |
| `maximum_entropy_distribution(n)` | Uniform `n`-category maximum-entropy distribution. |
| `euler_integrate(flow, x0, *, dt, steps)` | Deterministic Euler integration for small dynamical examples. |
| `jensen_gap(fn, samples)` | Mean `f(x)` minus `f(mean x)` for Jensen-inequality diagnostics. |
| `kronecker_delta(i, j)` | Discrete identity helper. |
| `dirac_delta_grid(grid, location)` | Grid-normalized delta proxy for Appendix C identity demos. |

## `core.model_comparison`

Appendix C.11 model evidence, Bayes factor, and model-reduction helpers.

| Symbol | Role |
|---|---|
| `model_posterior(log_evidence, log_prior=None)` | Softmax posterior over candidate models. |
| `log_bayes_factor(log_evidence_a, log_evidence_b)` | Log Bayes factor. |
| `bayes_factor(log_evidence_a, log_evidence_b)` | Bayes factor in ordinary scale. |
| `bayesian_model_average(values, log_evidence, log_prior=None)` | Evidence-weighted average over model-specific values. |
| `bayesian_model_reduction(full_log_evidence, reduced_log_prior, full_log_prior=None)` | Score reduced models by replacing priors under full-model evidence. |
| `bayesian_model_expansion(reduced_log_evidence, expansion_gain)` | Score expanded models from reduced evidence plus a gain term. |

## `core.noise`

Colored-noise and finite-difference utilities used by Appendix C and
generalized-coordinate demos.

| Symbol | Role |
|---|---|
| `squared_exponential_covariance(time, *, length_scale, variance=1.0)` | Smooth covariance over time points. |
| `colored_noise_precision(time, *, length_scale, variance=1.0, jitter=1e-8)` | Symmetric inverse covariance with diagonal jitter. |
| `sample_colored_noise(time, *, length_scale, variance=1.0, rng=None)` | Reproducible smooth colored-noise sample. |
| `finite_difference_derivative(values, time)` | Stable first derivative on a strictly increasing grid. |

## `core.free_energy_forms`

Pedagogical decompositions for comparing free-energy variants without claiming
one universal literature convention for every form.

| Symbol | Role |
|---|---|
| `FreeEnergyForm` | Frozen dataclass with `name`, scalar `total`, additive `terms`, and `term_vector(order)`. |
| `expected_free_energy_form(risk, ambiguity, epistemic_value=0.0)` | Returns EFE as `risk + ambiguity - epistemic_value`. |
| `free_energy_of_future(present_free_energy, expected_future_free_energy)` | Adds present free energy and expected future free energy. |
| `observed_predicted_free_energy(observed_free_energy, predicted_free_energy, observation_weight=0.5)` | Convex blend of observed and predicted terms. |
| `generalized_free_energy_form(present_free_energy, future_free_energy, information_gain=0.0)` | Returns `present + future - information_gain`. |
| `bethe_free_energy_form(factor_energy, variable_entropy, consistency_penalty=0.0)` | Bethe-style energy-minus-entropy form with caller-supplied consistency penalty. |
| `constrained_bethe_free_energy_form(factor_energy, variable_entropy, constraint_violation, constraint_weight=1.0)` | Bethe-style form with an explicit weighted constraint penalty. |
| `action_perception_divergence_form(perception_error, action_error, coupling=0.0)` | Coupled action-perception divergence teaching form. |
| `free_energy_of_expected_future(expected_free_energy, temporal_precision=1.0)` | Scale a future free-energy trajectory by temporal precision. |
| `static_vfe_decomposition(energy, entropy)` | Appendix D static VFE `energy - entropy`. |
| `dynamic_vfe_decomposition(state_error, sensory_error, flow_complexity=0.0)` | Appendix D dynamic VFE components. |
| `expected_free_energy_decomposition(risk, ambiguity, epistemic_value=0.0)` | Appendix D EFE components and total. |
| `renyi_bound(probabilities, energies, alpha)` | Renyi-style certainty-equivalent energy for `alpha != 1`. |
| `renyi_limit_energy(probabilities, energies)` | Alpha-to-one Renyi limit, equal to probability-weighted expected energy. |
| `free_energy_variant_table(risk, ambiguity, epistemic_value)` | Returns comparable policy-indexed arrays for EFE, FEF, and GFE sweeps. |

## `core.factor_graph`

Small categorical factor-graph helpers for extras and teaching examples.

| Symbol | Role |
|---|---|
| `normalize_message(message)` | Validate and normalize a non-negative 1-D categorical message. |
| `categorical_factor_message(factor, incoming, target_axis)` | Sum-product factor-to-variable message for one categorical target axis. |
| `sum_product_chain(prior, transitions, likelihoods)` | Normalized forward messages for a categorical state-space chain. |
| `variational_message_update(log_factor, incoming_expectations, target_axis)` | VMP-style categorical softmax update for one variable. |
| `FactorGraphChain` | Frozen categorical chain container with normalized prior, transition, and likelihood messages. |
| `backward_smoothing_messages(transitions, likelihoods)` | Normalized backward messages for a categorical chain. |
| `forward_backward_smoothing(prior, transitions, likelihoods)` | Forward-backward smoothed categorical state beliefs. |
| `marginal_message_passing(factors, incoming)` | Approximate marginal messages from local factors and incoming beliefs. |
| `active_inference_factor_messages(likelihood, prior, preferences)` | Preference-weighted categorical messages for active-inference factor graphs. |
| `learning_attention_message(evidence, precision)` | Precision-weighted evidence message for learning/attention examples. |
| `hybrid_model_bridge(continuous_state, discrete_belief)` | Outer-product bridge from continuous features to a categorical belief. |

## `core.pomdp_extensions`

Small Chapter 11 helpers that build on `core.pomdp` without changing the
established `POMDPModel` import surface.

| Symbol | Role |
|---|---|
| `time_dependent_preferences(preferences)` | Softmax-normalize a `(T, O)` schedule of future observation preferences. |
| `forget_dirichlet_counts(counts, *, rate, floor=1.0)` | Apply bounded exponential forgetting to Dirichlet pseudocounts. |
| `structure_log_evidence(scores, complexity=0.0)` | Penalize candidate structure scores by model complexity. |
| `structure_posterior(scores, complexity=0.0)` | Softmax posterior over candidate structures. |
| `update_preference_counts(counts, observations, learning_rate=1.0)` | Accumulate categorical preference pseudocounts from observations. |
| `habit_prior_from_counts(counts, concentration=1.0)` | Convert action-count evidence into a normalized habit prior. |
| `path_based_policy_scores(paths, preferences, transition_cost=0.0)` | Score candidate state paths by preferences minus transition cost. |
| `TreePolicySearchResult` | Policies, expected free energies, posterior probabilities, and selected best policy. |
| `tree_policy_search(model, belief, policies, preferences, *, gamma=4.0)` | Score bounded policy trees by EFE and return a policy posterior. |
| `SophisticatedInferenceTrace` | Policy rollout, future beliefs, belief entropies, and total expected free energy. |
| `sophisticated_policy_trace(model, belief, policy, preferences)` | Roll one policy forward to expose future-belief diagnostics. |

## `core.ergodic`

Ergodic-density and entropy-bound helpers for FEP extras topics.

| Symbol | Role |
|---|---|
| `EntropyBound` | Frozen dataclass storing `entropy`, `upper_bound`, and non-negative `gap`. |
| `ergodic_density(trajectory, *, bins=80, bounds=None)` | Histogram estimate of a normalized one-dimensional ergodic density. |
| `density_entropy(x_grid, density)` | Differential entropy `-int p(x) log p(x) dx` on a grid. |
| `entropy_upper_bound_from_vfe(entropy, vfe_upper_bound)` | Build an `EntropyBound` from an entropy and VFE-like upper bound. |
| `ergodic_ou_trajectory(n_steps=500, drift=0.08, diffusion=0.25, initial=2.5)` | Deterministic-noise OU teaching trajectory for reproducible demos. |

## `core.bayesian_mechanics`

Compact Chapter 14 helpers for ergodic-density, entropy-bound, and
Markov-blanket demonstrations.

| Symbol | Role |
|---|---|
| `MarkovBlanketFlow` | Time, external, blanket, and internal trajectories for a simple blanket flow. |
| `simulate_markov_blanket_flow(n_steps=160)` | Deterministic coupled trajectories with phase-separated external, blanket, and internal states. |
| `BayesianMechanicsSummary` | Ergodic-density grid, density, entropy, upper bound, and residual gap. |
| `bayesian_mechanics_summary(trajectory, *, bins=80, vfe_margin=0.5)` | Convert a trajectory into an ergodic density plus entropy/VFE-bound summary. |
| `blanket_coupling_matrix(flow)` | Correlation matrix among external, blanket, and internal trajectories. |
| `viability_indicator(states, lower, upper)` | Binary viability mask for survival/viability demos. |
| `survival_probability(viable)` | Fraction of time spent in viable states. |
| `entropy_vfe_bound_curve(entropies, margins)` | Entropy, upper-bound, and residual-gap arrays. |
| `phase1_fep_bridge(entropy, vfe, action_flow)` | Compact Phase-I FEP bridge summary over entropy, VFE, and action flow. |

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

## `core.variational` — variational free energy (Chapter 4)

Variational Bayesian inference: score a variational density `q(x)` by
**variational free energy** `𝓕` and minimize it to approximate the posterior.
The same grid + trapezoid scheme as `core.inference` is used, so VFE results are
directly comparable to the exact grid posterior, which serves as an oracle.

```python
from active_inference import (
    GaussianBelief, LinearGaussianModel, variational_free_energy,
)
import numpy as np

model = LinearGaussianModel(beta0=3, beta1=2, sigma2_y=0.25, m_x=4, s2_x=0.25)
grid  = np.linspace(-6, 12, 2001)
comp  = variational_free_energy(GaussianBelief(2.4, 0.05), model, 7.0, grid)

comp.check()                 # asserts all five forms agree to grid precision
print(comp.free_energy, comp.surprisal, comp.bound_gap)  # 𝓕 ≈ −log p(y), gap ≈ 0
```

| Symbol | Role |
|---|---|
| `GaussianBelief(mu, var)` | Parametric variational density `q(x)=N(mu, var)`; `pdf`, `logpdf`, `entropy`, `std`, `sample`. Validated (finite `mu`, positive `var`). |
| `evaluate_q(q, x_grid, normalize=True)` | Evaluate a belief / density array / callable on the grid as a (normalized) density. |
| `variational_free_energy(q, model, y, x_grid, *, log_evidence=None, posterior=None, normalize_q=True)` | Compute VFE and **all** decompositions → `VFEComponents`. Pass `log_evidence` in inner loops to skip recomputing the oracle. |
| `VFEComponents` | Frozen dataclass: `free_energy`, `divergence`, `surprisal`, `complexity`, `accuracy`, `neg_entropy`, `energy`, `log_evidence`; properties `g_form`/`d_form`/`c_form`/`e_form`/`bound_gap`; `.check(atol)` and `.summary()`. |
| `vfe_g_form` / `vfe_d_form` / `vfe_c_form` / `vfe_e_form` | Thin wrappers returning `𝓕` (and the named terms) for one form. |
| `vfe_map_form(model, y, mu)` / `vfe_mle_form(model, y, mu)` | Point-mass special cases (Eq. 30/31). |
| `log_model_evidence(model, y, x_grid)` | `log p(y)` on the grid (the oracle). |
| `surprisal(model, y, x_grid)` | `−log p(y)`. |
| `free_energy_bound_gap(q, model, y, x_grid)` | Slack in `𝓕 ≥ −log p(y)`; equals `D_KL(q‖p(x\|y))`, zero at the posterior. |

The five forms (G/D/C/E plus MAP/MLE) are algebraically equal; `VFEComponents`
computes each from a different grouping of the same integrals and `.check()`
verifies their agreement. The D-form builds `log p(x|y) = log p(x,y) − log p(y)`
directly (not `log` of the normalized posterior array) so it stays stable even
for beliefs far in the tail where the posterior array underflows.

## `core.thermodynamics` — thermodynamic / FEP bridge

Small helpers for the explicit analogy used by the extras topic scripts:
`U = E_q[-log p(x,y)]`, `S = H[q]`, and `A = U - T S`. At unit temperature
and zero pressure-volume contribution, `A` equals variational free energy. See
the [thermodynamic bridge topic](../topics/thermodynamic_bridge.md) for the
equation flow and analogy boundary.

| Symbol | Role |
|---|---|
| `ThermodynamicState` | Frozen dataclass storing `energy`, `entropy`, `temperature`, `pressure`, and `volume`; properties expose `helmholtz_free_energy`, `enthalpy`, and `gibbs_free_energy`. |
| `canonical_probabilities(energies, temperature=1.0)` | Stable Boltzmann probabilities proportional to `exp(-energy / temperature)`. |
| `expected_energy(probabilities, energies)` | Probability-weighted internal energy. |
| `boltzmann_entropy(probabilities)` | Discrete entropy `-sum p log p` in nats. |
| `helmholtz_free_energy(energy, entropy, temperature=1.0)` | `A = U - T S`. |
| `enthalpy(energy, pressure=0.0, volume=0.0)` | `H = U + pV`; pressure-volume terms are caller supplied. |
| `gibbs_free_energy(energy, entropy, temperature=1.0, pressure=0.0, volume=0.0)` | `G = H - T S`. |
| `vfe_thermodynamic_state(q, model, y, x_grid, ...)` | Converts `variational_free_energy` components into a `ThermodynamicState`. |

## `core.predictive_coding` — predictive coding (Chapter 5)

The MAP/Laplace specialization of VFE used by predictive coding: a generative
function `g`, two precision-weighted prediction errors, and the analytic gradient
that drives recognition dynamics. See the
[Chapter 5 concept map](../chapters/chapter_05.md).

| Symbol | Signature | Purpose |
|---|---|---|
| `GenerativeFunction` | ABC | `__call__(x)` + `derivative(x)`; the generating map `g` and its slope. |
| `LinearFunction(slope=1, intercept=0)` | callable | `g(x)=slope·x+intercept`, `g'=slope`. |
| `QuadraticFunction(a=1, b=0)` | callable | `g(x)=a·x²+b`, `g'=2a·x`. |
| `GenericFunction(fn, dfn=None, eps=1e-5)` | callable | Arbitrary `g`; central-difference derivative if `dfn` omitted. |
| `PredictiveCodingModel(g, sigma2_y=0.25, m_x=4.0, s2_x=0.25)` | dataclass | The MAP model; properties `lambda_y`, `lambda_x`; `predict(mu)`. Validates positive variances. |
| `sensory_prediction_error(model, y, mu)` | floats → float | `ε_y = y − g(μ)` (Eq. 6a). |
| `state_prediction_error(model, mu)` | floats → float | `ε_x = μ − m_x` (Eq. 6b). |
| `predictive_coding_free_energy(model, y, mu)` | floats → `PCFreeEnergy` | MAP free energy `F` + its parts (Eq. 7a). |
| `PCFreeEnergy` | dataclass | `free_energy`, `eps_y`, `eps_x`, `weighted_eps_y`, `weighted_eps_x`, `mu_y`; `summary()`. |
| `pc_free_energy_grad(model, y, mu)` | floats → float | Analytic `∂F/∂μ = λ_x ε_x − λ_y ε_y g'(μ)` (Eq. 16, sign **derived**). |
| `pc_free_energy_grad_fd(model, y, mu, eps=1e-5)` | floats → float | Central finite-difference gradient (the numerical oracle). |
| `pc_linear_fixed_point(model, y)` | floats → float | Closed-form recognition fixed point `μ* = (λ_x m_x + λ_y a(y−b))/(λ_x + λ_y a²)` for a `LinearFunction` g (= the Gaussian posterior mean). Raises `TypeError` for nonlinear g. The analytical landmark figures annotate. |
| `pc_curvature_linear(model)` | model → float | Local curvature `L = λ_x + g'² λ_y` of `F`; exact `∂²F/∂μ²` for linear g. Fixed-step PC converges iff `κ < 2/L`, contracting by `\|1−κL\|` per step. |

## `core.generalized_filtering` — generalized filtering (Chapter 6)

The dynamic state-space model of §6.1: the static prior of Chapter 5 becomes a
state-transition function `f_M`, and perception is online gradient-flow recognition.
See the [Chapter 6 concept map](../chapters/chapter_06.md).

| Symbol | Signature | Purpose |
|---|---|---|
| `DynamicStateSpaceModel(f, g, s2_x=1, sigma2_y=1)` | dataclass | Dynamic model: state-transition `f_M` + observation `g_M` (`GenerativeFunction`s) with state/sensory variances; properties `lambda_x`, `lambda_y`, `predict_state`, `predict_obs`. |
| `gf_state_prediction_error(model, mu)` | floats → float | `ε_x = μ − f_M(μ)` (Eq. 6). |
| `gf_sensory_prediction_error(model, y, mu)` | floats → float | `ε_y = y − g_M(μ)` (Eq. 6). |
| `gf_free_energy(model, y, mu)` | floats → float | Laplace VFE `F = ½(λ_y ε_y² + λ_x ε_x² + log(σ_y² s_x²))` (Eq. 7a). |
| `gf_free_energy_grad(model, y, mu)` | floats → float | Analytic `∂F/∂μ = λ_x ε_x(1−f_M'(μ)) − λ_y ε_y g_M'(μ)` (derived, FD-verified). |
| `gf_free_energy_grad_fd(model, y, mu, eps=1e-5)` | floats → float | Central finite-difference gradient (numerical oracle). |
| `gf_fixed_point_linear(model, y)` | floats → float | Closed-form recognition fixed point for linear `f_M`/`g_M` (the steady-state landmark). Raises `TypeError` otherwise. |

### Multivariate filter (§6.2)

| Symbol | Signature | Purpose |
|---|---|---|
| `VectorFunction` / `LinearVectorFunction(A, b)` / `GenericVectorFunction(fn, out_dim, eps)` | — | Differentiable maps `ℝ^C→ℝ^k` with a Jacobian (`LinearVectorFunction` constant `A`; `GenericVectorFunction` central-difference Jacobian). |
| `MultivariateDynamicModel(f, g, precision_x, precision_y, dim_x=2, dim_y=2)` | dataclass | Vector dynamic model; precisions accept scalar/diagonal-vector/matrix → `Pi_x`, `Pi_y`. |
| `mv_gf_free_energy(model, y, mu)` | arrays → float | `F = ½(ε_xᵀΠ_x ε_x + ε_yᵀΠ_y ε_y + log\|Σ_xΣ_y\|)` (Eq. 12). |
| `mv_gf_free_energy_grad(model, y, mu)` | arrays → (C,) | Analytic `(I−J_f)ᵀΠ_x ε_x − J_gᵀΠ_y ε_y` (derived, FD-verified). |
| `mv_gf_free_energy_grad_fd(model, y, mu, eps=1e-5)` | arrays → (C,) | Finite-difference gradient (numerical oracle). |
| `mv_gf_fixed_point_linear(model, y)` | array → (C,) | Closed-form fixed point for linear `f`/`g` (a linear solve). |

### Generalized coordinates of motion (§6.3–6.5)

| Symbol | Signature | Purpose |
|---|---|---|
| `shift_operator(embedding_dim, n_states=1)` | ints → (M,M) | The derivative shift operator `D` (Eq. 37); `D μ̃ = μ̃'`. `D = I_C ⊗ S`. |
| `embed_flow(f_val, f_prime, mu_tilde)` | floats + (M,) → (M,) | Embed a function into generalized coordinates: `[f, f'·μ', f'·μ'', …]` (Eq. 30/36). |
| `GeneralizedModel(f, g, precision_x, precision_y, embedding_dim=3)` | dataclass | Univariate model in generalized coordinates; `D`, `Pi_x`, `Pi_y`, `embed_f`, `embed_g`. |
| `generalized_state_error(model, mu_tilde)` | (M,) → (M,) | `ε̃_x = D μ̃ − f̃(μ̃)` (Eq. 46b). |
| `generalized_sensory_error(model, y_tilde, mu_tilde)` | (M,),(M,) → (M,) | `ε̃_y = ỹ − g̃(μ̃)` (Eq. 46a). |
| `generalized_free_energy(model, y_tilde, mu_tilde)` | → float | Generalized VFE (Eq. 45). |
| `generalized_free_energy_grad(model, y_tilde, mu_tilde)` | → (M,) | `(D−f'I)ᵀΠ̃_x ε̃_x − (g'I)ᵀΠ̃_y ε̃_y` (Eq. 50a; local-linearity, exact for linear). |
| `generalized_free_energy_grad_fd(model, y_tilde, mu_tilde, eps=1e-5)` | → (M,) | Finite-difference gradient (numerical oracle). |

### Correlated embedding orders and vector generalized coordinates (§6.6)

| Symbol | Signature | Purpose |
|---|---|---|
| `gaussian_temporal_covariance(embedding_dim, gamma)` | int, float → (M,M) | Gaussian temporal covariance `S(γ)` from autocorrelation derivatives at zero. |
| `correlated_embedding_precision(precision, embedding_dim, *, gamma, layout="state_major")` | scalar/vector/matrix → full matrix | Generalized precision `S(γ)^-1 ⊗ Π` (book/order-major) or `Π ⊗ S(γ)^-1` (repo state-major). |
| `flatten_generalized_coordinates(values)` / `unflatten_generalized_coordinates(values, embedding_dim, dim)` | `(M,D)` ↔ `(M·D,)` | Convert between educational tensors and state-major vectors used by `D`. |
| `GeneralizedVectorModel(f, g, precision_x, precision_y, embedding_dim=3, dim_x=2, dim_y=2)` | dataclass | Vector generalized-coordinate model with full correlated precisions. |
| `embed_vector_flow(fn, mu_tilde, embedding_dim, dim_x)` | vector model pieces → state-major vector | Local-linear vector embedding for `f̃` or `g̃`. |
| `generalized_vector_state_error` / `generalized_vector_sensory_error` | arrays → arrays | Vector `ε̃_x`, `ε̃_y` in state-major layout. |
| `generalized_vector_free_energy` / `generalized_vector_free_energy_grad` / `_fd` | arrays → scalar / vector | Eq. 61 full-matrix VFE and finite-difference-verified gradient. |

## `core.continuous_learning` — learning, attention, and hierarchy (Chapter 8)

Chapter 8 treats first-order model parameters and second-order precision parameters as
hidden variables alongside continuous hidden states. The companion keeps the model small
and finite-difference-verifiable while preserving the book's time-scale separation.

| Symbol | Signature | Purpose |
|---|---|---|
| `log_precision_to_precision(zeta)` | scalar/array → scalar/array | Convert log precision `ζ` to positive precision `λ=exp(ζ)`. |
| `log_precision_to_variance(zeta)` | scalar/array → scalar/array | Convert log precision to positive variance `σ²=exp(-ζ)`. |
| `LearningAttentionModel(state_attractor, theta_prior_mean=0, zeta_prior_mean=0, sigma2_y=1, sigma2_theta=1, sigma2_zeta=1)` | dataclass | Compact triple-estimation model with observation prediction `mu_x - mu_theta`. |
| `LearningAttentionState(mu_x, mu_theta, mu_zeta)` | dataclass | Variational means for hidden state, first-order parameter, and log precision. |
| `LearningAttentionComponents` | dataclass | VFE, prediction errors, learned precision/variance, and per-term contributions. |
| `LearningAttentionGradient` | dataclass | Gradient components for `(mu_x, mu_theta, mu_zeta)` with `as_vector()`. |
| `learning_attention_free_energy(model, y, state)` | → `LearningAttentionComponents` | Chapter 8 free energy for perception, learning, and attention. |
| `learning_attention_grad(model, y, state)` | → (3,) | Analytic gradient w.r.t. `(mu_x, mu_theta, mu_zeta)`. |
| `learning_attention_grad_fd(model, y, state, eps=1e-5)` | → (3,) | Finite-difference gradient oracle. |
| `ContinuousHierarchyLayer(obs_offset=0, sensory_precision=1)` | dataclass | Lower continuous layer with `y = mu_x - obs_offset`. |
| `HierarchicalContinuousModel(lower, link_precision=1, context_prior_mean=0, context_precision=1, link_gain=1)` | dataclass | Two-layer hierarchy: upper context predicts lower state through `link_gain`. |
| `HierarchicalMessageTerms` | dataclass | Sensory/link/context errors, weighted messages, top-down prediction, and gradient. |
| `hierarchical_continuous_free_energy(model, y, belief)` | → float | Two-layer continuous VFE. |
| `hierarchical_continuous_grad(model, y, belief)` | → (2,) | Analytic gradient for lower state and upper context. |
| `hierarchical_continuous_grad_fd(model, y, belief, eps=1e-5)` | → (2,) | Finite-difference gradient oracle. |
| `hierarchical_message_terms(model, y, belief)` | → `HierarchicalMessageTerms` | Forward error and backward prediction messages used by the Ch.8 diagrams. |

## `core.active_inference` — active generalized filtering (Chapter 7)

Continuous-state active inference: the agent gains an **action** that descends free energy
through the sensory channel. See the [Chapter 7 concept map](../chapters/chapter_07.md).

| Symbol | Signature | Purpose |
|---|---|---|
| `ActiveInferenceAgent(perception_model, forward_model=1.0, kappa_x=0.2, kappa_a=0.4)` | dataclass | Wraps a `DynamicStateSpaceModel` (whose `f` attractor encodes the preference `v`) + forward model + learning rates; `.preference`. |
| `perception_gradient(agent, y, mu)` | floats → float | `μ̇_x = −κ_x ∂F/∂μ_x` (Eq. 10a). |
| `action_gradient(agent, y, mu)` | floats → float | `ȧ = −κ_a·(∂y/∂a)·λ_y·ε_y` (Eq. 9/11/17). |
| `ai_free_energy(agent, y, mu)` | floats → float | The perception free energy (Eq. 7a). |
| `MultivariateActiveInferenceAgent(perception_model, forward_model, kappa_x=0.2, kappa_a=0.4)` | dataclass | §7.5 vector generalized-coordinate agent; `forward_model` is `∂ỹ/∂a`. |
| `multivariate_perception_flow(agent, y_tilde, mu_tilde)` | arrays → vector | `Dμ̃_x − κ_x ∂F/∂μ̃_x` for Algorithm 7.5.1. |
| `multivariate_action_gradient(agent, y_tilde, mu_tilde)` | arrays → action vector | `ȧ = −κ_a(∂ỹ/∂a)^TΠ̃_yε̃_y`. |
| `multivariate_ai_free_energy(agent, y_tilde, mu_tilde)` | arrays → float | Generalized VFE used by vector perception and action. |

## `core.pomdp` — discrete state-space active inference (Chapters 9–10)

The categorical (POMDP) formulation: a generative model built from `A`/`B`/`C`/`D` matrices,
exact hidden-state inference, expected-free-energy policy selection (§9.5), dynamic filtering
(§9.2–9.3), and **Dirichlet parameter learning** (§10.1). See the
[Chapter 9](../chapters/chapter_09.md) and [Chapter 10](../chapters/chapter_10.md) concept maps.

| Symbol | Signature | Purpose |
|---|---|---|
| `softmax(x, axis=-1)` | array → array | Numerically stable softmax (Boltzmann `σ`) along the selected axis. |
| `one_hot(index, n)` | ints → (n,) | One-hot observation vector `ô ∈ {0,1}^n` (Eq. 8); rejects out-of-range indices. |
| `is_stochastic_matrix(M, atol=1e-9)` | (·,·) → bool | True if `M` is finite, 2-D, non-negative, and every column is a categorical (sums to 1). |
| `POMDPModel(A, D, B=None, C=None, E=None)` | dataclass | Discrete generative model; validates finite categorical `A`/`D`, `B` shape/orientation, and non-negative preference/prior vectors; `n_states`, `n_obs`. |
| `infer_states(model, obs, prior=None)` | model, int/array → (C,) | Exact posterior `σ(log Aᵀô + log D)` (Eq. 12/13); `obs` is an index or categorical vector; `prior` overrides `D`. |
| `predict_state(model, s, u)` | model, (C,), int → (C,) | One-step state prediction `B[u]·s` under control `u`. |
| `EFEComponents` | dataclass | One predicted state's `risk`, `ambiguity`, optional `novelty`, expected observation, and total `G = risk + ambiguity - novelty`. |
| `PolicyEFETrace` | dataclass | Per-step policy rollout: predicted states/observations, risks, ambiguities, novelties, `totals_per_step`, and policy totals. |
| `efe_components(model, s, C, *, a=None)` | → `EFEComponents` | First-class decomposition of one-step EFE (Eq. 52; optional Ch.10 novelty Eq. 15). |
| `efe_risk(model, s, C)` | model, (C,), (O,) → float | Risk term `o·(log o − log C)`, `o = A·s` (reward-seeking, Eq. 60); `C` raw, ε-floored. |
| `efe_ambiguity(model, s)` | model, (C,) → float | Ambiguity `H·s`, `H = −diag(Aᵀlog A)` (information-seeking, Eq. 64). |
| `expected_free_energy(model, s, C)` | → float | `efe_risk + efe_ambiguity` (Eq. 52). |
| `evaluate_policy(model, s0, policy, C)` | → float | Total EFE of a policy, summing over its B-propagated horizon (Eq. 54). |
| `evaluate_policy_components(model, s0, policy, C, *, a=None)` | → `PolicyEFETrace` | Same rollout as `evaluate_policy`, preserving per-step risk/ambiguity/novelty for diagnostics and figures. |
| `policy_posterior(G, *, gamma=1.0)` | (n_π,) → (n_π,) | `σ(−γ G)` — lowest EFE ⇒ highest probability (Eq. 55). |
| `action_posterior(policy_post, policies, n_controls, *, tau=0)` | → (U,) | Marginalize policies onto the action at step `τ` (Eq. 69). |

**§9.2–9.3 — dynamic filtering + discrete variational free energy.**

| Symbol | Signature | Purpose |
|---|---|---|
| `forward_filter(model, observations, *, B=None)` | → (T, C) | HMM forward pass (Alg. 9.2.1): rolling belief `s⁽ᵗ⁾ = σ(log Aᵀôᵗ + log B s⁽ᵗ⁻¹⁾)`. |
| `predict_beliefs(model, s, horizon, *, B=None)` | → (H, C) | Observation-free prediction rollout `s⁽τ⁾ = σ(log B s⁽τ⁻¹⁾)` (Alg. 9.2.2). |
| `expected_observation(model, s)` | (C,) → (O,) | Expected observation distribution `o = A s`. |
| `discrete_vfe(model, s, obs, prior)` | → float | Discrete VFE (G-form, Eq. 29) `s·(log s − log prior − log Aᵀô)`; ≥ surprisal, equality at the posterior. |

**§10.1 — learning the POMDP parameters (Dirichlet).** See the
[Chapter 10 concept map](../chapters/chapter_10.md).

| Symbol | Signature | Purpose |
|---|---|---|
| `dirichlet_expected_value(counts, *, axis=0)` | array → array | Dirichlet mean `α/Σα` along `axis` (Eq. 5). |
| `expected_A(a)` / `expected_B(b)` / `expected_D(d)` | counts → probs | Column-normalized `A`/`B` (per slice) and normalized `D`. |
| `accumulate_a_counts(o, s)` | (O,),(C,) → (O,C) | Likelihood evidence `o ∘ s` (outer product, Eq. 4a). |
| `accumulate_b_counts(s_curr, s_prev)` | (C,),(C,) → (C,C) | Transition evidence `s⁽τ⁾ ∘ s⁽τ⁻¹⁾` (Eq. 4b). |
| `accumulate_d_counts(s0)` | (C,) → (C,) | State-prior evidence `s⁽⁰⁾` (Eq. 4c). |
| `novelty_matrix(a)` | (O,C) → (O,C) | Parameter-novelty matrix `W ≈ ½(1/a − 1/a₀)`, `a₀` = column-sum broadcast (Eq. 12). |
| `parameter_novelty(a, s)` | (O,C),(C,) → float | Expected information gain `o·(W s)`, `o = E[Dir(a)] s` (Eq. 13b/19). |
| `efe_with_novelty(model, s, C, a)` | → float | Novelty-augmented EFE `risk + ambiguity − novelty` (Eq. 15). |

**§10.2 — habit (baseline prior `E`) + policy precision `γ`.**

| Symbol | Signature | Purpose |
|---|---|---|
| `policy_posterior_full(G, *, F=None, E=None, gamma=1.0)` | → (P,) | Full policy posterior `σ(log E − F − γ G)` (Eq. 20–22); reproduces Example 10.5 exactly. |
| `precision_gradient(G, F, gamma, *, E=None, beta0=1.0)` | → float | VFE gradient `∂F/∂γ = (β−β₀) + (π−π₀)·(−G)` (Eq. 23). |
| `learn_precision(G, F, *, E=None, beta0=1.0, kappa=0.25, n_iter=64, tol=1e-10)` | → `PrecisionResult` | Learn `γ = 1/β` by descent (Eq. 24/25); self-consistent fixed point. |
| `PrecisionResult` | dataclass | `gamma`, `beta`, `gamma_trace`, `converged`, `grad_final`. |

**§10.3 — factorial depth (state factors + observation modalities).**

| Symbol | Signature | Purpose |
|---|---|---|
| `FactorialPOMDP(A, D, B=None, C=None)` | dataclass | Sets of arrays: `A=[A^(m)]` (each `(O_m, C_0,…,C_N)`), `B=[B^(n)]`, `C=[C^(m)]`, `D=[D^(n)]`; `n_factors`, `n_modalities`, `factor_sizes`, `obs_sizes`. |
| `factorial_expected_observation(A_m, states)` | → (O_m,) | Contract `A^(m)` with all factor beliefs (`o^(m)`); reduces to `A·s`. |
| `factorial_likelihood_message(A_m, obs_m, states, factor)` | → (C_factor,) | VMP message `E_{s∖n}[log A^(m)]·o` into one factor (Eq. 36). |
| `factorial_infer_states(model, obs, *, priors=None, n_iter=64, tol=1e-10)` | → list | Mean-field state inference (Eq. 35–37); reduces to Ch.9 for `N=M=0`. |
| `factorial_predict_states(model, states, actions)` | → list | Per-factor `B^(n)` propagation. |
| `factorial_modality_ambiguity(A_m, states)` | → float | Expected observation entropy for one modality (Eq. 38a ambiguity). |
| `factorial_efe(model, states, *, C=None)` | → float | Factorial EFE `Σ_m risk_m + ambiguity_m` (Eq. 38a); reduces to `expected_free_energy`. |

**§10.4 — hierarchical depth (nested POMDP layers).**

| Symbol | Signature | Purpose |
|---|---|---|
| `HierarchicalPOMDP(layers, link=None)` | dataclass | A stack of `POMDPModel` layers + column-stochastic `link[l]` maps; `n_layers`, `layer_prior(l, higher_belief)` (Eq. 42). |
| `hierarchical_layer_vfe(s, obs, A, *, prior)` | → float | Per-layer VFE (Eq. 50); reduces to `discrete_vfe`. |
| `hierarchical_layer_efe(A, s, C)` | → float | Per-layer EFE (Eq. 43); reduces to `expected_free_energy`. |
| `hierarchical_policy_posterior(G, *, F=None, E=None)` | → (P,) | Layer-wise policy posterior `σ(log E − Σ_τ F − Σ_τ G)` (Eq. 49). |

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
