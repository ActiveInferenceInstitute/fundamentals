# `active_inference.estimators` — module reference

Point-estimate and parameter-learning algorithms. Together with
`core.inference` (exact grid-based posterior) and `core.lgs` (closed-form
multivariate posterior) these cover every learning / inference path used in
Chapters 2 and 3.

## `estimators.mle`

Univariate maximum-likelihood estimation of the hidden state given a linear
generating function.

| Symbol | Role |
|---|---|
| `mle_analytic_linear(y_obs, beta0, beta1)` | Closed-form ``x_MLE = (mean(y) − β₀) / β₁``. |
| `mle_loss(x, y_obs, beta0, beta1, sigma2_y, psi=None)` | Sum of negative log-likelihoods at scalar / array `x`. |
| `mle_grad_x(x, y_obs, beta0, beta1, sigma2_y)` | Analytic gradient of the linear-Gaussian NLL. |

## `estimators.map`

The MAP estimator adds a Gaussian prior to the MLE objective.

| Symbol | Role |
|---|---|
| `map_analytic_linear(y_obs, beta0, beta1, sigma2_y, m_x, s2_x)` | Closed-form precision-weighted average of MLE and prior mean. |
| `map_loss(x, y_obs, beta0, beta1, sigma2_y, m_x, s2_x, psi=None)` | Negative log-posterior up to a constant. |
| `map_grad_x(x, y_obs, beta0, beta1, sigma2_y, m_x, s2_x)` | Gradient of the MAP loss. |

## `estimators.gradient_descent`

Generic 1-D minimizer with optional analytic gradient (falls back to
centered finite differences when none is provided).

```python
from active_inference import gradient_descent

result = gradient_descent(
    loss_fn=lambda x: (x - 3.0) ** 2,
    x0=0.0,
    learning_rate=0.1,
    max_iter=200,
)
print(result.x, result.converged, result.n_iterations)
```

`GradientDescentResult` carries the final iterate, full iterate / loss
history, iteration count, and a `converged` flag based on the iterate-step
threshold.

## `estimators.linear_regression`

Vectorized linear regression for the design-matrix convention
`X ∈ ℝ^{N×C}` (a column of ones is prepended automatically when
`intercept=True`).

| Symbol | Role |
|---|---|
| `add_intercept(X)` | Prepend a column of ones to the design matrix. |
| `mle_linear_regression(X, y, intercept=True)` | Closed-form normal equation via `np.linalg.lstsq` (Moore–Penrose). |
| `GDRegressionResult` | dataclass carrying `theta`, `history`, `losses`, `n_iterations`, `converged`. Returned by `gd_linear_regression`. |
| `gd_linear_regression(X, y, learning_rate, ..., l2=0.0)` | Vectorized gradient descent with optional L2 regularization. |
| `squared_loss(theta, X, y, ..., l2=0.0)` | Loss value at a parameter vector. |
| `squared_loss_grad(theta, X, y, ..., l2=0.0)` | Analytic gradient of the squared loss. |

### Bayesian linear regression

```python
from active_inference import BayesianLinearRegression
import numpy as np

blr = BayesianLinearRegression(
    prior_mean=np.zeros(C + 1),
    prior_cov=np.eye(C + 1) * 4.0,
    sigma2_y=0.25,
)
posterior = blr.fit(X, y)
print(posterior.mean, posterior.std())

mean_pred, var_pred = posterior.predictive(X_new, sigma2_y=blr.sigma2_y)
```

`BayesianLinearRegression.fit_sequential(X, y)` yields `(i, BLRPosterior)`
after each row is assimilated — this is what powers
`animation_blr_tightening.py`.

## `estimators.em`

Expectation–Maximization for linear factor analysis. Standard-normal prior
on the latent states, diagonal observation noise, zero-centered data.

| Symbol | Role |
|---|---|
| `factor_analysis_e_step(Y, Theta, cov_y)` | Posterior mean (per row) + (shared) covariance. |
| `factor_analysis_m_step(Y, mu, cov)` | Updated `Θ` and diagonal noise covariance. |
| `incomplete_log_likelihood(Y, Theta, cov_y)` | Marginal `log p(Y)` under the current parameters. |
| `fit_factor_analysis(Y, n_factors, ...)` | Full EM loop with convergence diagnostics. |

`FactorAnalysisResult` holds the final loadings, posteriors, log-likelihood
trace, iteration count, convergence flag, and per-iteration history of `Θ`
and the noise diagonal — used by `animation_em_convergence.py`.

## `estimators.variational` — variational inference (Chapter 4)

Three ways to minimize variational free energy (`core.variational`), in
increasing sophistication. Each returns a trace dataclass so orchestrators can
plot the descent and tests can assert monotonicity and convergence to the exact
grid posterior.

```python
from active_inference import LinearGaussianModel, fixed_form_vi
import numpy as np

model = LinearGaussianModel(beta0=3, beta1=2, sigma2_y=0.25, m_x=4, s2_x=0.25)
grid  = np.linspace(-6, 12, 2001)
res   = fixed_form_vi(model, 7.0, grid, lr=5e-3, n_iter=3000)
print(res.belief)            # GaussianBelief(mu≈2.4, var≈0.05) — the exact posterior
print(res.final_free_energy) # ≈ −log p(y): the bound is tight at the posterior
```

| Symbol | Role |
|---|---|
| `coordinate_search_vfe(model, y, x_grid, *, mu0=None, var0=None, kappa=0.01, n_iter=20, tol=0.0, min_var=1e-4)` | Algorithm 4.2.1 — zero-order coordinate search over the eight `(μ±κ, σ²±κ)` neighbours. → `CoordinateSearchResult`. |
| `fixed_form_vi(model, y, x_grid, *, mu0=None, var0=None, lr=5e-3, n_iter=2000, tol=1e-9, min_var=1e-4, fd_eps=1e-4)` | Algorithm 4.6.1 — gradient descent on `(μ, σ²)` via central finite differences (no torch). `μ` clamped to the grid span. → `FixedFormResult`. |
| `free_form_cavi(y, cfg=None, *, n_sweeps=50, tol=1e-9)` | Algorithm 4.5.1 — mean-field CAVI on `(x, β₀, β₁)`; each update closed-form Gaussian (Eq. 43). → `CAVIResult`. |
| `MeanFieldConfig(...)` | Generative model + priors for CAVI (Eq. 32–34); defaults are the book's `φ`. Validates positive variances. |
| `CoordinateSearchResult` / `FixedFormResult` | `mus`, `vars_`, `free_energies`, `belief`, `converged`, `n_iter_run`, `final_free_energy`. `FixedFormResult` also carries the per-step `components`. |
| `CAVIResult` | `q_x`, `q_b0`, `q_b1`, `free_energies`, per-sweep `mu_x`/`mu_b0`/`mu_b1`, `converged`, `n_sweeps_run`, `final_free_energy`. |

All three VFE traces are (weakly) monotone non-increasing and bounded below by
the surprisal `−log p(y)`. With the book's `κ=0.01, 20`-iteration settings the
coordinate search deliberately stops short of the minimum (§4.4); the
orchestrator's `--extended` flag (and larger `kappa`/`n_iter`) reaches it.

## `estimators.predictive_coding` (Chapter 5)

Predictive coding is gradient descent on the MAP free energy (Chapter 4's VFE with
a point belief), driven by precision-weighted prediction errors. See the
[Chapter 5 concept map](../chapters/chapter_05.md).

```python
from active_inference import (
    LinearFunction, PredictiveCodingModel, predictive_coding_inference,
)

model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25, m_x=4.0, s2_x=0.25)
res = predictive_coding_inference(model, 7.0, kappa=0.02, n_iter=2000)
print(res.mu_star)   # ≈ 2.4 — the Ch.4 grid posterior mean (cross-chapter oracle)
```

| Symbol | Role |
|---|---|
| `predictive_coding_inference(model, y, *, mu0=None, kappa=0.1, n_iter=500, tol=1e-9)` | Algorithm 5.2.1 — univariate perception `μ ← μ − κ(λ_x ε_x − λ_y ε_y g'(μ))`. → `PredictiveCodingResult`. Fixed-step descent is stable for `κ < 2/L`, `L = λ_x + β₁²λ_y`. |
| `multivariate_predictive_coding(g, jacobian, y, m_x, *, precision_y, precision_x, mu0=None, kappa=0.1, n_iter=500, tol=1e-9)` | §5.3 — vector states, `μ ← μ − κ(Π_x ε_x − Jᵀ Π_y ε_y)`. → `MultivariatePCResult`. |
| `hierarchical_predictive_coding(model, y, *, mu0=None, kappa=0.05, n_iter=500, tol=1e-12)` | §5.4 — `L+1` layers, `μ^{[0]}=y` clamped, simultaneous (Jacobi) updates. → `HierarchicalPCResult`. |
| `HierarchicalPCModel(generators, variances, m_x)` | `L` generators, `L+1` variances; `m_x=0` ⇒ unconstrained top. Methods `prediction`, `errors`, `free_energy`, `layer_free_energies`. |
| `PredictiveCodingResult` | `mus`, `free_energies`, `eps_x`, `eps_y`, `mu_y`, `mu_star`, `converged`, `n_iter_run`; `final_free_energy`, `summary()`. |
| `MultivariatePCResult` / `HierarchicalPCResult` | vector / per-layer analogues (`mus`, `errors`, `free_energies`, `layer_free_energies`, `mu_star`, …). |

Every PC free-energy trace is monotone non-increasing (verified by
`convergence_report`). The linear fixed point equals the exact posterior mean
across a 144-config sweep (`TestLinearOracleSweep`); the nonlinear fixed point
equals the grid argmin of `F`.

## `estimators.generalized_filtering` (Chapter 6)

Online generalized filtering (§6.1, Algorithm 6.1.1): simulate a dynamic stochastic
environment, then filter its observation stream one Euler step at a time. See the
[Chapter 6 concept map](../chapters/chapter_06.md).

```python
from active_inference import (
    DynamicProcess, DynamicStateSpaceModel, LinearFunction,
    simulate_dynamic_process, generalized_filter,
)
import numpy as np

proc = DynamicProcess(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0))
tr = simulate_dynamic_process(proc, x0=5.0, n_steps=1000, dt=0.01, rng=np.random.default_rng(0))
model = DynamicStateSpaceModel(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                               s2_x=5.0, sigma2_y=0.02)
res = generalized_filter(model, tr.ys, dt=0.01, kappa=0.1, mu0=15.0)
print(res.tracking_error(tr.xs, burn_in=300))   # ≈ 0.08 — close tracking
```

| Symbol | Role |
|---|---|
| `DynamicProcess(f, g, omega_x=0.1, omega_y=0.1)` | Stochastic generative process `E` (Eq. 9): `ẋ*=f_E(x*)+ω_x`, `y=g_E(x*)+ω_y`. |
| `simulate_dynamic_process(process, x0, *, n_steps, dt=0.01, rng=None)` | Euler-integrate the environment → `DynamicProcessTrace(xs, ys, dt)`. |
| `generalized_filter(model, ys, *, dt=0.01, kappa=0.1, mu0=0.0)` | Online recognition (Alg. 6.1.1, agent step) → `GeneralizedFilterResult`. |
| `GeneralizedFilterResult` | `mus`, `mu_ys`, `free_energies`, `eps_x`, `eps_y`, `ys`; `final_mu`, `tracking_error(truth, burn_in=0)`. |

The belief tracks the true state to `≈ 0.08` mean error (vs `≈ 5` for the unfiltered
prior); at a held observation the relaxation converges to `gf_fixed_point_linear`.

### Multivariate filter (§6.2)

| Symbol | Role |
|---|---|
| `MultivariateDynamicProcess(f, g, omega_x=0.1, omega_y=0.1)` | Vector generative process. |
| `simulate_multivariate_process(process, x0, *, n_steps, dt=0.01, rng=None)` | Euler-integrate → `MultivariateProcessTrace(xs (T+1,C), ys (T+1,D), dt)`. |
| `multivariate_generalized_filter(model, ys, *, dt=0.01, kappa=1.0, mu0=None)` | Online vector filter (Eq. 14/15) → `MultivariateFilterResult`. |
| `MultivariateFilterResult` | `mus`, `mu_ys`, `free_energies`, `ys`; `tracking_error(truth, burn_in=0)` (mean Euclidean). |

Verified by reducing exactly to the scalar §6.1 filter on a 1-D problem, and by tracking
the Hooke's-law oscillator to `≈ 1.1` mean error vs `≈ 11` for the static prior (the
residual is the perception lag at the turning points).

### Generalized coordinates (§6.3–6.5)

| Symbol | Role |
|---|---|
| `generalized_filter_gc(model, ys_tilde, *, dt=0.01, kappa=1.0, mu0_tilde=None)` | Generalized filtering in generalized coordinates (Alg. 6.5.1); recognition flow `μ̃̇ = Dμ̃ − κ∂F/∂μ̃` → `GeneralizedFilterGCResult`. |
| `GeneralizedFilterGCResult` | `mus (T,M)`, `free_energies`, `eps_x`, `eps_y`, `ys`; convenience `positions` (`μ_x^[0]`) and `velocities` (`μ_x^[1]`). |

The order-0 belief recovers the position and the order-1 belief recovers the velocity;
verified by exact at-rest recovery (`[10,0,0]`) and ≈2 % velocity recovery on a
constant-velocity trajectory with a free-motion model.

### Correlated/vector generalized coordinates (§6.6)

| Symbol | Role |
|---|---|
| `generalized_measurements_from_series(ys, *, embedding_dim, dt)` | Estimate `ỹ[t,m,d]` from a sampled observation series using finite differences. |
| `generalized_vector_filter(model, ys_tilde, *, dt=0.01, kappa=1.0, mu0_tilde=None)` | Vector generalized-coordinate filter with full correlated precisions → `GeneralizedVectorFilterResult`. |
| `GeneralizedVectorFilterResult` | `mus (T,M,C)`, `free_energies`, `eps_x`, `eps_y`, `ys`; `positions`, `tracking_error(truth)`. |

## `estimators.continuous_learning` (Chapter 8)

Learning and attention extend generalized filtering by adding slow flows for first-
and second-order parameters. The estimator keeps the hidden-state flow fast while
using damped velocity updates for `mu_theta` and `mu_zeta`.

| Symbol | Role |
|---|---|
| `simulate_learning_attention(model, ys, *, initial, dt=0.01, kappa_x=0.5, kappa_theta=0.5, kappa_zeta=0.1, damping=1.0, learning_gain=4.0, attention_gain=12.0)` | Chapter 8 perception, learning, and attention loop over an observation stream → `LearningAttentionResult`. |
| `LearningAttentionResult` | `mus`, `mu_thetas`, `mu_zetas`, velocities, predicted observations, free energies, errors, learned variances, `final_state`, `final_variance_x`, `tracking_error()`. |

## `estimators.active_inference` (Chapter 7)

The coupled action-perception loop (Algorithm 7.2.1): the agent's action feeds back into
the generative process, so environment and agent are simulated together. See the
[Chapter 7 concept map](../chapters/chapter_07.md).

| Symbol | Role |
|---|---|
| `ActiveEnvironment(drift, g, omega_x=0.05, omega_y=0.05)` | The generative process; state update `x ← x + Δt·(drift(x) + a) + ω_x`, obs `g_E(x) + ω_y`. |
| `simulate_active_inference(agent, env, *, x0, mu0, n_steps=6000, dt=0.01, action_start=0, rng=None)` | Run the coupled loop → `ActiveInferenceResult`. |
| `ActiveInferenceResult` | `xs`, `mus`, `actions`, `ys`, `free_energies`, `eps_y`, `action_start`; `settled_state()`, `settled_action()`. |
| `MultivariateActiveEnvironment(drift, g, action_matrix, omega_x=0, omega_y=0)` | Vector generative process for §7.5; action enters through an explicit matrix. |
| `simulate_multivariate_active_inference(agent, env, *, x0, mu0_tilde=None, n_steps=2000, dt=0.01, action_start=0, rng=None)` | Run Algorithm 7.5.1 with vector states, online generalized measurements, and vector action → `MultivariateActiveInferenceResult`. |
| `MultivariateActiveInferenceResult` | `xs`, `mus (T,M,C)`, `actions`, `ys`, `y_tildes`, `free_energies`, `eps_y`; `settled_state()`, `settled_action()`, `preference_error(preference)`. |

Verified by the defining AIF property: with action the true state is driven to the
preferred set-point `v`; with `κ_a=0` it stays at the uncontrolled exogenous attractor.

## `estimators.pomdp` — Grid World planning (Chapter 9, §9.5)

A discrete active-inference agent that plans by minimizing **expected free energy** over
policies (Algorithm 9.5.1). See the [Chapter 9 concept map](../chapters/chapter_09.md).

| Symbol | Role |
|---|---|
| `make_gridworld(rows, cols, goal)` | Build a deterministic, fully-observable Grid World `POMDPModel`: identity `A`, four wall-clamped move `B[u]` (up/down/left/right), one-hot goal `C`, start-cell `D`. |
| `enumerate_policies(n_controls, horizon)` | All length-`horizon` action sequences (`itertools.product`). |
| `simulate_pomdp_agent(model, *, start=0, horizon=2, gamma=4.0, max_steps=12, rng=None)` | Receding-horizon loop: infer state, score policies by EFE, `σ(−γG)`, marginalize to an action, act → `GridWorldResult`. |
| `GridWorldResult` | `states`, `actions`, `beliefs`, `goal`, `reached`, `n_steps_to_goal`. |

Verified behaviorally: on a 3×3 grid with the goal in the opposite corner the agent reaches
it in the minimum **4 steps** (`states = [0,3,4,7,8]`), and a goal-reaching policy scores
strictly below a non-reaching one.

## `estimators.pomdp_extensions` — Chapter 11 planning-extension demos

Small simulations built from `core.pomdp_extensions` and the established Chapter 9/10
POMDP primitives.

| Symbol | Role |
|---|---|
| `make_line_world(n_states=5, goal=None)` | Deterministic one-dimensional POMDP with left/right controls and a goal preference. |
| `simulate_sophisticated_planning(n_states=5, horizon=3, gamma=4.0)` | Bounded tree search in the line world, returning policy scores plus future-belief diagnostics. |
| `StatePreferenceResult` | Normalized preference schedule, selected target states, and terminal preference posterior. |
| `simulate_state_preference_schedule(log_preferences)` | Normalize time-varying log preferences and expose their target-state trace. |
| `ParameterForgettingResult` | Before/after pseudocounts plus inverse-count uncertainty diagnostics. |
| `simulate_parameter_forgetting(counts, *, rate, floor=1.0)` | Apply Dirichlet forgetting and report uncertainty before and after. |
| `StructureLearningResult` | Candidate structure evidence, posterior probabilities, and selected model index. |
| `simulate_structure_learning(evidence, *, complexity=0.0)` | Penalize model evidence by complexity and return a structure posterior. |
| `PreferenceHabitLearningResult` | Preference pseudocounts, learned preference posterior, habit prior, and selected preference/action traces. |
| `simulate_preference_habit_learning(observations, actions=None, *, learning_rate=1.0, concentration=1.0)` | Couple learned preferences and habits for Chapter 11 demos. |
| `PathPolicyResult` | Candidate paths, path scores, and posterior probabilities. |
| `simulate_path_policy_computation(paths, preferences, *, transition_cost=0.0, gamma=4.0)` | Score path-based policies and normalize their posterior. |

## `estimators.applications` — Chapter 13 application demos

Application-level helpers that keep robotics/social demonstrations deterministic and
lightweight while reusing the package's active-inference conventions.

| Symbol | Role |
|---|---|
| `NavigationResult` | Navigation path, goal, distance-to-goal trace, and preference trace. |
| `simulate_robot_navigation(n_steps=80, goal=(1.0, 0.75))` | Smooth deterministic 2-D navigation trajectory with increasing goal preference. |
| `FaultTolerantControlResult` | Desired control, actuator efficacy, compensated action, output, and error traces. |
| `simulate_fault_tolerant_control(n_steps=90, fault_start=45)` | Deterministic fault-compensation controller for robotics control demos. |
| `SocialInferenceResult` | Communicative observations, belief trace, and inferred final intention. |
| `simulate_social_inference(observations=None, *, accuracy=0.82)` | Binary hidden-intention inference from social observations. |
| `RoboticsTheoryResult` | Controllability, epistemic value, preference satisfaction, and combined score landscape. |
| `robotics_theory_landscape(n_points=80)` | Theory-level tradeoff curve for Chapter 13 robotics discussion. |

### §10.1 — learning the POMDP parameters (Dirichlet)

Trial-based learning of the `A`/`B`/`D` arrays by Dirichlet counting (Algorithm 10.1.1).
See the [Chapter 10 concept map](../chapters/chapter_10.md).

| Symbol | Role |
|---|---|
| `DirichletParameters(a, d, b=None)` | Container for the concentration parameters; `.A`/`.B`/`.D` return the Dirichlet means, `.copy()` clones. |
| `LearningResult` | Per-trial trace: `A_history`, `B_history`, `D_history`, `a_confidence`, `b_confidence`, `d_final`, ground-truth fields, `final_A_error()`. |
| `learn_D_vector(d0, initial_state_beliefs)` | Example 10.1: accumulate `s⁽⁰⁾` into `d`; reproduces `d=[45.1,5.9]`, `D≈[0.884,0.116]`. |
| `simulate_array_learning(*, A_true, B_true, n_trials=5, steps_per_trial=1000, learn="A", ...)` | Examples 10.2/10.3: learn `A` or `B` by counting pairs under a true generative process → `LearningResult`. |
| `LearningAgentResult` | Trace of the learning agent: `A_history`, `a_confidence`, `novelty_first_trial`, `final_A_error()`. |
| `simulate_learning_agent(*, A_true, n_states, n_trials=30, steps_per_trial=20, ...)` | Algorithm 10.1.1: a novelty-driven agent that learns `A` while acting. |

Verified against the book's worked examples: `learn_D_vector` reproduces Example 10.1
exactly; `simulate_array_learning` converges to the true `A`/`B` (max abs error < 0.05) with
monotonically growing confidence; the learning agent reaches < 0.1 likelihood error.

### §10.2 — precision / habit policy optimization

| Symbol | Role |
|---|---|
| `precision_policy_sweep(G, gammas, *, E=None, F=None)` | `(K, P)` policy posteriors across a range of precisions `γ` (Example 10.5 / Fig 10.2.2); each row is `σ(log E − F − γ G)`. |

Reproduces Example 10.5 exactly (γ=0 ⇒ uniform/habit; γ↑ ⇒ mass concentrates on the
lowest-EFE policy). See also `core.pomdp.learn_precision` for learning `γ` itself.

### §10.3 — factorial depth: the two-armed bandit (Example 10.7)

| Symbol | Role |
|---|---|
| `make_two_armed_bandit(*, p_win=0.9, hint_reliability=1.0, reward_prefs=(0,-3,4))` | Build the two-armed bandit `FactorialPOMDP` (context + choice factors; hint/reward/choice modalities). `reward_prefs` is softmax-normalized into `C[1]`. |
| `simulate_two_armed_bandit(model, *, true_context=1, n_steps=12, gamma=4.0, rng=None)` | Receding-horizon factorial agent → `TwoArmedBanditResult`. |
| `TwoArmedBanditResult` | `context_belief`, `choices`, `reward_obs`, `policy_posterior`, `true_context`, `n_wins`, `n_hints`. |

Verified behaviorally: the agent learns the hidden context (belief → the true better machine
for both contexts) and exploits it; the hint carries `ln 2` nats of context information gain.

### §10.4 — hierarchical depth (nested time scales)

| Symbol | Role |
|---|---|
| `simulate_hierarchical_agent(model, *, true_top=0, bottom_observations=None, n_macro=4, inner_steps=3, rng=None)` | Two-layer nested rollout; the slow top layer's belief seeds the fast bottom layer's prior each macro-step → `HierarchicalResult`. |
| `HierarchicalResult` | `top_belief`, `bottom_priors`, `bottom_belief`, `layer_vfe`, `true_top`. |

Verified: the top regime is inferred and demonstrably steers the bottom layer's prior (no
book numerical oracle for §10.4, so verification is by reduction + this top-down-control demo).
