# `active_inference.visualizations` — module reference

Reusable figure helpers for the chapter orchestrators. Three modules: static
plots, interactive sliders, and matplotlib animations. Every function accepts
an optional `ax` (for composition) and an optional `save_path` (for headless
rendering).

## `visualizations.plotting`

Static matplotlib helpers. Each returns the `Figure` so callers can keep
composing or save themselves.

| Symbol | Role |
|---|---|
| `save_or_show(fig, save_path=None, *, show=False, dpi=150)` | Either save to disk or call `plt.show()`; returns the resolved path. |
| `plot_prior_likelihood_posterior(result, *, truth=None, save_path=None, ...)` | Three-panel prior / likelihood / posterior figure. |
| `plot_generating_function(x, f_x, samples_x=None, samples_y=None, ...)` | `y = g(x)` curve with optional sample scatter. |
| `plot_likelihood_ridge(x_grid, likelihoods, labels=None, ...)` | Vertically stacked ridge plot of per-sample likelihoods. |
| `plot_joint_heatmap(x_grid, y_grid, joint, ...)` | 2-D heatmap of `p(x, y)`. |
| `plot_gradient_descent(history, losses, *, truth=None, ...)` | Side-by-side loss-vs-iter and iterate-vs-iter view. |
| `plot_precision_comparison(results, ...)` | Overlay multiple posteriors for a precision sweep. |
| `confidence_ellipse(mean, cov, *, n_std=2.0, **kwargs)` | Eigen-aligned `Ellipse` patch for a 2-D Gaussian. |
| `plot_2d_gaussian(mean, cov, *, samples=None, truth=None, ...)` | 1-σ / 2-σ ellipses + optional sample scatter. |

## `visualizations.interactive`

Slider-driven exploration with no `ipywidgets` dependency — purely
`matplotlib.widgets.Slider`.

| Symbol | Role |
|---|---|
| `interactive_inference(...)` | Live sliders for observation, prior mean, prior variance, likelihood variance. |
| `interactive_precision(...)` | Single-slider sweep of the prior-vs-data precision ratio. |
| `interactive_topic_demo(slug)` | Registry-driven extras slider view backed by `active_inference.extra_topics.build_topic_demo`. |

```python
from active_inference.visualizations import interactive_inference

interactive_inference()
import matplotlib.pyplot as plt; plt.show()
```

## `visualizations.animations`

Matplotlib `FuncAnimation` builders. The bundled pillow writer means GIF
output works with no FFmpeg / ImageMagick install.

| Symbol | Role |
|---|---|
| `animate_sequential_posterior(x_grid, posteriors, *, truth=None, prior=None, ...)` | One frame per assimilated observation. |
| `animate_gradient_descent(loss_grid, x_grid, history, losses, ...)` | Iterate rolling down the loss curve + sync'd loss trace. |
| `animate_2d_posterior(means, covs, *, truth=None, prior_mean=None, prior_cov=None, ...)` | 1-/2-σ ellipses tightening over frames. |
| `animate_em_convergence(log_likelihoods, Theta_history, ...)` | LL curve + heat-map of loadings per iteration. |
| `animate_em_steps(...)` | Detailed factor-analysis E-step / M-step progression. |
| `animate_sufficient_statistics(...)` | Running Gaussian sufficient statistics over an observation stream. |
| `animate_calibration_growth(...)` | Calibration curve filling in as trials accumulate. |
| `animate_precision_sweep(...)` | Prior/likelihood precision sweep animation. |
| `animate_bimodal_emergence(...)` | Bi-modal posterior emergence under a non-injective generator. |
| `animate_lgs_online(...)` | Online Linear Gaussian System posterior animation. |
| `animate_blr_predictive_band(...)` | Bayesian linear-regression predictive-band animation. |
| `animate_vfe_descent(x_grid, beliefs, free_energies, *, posterior=None, surprisal=None, ...)` | Ch.4 — `q(x)` tightening onto the posterior while VFE falls to the surprisal bound. |
| `animate_recognition_dynamics(result, *, truth=None, oracle=None, surprisal=None, label=…, interval_ms=90, stride=1)` | **Composable** Ch.4/Ch.5 descent — duck-typed on `.mus`/`.free_energies`/`.eps_*`; 2 panels for a `FixedFormResult`, 3 for a `PredictiveCodingResult`; moving markers + live stat box. |
| `animate_hierarchical_pc(result, *, truth=None, interval_ms=60, stride=1)` | Ch.5 §5.4 — layer beliefs, errors → 0, and `Σ F` collapsing (Fig. 5.4.4 in motion). |
| `animate_learning_attention(mus, thetas, zetas, free_energies, *, truth_x=None, truth_theta=None, ...)` | Ch.8 §8.1 — hidden state, first-order parameter, log precision, learned variance, and VFE convergence. |
| `animate_multivariate_active_inference(result, *, preference, exogenous=None, dt=1.0, frame_stride=25, ...)` | Ch.7 §7.5 — 2-D state/belief path plus action, sensory error, and VFE traces. |
| `animate_discrete_beliefs(beliefs, *, observations=None, state_labels=None, interval_ms=350)` | Ch.9 §9.2 — categorical forward filtering: current posterior bars plus accumulated state-belief traces. |
| `animate_policy_efe_tradeoff(risks, ambiguities, *, posteriors=None, novelties=None, ...)` | Ch.9 §9.6 — animated risk/ambiguity decomposition showing policy selection shift as preferences or uncertainty change. |
| `animate_parameter_learning(history, confidence, *, truth=None, symbol="A", interval_ms=220, stride=1)` | **Composable** Ch.10 §10.1 — Dirichlet learning of `A` or `B` over trials: entries → truth (dots) and pseudocounts growing (Figs 10.1.3/10.1.4 in motion). |
| `animate_policy_precision(G, gammas, *, E=None, F=None, interval_ms=90)` | Ch.10 §10.2 — policy-posterior bars redrawing as precision `γ` ramps (Fig 10.2.2 in motion). |
| `animate_two_armed_bandit(result, *, interval_ms=350)` | Ch.10 §10.3 — the two-armed bandit context belief + policy posterior evolving step-by-step (Figs 10.3.6/7 in motion). |
| `save_animation(anim, path, *, fps=12, dpi=110)` | Save to GIF via the bundled pillow writer. |

The composable animators mirror the static `unified` plotters: `result` in → animation
out, same palette and bold typography, a legend on every panel, and a `stride` knob to
keep long-descent GIFs small. They share `_frame_indices` (always renders the final frame).

## `visualizations.variational`

Chapter 4 VFE figures and scalar surfaces.

| Symbol | Role |
|---|---|
| `vfe_surface(model, y, x_grid, mus, vars_)` | Evaluate the fixed-form VFE landscape over `(mu, var)`. |
| `plot_vfe_contour(model, y, x_grid, mus, vars_, ...)` | Contour plot of the VFE surface with optional descent path. |
| `plot_density_evolution(x_grid, densities, labels=...)` | Overlay variational-density snapshots during optimization. |
| `plot_vfe_decomposition(components, ...)` | Visualize VFE decomposition terms across a trace. |
| `plot_surprisal_relationship(...)` | Plot evidence/surprisal relationships for Chapter 4 intuition. |

## `visualizations.diagnostics`

Statistical diagnostic figures used by Chapter 3 and the statistics pages.

| Symbol | Role |
|---|---|
| `plot_calibration(curve, ...)` | Reliability diagram for empirical vs nominal coverage. |
| `plot_cdf_comparison(samples_a, samples_b, ...)` | Empirical CDF comparison figure. |
| `plot_coverage_curve(levels, coverage, ...)` | Coverage over a sweep of credible masses. |
| `plot_kl_trace(values, ...)` | Running KL / information-gain trace. |
| `plot_posterior_predictive_check(check, ...)` | Posterior-predictive replicate histogram with observed statistic. |
| `plot_qq(samples, reference=...)` | Quantile-quantile diagnostic plot. |
| `plot_running_statistics(stats, ...)` | Mean/std/KL/log-evidence summaries from `RunningPosteriorStats`. |
| `plot_score_trace(scores, ...)` | Forecast-score trace over trials. |

## `visualizations.style`

The shared visual vocabulary every figure routes through (bold, slide-sized
typography by default; centralized palette and stat-box style).

| Symbol | Role |
|---|---|
| `COLORS` | Named palette (`prior`, `likelihood`, `posterior`, `data`, `truth`, `neutral`). |
| `DEFAULT_RC` | Global rcParams — 16 pt bold titles, 15 pt labels, 12 pt legends, grid α=0.3, no top/right spines. Applied at import. |
| `set_default_style(overrides=None)` / `figure_style(overrides=None)` | Apply (idempotent) or temporarily apply the defaults. |
| `annotate_stat_box(ax, text, *, loc="upper left", fontsize=12, monospace=True)` | Place a corner statistics readout (monospace so number columns align). |
| `annotate_point(ax, x, y, text, *, color, dx, dy, marker="o", arrow=True)` | Mark and label an **analytical landmark** (fixed point, oracle, closed-form minimum) with an arrow callout. |
| `stat_box_bbox()` | Shared rounded-box style for compact numeric annotations. |

## `visualizations.unified` (Chapters 4–10)

One streamlined layer so every inference *result* renders through the same
language. See the [Chapter 5 concept map](../chapters/chapter_05.md).

| Symbol | Role |
|---|---|
| `panel_grid(n, *, title=None, figsize=None, rows=1)` | Row/grid of `n` panels with the shared default size; returns `(fig, flat axes)`. |
| `finalize(ax, *, xlabel, ylabel, title, legend=True, legend_loc="best")` | Apply the shared aesthetic (grid, labels, legend) to one axis. |
| `layer_colors(n)` | `n` perceptually-ordered viridis colours for hierarchical layers. |
| `plot_recognition_dynamics(result, *, truth=None, oracle=None, surprisal=None, ...)` | Unified descent figure — Ch.4 `FixedFormResult` (2 panels) **or** Ch.5 `PredictiveCodingResult` (3 panels); annotates `μ*`, oracle error, `F₀→F*`, convergence rate. |
| `plot_prediction_errors(model, y, mu_grid, *, truth=None)` | Fig. 5.1.2 — `F(μ)` with grid argmin + **closed-form** minimizer & curvature, and the two weighted errors trading off. |
| `plot_hierarchical_pc(result, *, truth=None)` | Fig. 5.4.4 — per-layer `μ`/`ε`/`F` with final-value annotations and `Σ F` stats. |
| `plot_generalized_filter(result, *, truth=None, dt=1.0, ...)` | Fig. 6.1.3 — generalized filtering: belief tracking the hidden state, prediction errors, free energy. |
| `plot_correlated_embedding_precision(precisions, gammas, ...)` | Fig. 6.6.2 — heatmaps of correlated embedding-order precision as `γ` changes. |
| `plot_generalized_vector_filter(result, truth, *, ordinary=None, dt=1.0, ...)` | Example 6.7 — vector generalized-coordinate tracking, ordinary-filter baseline, velocity beliefs, and VFE. |
| `plot_multivariate_active_inference(result, *, preference, exogenous=None, baseline=None, dt=1.0, ...)` | §7.5 — 2-D active path, state/belief components, action, sensory error, and VFE. |
| `plot_learning_attention(result, *, truth_x=None, truth_theta=None, ...)` | Fig. 8.1 — state tracking, first-order parameter learning, second-order precision/variance learning, prediction errors, and VFE. |
| `plot_hierarchical_message_passing(model, *, y, belief, ...)` | Fig. 8.5 — accessible two-layer diagram of forward error messages and backward predictions. |
| `plot_discrete_belief_sequence(beliefs, *, observations=None, state_labels=None, free_energies=None, ...)` | Fig. 9.2/9.3 — dynamic POMDP filtering: state-by-time belief heatmap, trajectories, and optional discrete VFE trace. |
| `plot_policy_efe_decomposition(traces, *, policy_labels=None, posterior=None, ...)` | Fig. 9.6 — policy EFE split into reward-seeking risk, information-seeking ambiguity, optional novelty, and per-step `G`. |
| `plot_parameter_learning(history, confidence, *, truth=None, symbol="A", ...)` | Figs 10.1.3/10.1.4 — learned `A`/`B` entries converging on the truth (left) and Dirichlet pseudocounts (confidence) growing (right). |
| `plot_two_armed_bandit(result, *, title=...)` | Fig 10.3.6/7 — §10.3 two-armed bandit: context belief → truth, policy posterior, reward outcomes. |
| `plot_factorial_likelihood(model, *, title=...)` | Fig 10.3.4 — §10.3 factorial likelihood `A` set, one heatmap per modality (factors flattened). |
| `plot_hierarchical_timescales(result, *, title=...)` | Fig 10.4.1 — §10.4 hierarchical: slow top belief, top-down prior, fast bottom belief over nested time scales. |

## Conventions

- Every helper accepts an optional `save_path`; figures are dpi 150 by
  default and laid out with `constrained_layout=True`.
- Animations always set `_fig` on the returned object so `save_animation`
  can close the figure cleanly afterwards.
- Color choices route through `visualizations.style.COLORS`, an Okabe-Ito
  colourblind-safe vocabulary. Use perceptually ordered colormaps only for
  scalar fields and heatmaps where continuous value order is the message.
