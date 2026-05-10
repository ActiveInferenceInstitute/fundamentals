# Chapter 2 — concept map

Chapter 2 turns the conceptual machinery of Chapter 1 into a concrete
modelling pattern: define a generative process, define a generative model,
choose a grid, infer. The chapter then explores the joint distribution from
several angles and introduces the closed-form and iterative point-estimate
alternatives (MLE / MAP).

## The standard recipe

Every example in this chapter follows the same five steps; the `chapters/`
scripts simply parameterize them differently.

1. **Experimental setting** — pick a domain for the hidden state and a domain
   for the observations.
2. **Generative process** — `LinearGaussianProcess(beta0, beta1, sigma2_y)`,
   optionally with a `nonlinear` ``psi``.
3. **Generative model** — `LinearGaussianModel(...)` with a Gaussian or
   uniform prior over the hidden state.
4. **Inference** — `GridBayesianInference(model, grid).infer(y_obs)`.
5. **Diagnostics / visualizations** — posterior mode, mean, credible
   interval, and the three-panel prior / likelihood / posterior plot.

## What each example explores

| File                                 | Question it answers |
|--------------------------------------|---------------------|
| `example_2_1_linear_deterministic.py` | What does Bayes do in the noiseless limit? (Answer: it inverts ``g``.) |
| `example_2_2_linear_probabilistic.py` | How does a Gaussian prior pull the posterior away from the data? |
| `example_2_3_precision.py`            | How does the posterior interpolate between the prior and the likelihood as their precisions change? |
| `example_2_4_nonlinear_deterministic.py` | What goes wrong with non-injective generators? |
| `example_2_5_nonlinear_probabilistic.py` | Does a localized prior resolve the ambiguity? |
| `example_2_6_imperfect_model.py`      | What happens when the agent's model differs from the true process? |
| `example_2_7_multiple_samples.py`     | How does the posterior tighten as N grows, and is sequential = batch? |
| `example_2_8_mle_analytic.py`         | What is the closed-form MLE under linear-Gaussian assumptions? |
| `example_2_9_map_analytic.py`         | What is the closed-form MAP, and how does it move with `s2_x`? |
| `example_2_10_gradient_descent.py`    | Can we recover the same answers iteratively? |
| `visualize_generative_model.py`       | What does the joint `p(x, y)` look like as a heatmap and surface? |
| `interactive_explorer.py`             | Drag-and-drop exploration with sliders. |

## Reusable building blocks

* `active_inference.LinearGaussianProcess` — a configurable generative process.
* `active_inference.LinearGaussianModel` — a configurable generative model.
* `active_inference.GridBayesianInference` — one-line exact inference.
* `active_inference.gradient_descent` — drop-in replacement for analytic
  inversion when you want to study optimization dynamics.

## Where the book takes this next

Chapter 3 introduces parameter learning (MLE and Bayesian linear regression)
and Chapter 4 generalizes the entire setup with variational Bayes. The
classes here are designed so that those upgrades fit naturally — for example,
`LinearGaussianModel` already exposes both `log_likelihood` and `log_prior`,
which is exactly the interface a variational scheme needs.
