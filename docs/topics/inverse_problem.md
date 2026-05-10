# The inverse problem

The agent never observes the hidden state directly; it observes a noisy
function of it. Recovering ``x`` from ``y`` is therefore an *inverse
problem*, and Bayesian inference solves it only when the posterior
``p(x | y)`` is well-defined. When the generator is non-injective, when
the prior is improper, or when the noise is too tight relative to the
grid, the inverse problem becomes ill-posed.

## In this codebase

- **Bi-modal posteriors from non-injective generators:** see
  `chapters/chapter_01/04_inverse_problem.py` and
  `chapters/chapter_02/example_2_4_nonlinear_deterministic.py`.
- **Restoring identifiability via a localized prior:**
  `chapters/chapter_02/example_2_5_nonlinear_probabilistic.py`.
- **Hidden-state inference from sensor fusion (well-posed):**
  `chapters/chapter_03/example_3_6_lgs_food_localization.py`.
- **Detecting calibration failures (a downstream symptom of an
  ill-posed posterior):**
  `chapters/chapter_03/visualize_calibration.py`.

## The diagnostic toolkit

When you suspect an inversion is ill-conditioned, reach for:

| Diagnostic | What it tells you |
|---|---|
| `InferenceResult.kl_from_prior()` | Did the data move the belief at all? |
| `InferenceResult.entropy()` | How concentrated is the posterior? |
| `InferenceResult.credible_interval(0.95)` | Where is the mass? |
| `core.diagnostics.calibration_curve` | Are credible intervals trustworthy across many trials? |
| `core.diagnostics.posterior_predictive_check` | Do the agent's replicates look like the data? |

## Pitfalls

- A bi-modal posterior is not a bug — it's a faithful report that the
  generator is non-injective. Either widen the prior to allow both
  modes or narrow it to break the symmetry.
- A near-zero `kl_from_prior()` after assimilating data usually means
  the prior is so peaky that the likelihood is being dominated. Check
  the prior variance.
- A wide grid hides the problem; a narrow grid creates artefacts.
  Always plot the joint density before trusting numbers.

## See also

- [`bayesian_inference.md`](bayesian_inference.md) — the inversion
  recipe.
- [`generative_models.md`](generative_models.md) — what is being
  inverted.
- [`../statistics/calibration.md`](../statistics/calibration.md) —
  detecting systematic miscalibration.
- [`../statistics/posterior_predictive.md`](../statistics/posterior_predictive.md)
  — detecting model–data mismatch.
