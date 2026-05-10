# Active inference

Active inference is a framework for modeling agents that *minimize a single
objective* — variational free energy — by choosing both their internal
beliefs and their actions in the world. Perception updates beliefs to make
them consistent with observations; action changes the world to make
observations consistent with beliefs. Together they form a closed loop
governed by one optimization principle.

This article is the bridge between the Bayesian-inference machinery the
package already ships and the broader theory readers will meet later in
the book. Everything shown here is implementable today with classes and
functions in `active_inference`.

## The four ingredients

Every active-inference model needs:

1. **A generative process** — the environment that produces observations.
   In this codebase: `LinearGaussianProcess`, `LinearGaussianMVProcess`,
   or any subclass of `GenerativeProcess`.
2. **A generative model** — the agent's beliefs about that environment.
   `LinearGaussianModel`, `LinearGaussianMVModel`, or any subclass of
   `GenerativeModel`.
3. **A perception step** — invert the model to obtain a posterior over
   hidden states. `GridBayesianInference.infer` (1-D),
   `LinearGaussianSystem.posterior_batch` (multivariate), or the
   sequential trace from `core.compose.running_stats`.
4. **An action step** — change the environment so that future
   observations reduce expected free energy. The minimal embodiment of
   this in the current package is the `Pipeline.run(x_star, n)` loop:
   the agent picks ``n``, the environment produces ``y``, and the
   posterior tightens.

The first three ingredients are the *passive* agent of Part I of the
book; adding the fourth makes the agent *active*.

## How the codebase realizes the loop

```python
import numpy as np
from active_inference import Pipeline

# 1. Build the loop: process + model + grid in one call.
pipeline = Pipeline.linear_gaussian(
    beta0=3.0, beta1=2.0, sigma2_y=0.4,
    m_x=4.0, s2_x=1.0,
    rng=np.random.default_rng(0),
)

# 2. Perception: a single posterior from a small batch.
result = pipeline.run(x_star=2.0, n=20)
print(result.summary())

# 3. Action surrogate: choose `n` to reduce posterior entropy below a
#    tolerance — a one-step expected-free-energy proxy.
target_entropy = -0.5
n = 1
while result.entropy() > target_entropy and n < 200:
    n += 5
    result = pipeline.run(x_star=2.0, n=n)
print(f"converged at N = {n}, entropy = {result.entropy():.3f}")
```

The third block is a stripped-down stand-in for the *expected free energy*
calculation that drives action selection in full active-inference models.
In a richer setup, the agent would compare candidate actions by their
expected posterior entropy and information gain, not just accept whatever
``n`` happens to be next.

## Markov blankets in this codebase

A Markov blanket separates *internal* states (what the agent computes)
from *external* states (what generates the data) by way of *blanket*
states (sensory inputs and actions). The package's separation between
`generative_process` and `generative_model` already enforces this
distinction: the model never *samples* from the environment, the process
never *evaluates* the model's densities.

| Layer | What it is | What it can do |
|---|---|---|
| External | true ``x*`` | sampled from `process.sample(x_star, n)` only |
| Sensory  | observed ``y`` | passed through `Pipeline.infer(y)` into the model |
| Internal | belief over ``x`` | `InferenceResult` / `LGSPosterior` |
| Active   | action / experiment | choosing ``n``, the prior, the grid |

The package does not yet ship a control layer (deciding *which* action
minimizes expected free energy) — adding one is the natural next
extension; see the planned Chapter 7+ work in the roadmap.

## Information-theoretic bookkeeping

Active inference takes information theory at face value. Every quantity
listed below is a one-liner with the helpers in `core.diagnostics`:

| Quantity | Identifier | Reading |
|---|---|---|
| Posterior entropy | `result.entropy()` | uncertainty in the current belief |
| Belief update size | `result.kl_from_prior()` | KL[posterior ‖ prior]; how much the data moved the agent |
| Forecast quality | `log_score_gaussian`, `crps_gaussian` | proper scoring rules of the predictive |
| Calibration of the agent | `calibration_curve` | whether credible regions cover at the nominal rate |
| Replicate agreement | `posterior_predictive_check` | does data we'd simulate look like data we saw |

These are the bookkeeping entries the agent's loop ought to track. In
the running-stats helper they all appear as functions of ``N``.

## Pitfalls

- **Action ≠ control law (yet).** The current package implements the
  perception side fully; "action" here is loosely the choice of how
  many observations to assimilate. Don't claim a full active-inference
  loop without an explicit policy.
- **Free energy in the package is *variational* free energy** computed
  on a grid for 1-D models. Closed-form Gaussian KL +
  `gaussian_entropy_*` give analytic bounds for the multivariate case.
- **The blanket separation is a convention, not a runtime barrier.**
  Mixing process methods into model code will silently work but break
  the abstraction. Stay disciplined about which side you're on.

## See also

- [`free_energy_principle.md`](free_energy_principle.md) — what the
  agent is minimizing.
- [`bayesian_mechanics.md`](bayesian_mechanics.md) — the
  density-dynamics formalism behind the loop.
- [`bayesian_inference.md`](bayesian_inference.md) — the inversion step.
- [`generative_models.md`](generative_models.md) — the process/model
  split the Markov blanket leans on.
- [`../reference/core.md`](../reference/core.md) — `Pipeline`,
  `running_stats`, `LinearGaussianSystem`, and the full diagnostics
  table.
