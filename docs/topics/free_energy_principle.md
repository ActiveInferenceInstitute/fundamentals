# The free energy principle

The free energy principle (FEP) says that any system that maintains its
own boundary against decay must, on average, minimize a quantity called
*variational free energy* — an upper bound on the negative log evidence
of the data it observes. It is a single objective that simultaneously
explains perception (update beliefs to reduce free energy) and action
(change the world to reduce free energy). Variational inference is the
machinery; the FEP is the claim that biological agents *implement* it.

This article is the mathematical companion to
[`active_inference.md`](active_inference.md). Where that page describes
the agent loop, this page makes precise what is being minimized.

## Variational free energy

Given a generative model ``p(x, y)`` and an *approximate posterior*
``q(x)``, variational free energy is

```
F[q] = E_q[ log q(x) − log p(x, y) ]
     = KL[ q(x) ‖ p(x | y) ]  −  log p(y)
     = E_q[ −log p(y | x) ]   +  KL[ q(x) ‖ p(x) ]
       └── expected energy ──┘   └── complexity ──┘
```

Three readings of the same number:

1. **Bound on surprise.** ``F ≥ −log p(y)``, with equality when
   ``q(x) = p(x | y)``. Minimizing ``F`` therefore upper-bounds the
   surprise (negative log evidence) the agent assigns to its data.
2. **Posterior approximation error + irreducible cost.** ``F`` decomposes
   into ``KL[ q ‖ posterior ]`` (how wrong the agent is) plus
   ``−log p(y)`` (how unexpected the data are under the model). The
   first term is what variational inference drives down.
3. **Accuracy − complexity.** Equivalent rewriting: ``F`` is the
   negative expected log-likelihood (an *accuracy* cost) plus
   ``KL[ q ‖ prior ]`` (a *complexity* cost). Good agents fit the data
   without straying too far from their priors.

For Gaussian models with Gaussian variational posteriors, every term
has a closed form — and the package ships every one of them.

## How the codebase realizes each term

| Term | Identifier | When it lights up |
|---|---|---|
| Posterior entropy ``H[q]`` | `InferenceResult.entropy`, `gaussian_entropy_univariate`, `gaussian_entropy_mvn` | Reading uncertainty off any grid posterior or analytic Gaussian. |
| Complexity ``KL[q ‖ prior]`` | `InferenceResult.kl_from_prior`, `grid_kl_divergence`, `gaussian_kl_univariate`, `gaussian_kl_mvn` | Reading how much the data moved the agent. |
| Log evidence ``log p(y)`` | `InferenceResult.log_evidence`, `RunningPosteriorStats.log_evidences` | Comparing models / monitoring streams. |
| Accuracy ``E_q[ log p(y\|x) ]`` | implicitly via `log_score_gaussian(y, mu, sigma2)` summed over a held-out set | Quantifying predictive fit. |
| ``F`` itself | `−log_evidence + grid_kl_divergence(posterior, prior, x_grid)` | Direct numerical bound. |

Concretely, after one call to `GridBayesianInference.infer`, every term
is one method away on the returned `InferenceResult`:

```python
F = -result.log_evidence + result.kl_from_prior()
# When the grid is fine and the model is well-specified, this should
# track −log p(y) closely (the bound is tight when q ≈ true posterior).
```

For the multivariate case, the same identity holds with
`gaussian_kl_mvn(post.mean, post.cov, prior_mean, prior_cov)` taking
the place of `kl_from_prior`.

## The accuracy / complexity tradeoff in this package

The decomposition ``F = expected_energy + complexity`` is what makes
priors *do work*. A peaky prior cuts complexity but pays an
expected-energy cost when the data disagree; a vague prior is the
opposite. The package's precision sweep
(`chapters/chapter_02/example_2_3_precision.py`) is exactly this
tradeoff swept on a grid.

```python
import numpy as np
from active_inference import (
    LinearGaussianModel, GridBayesianInference, make_grid,
    grid_kl_divergence,
)

x_grid = make_grid(0, 5, 500)
y = 7.0
records = []
for sigma2_y, s2_x in [(1.5, 0.1), (0.25, 0.25), (0.1, 1.5)]:
    m = LinearGaussianModel(beta0=3, beta1=2, sigma2_y=sigma2_y,
                            m_x=4, s2_x=s2_x, prior_kind="gaussian")
    res = GridBayesianInference(m, x_grid).infer(y)
    F = -res.log_evidence + res.kl_from_prior()
    records.append((sigma2_y, s2_x, res.posterior_mode, F))

for r in records:
    print(f"σ²_y={r[0]:.2f} s²_x={r[1]:.2f} → mode={r[2]:.2f} F={r[3]:.3f}")
```

Each row reports a different tradeoff. The free energy ``F`` makes the
comparison commensurable: lower is better.

## Online minimization

If the agent receives observations one at a time, the running
statistics in `core.compose.running_stats` give the trajectory of every
free-energy ingredient in one pass:

```python
from active_inference import Pipeline, running_stats

pipe = Pipeline.linear_gaussian(beta0=3, beta1=2, sigma2_y=0.4,
                                m_x=4, s2_x=1.0)
ys = pipe.sample(x_star=2.0, n=80).flatten()
stats = running_stats(pipe.model, pipe.x_grid, ys)

# F_t (cumulative) for each t.
F_t = -stats.log_evidences + stats.kl_from_prior
```

`F_t` should be monotone non-decreasing — each new observation lowers
the (cumulative) log evidence and raises the complexity term — yet the
*rate* of growth slows as the posterior settles. That slope is the
agent's instantaneous free-energy contribution.

## Pitfalls

- **The bound is variational, not exact.** When ``q`` is constrained
  to a parametric family, ``F`` strictly upper-bounds the surprise.
  Compare two models by their ``F`` only when their ``q`` families are
  comparable.
- **Entropy is differential, not Shannon.** Negative differential
  entropy on a continuous domain is normal for tight Gaussians and is
  not a sign of error.
- **Complexity grows with N.** The ``KL[q ‖ prior]`` term increases as
  data shift the posterior; that is *expected* and visible in
  `RunningPosteriorStats.kl_from_prior`. A flat KL trace usually means
  the prior is overpowering the likelihood.
- **Precision vs variance.** Many FEP papers use precision (= 1/variance)
  natively. The codebase uses variance; convert at the boundary.

## See also

- [`active_inference.md`](active_inference.md) — the agent loop that
  minimizes ``F``.
- [`bayesian_mechanics.md`](bayesian_mechanics.md) — the dynamical
  picture: ``F`` as a Lyapunov function for belief flow.
- [`bayesian_inference.md`](bayesian_inference.md) — exact inference,
  the limit of variational inference when ``q`` is unrestricted.
- [`../statistics/divergences.md`](../statistics/divergences.md),
  [`../statistics/entropy.md`](../statistics/entropy.md) — the building
  blocks of ``F``.
- [`../reference/core.md`](../reference/core.md) — full diagnostics
  table.
