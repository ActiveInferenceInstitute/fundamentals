# Chapter 6 — Generalized Filtering for Perception

Chapter 6 opens **Part II (Active inference core)** by moving from a *static* hidden
state (Chapters 4–5) to a *dynamic* one that is filtered **online**: at each time
point the environment transitions and emits an observation, and the agent updates
its belief `μ_x` by one Euler step down the variational free energy (Algorithm
6.1.1). The concept map is [`docs/chapters/chapter_06.md`](../../docs/chapters/chapter_06.md).

> **Scope.** This folder implements §6.1 (univariate generalized filtering), §6.2 (the
> multivariate filter), §6.3–6.5 (generalized coordinates of motion + the generalized
> filter), and §6.6 (correlated embedding orders and Example 6.7).

## Learning objective

- Track dynamic hidden states online with generalized filtering and explain how
  generalized coordinates of motion and precision coupling reduce tracking lag.

Practice this objective with `example_6_1_generalized_filter.py` and
`example_6_6_generalized_coordinates.py`.

## Scripts

| Script | Mirrors | What it shows |
|--------|---------|---------------|
| `example_6_1_generalized_filter.py` | Example 6.1 / Alg. 6.1.1 | An agent tracks a moving object's position from a noisy 1-D observation; belief `μ_x` locks onto the true state `x*` (Fig. 6.1.2/6.1.3). |
| `example_6_2_multivariate_filter.py` | Example 6.2 / §6.2 | A 2-D Hooke's-law oscillator; both elements of `μ_x` track the orbiting true state with the lag the book describes (Fig. 6.2.2/6.2.3). |
| `example_6_6_generalized_coordinates.py` | Example 6.6 / §6.5 / Alg. 6.5.1 | Generalized filtering: the belief carries position *and* velocity; the order-1 belief `μ_x'` recovers the true velocity `ẋ*` (Fig. 6.5). |
| `visualize_6_6_correlated_embedding_orders.py` | §6.6 / Fig. 6.6.2 | Heatmaps of `Π̃(γ)` show how smoothness controls precision coupling across embedding orders. |
| `example_6_7_multivariate_generalized_coordinates.py` | Example 6.7 / §6.6 | The Hooke oscillator in vector generalized coordinates with correlated precision; compares ordinary lag against generalized-coordinate tracking. |

## Running

```bash
uv run python chapters/chapter_06/example_6_1_generalized_filter.py --save
# or
./run.sh --chapter 6
```

`--save` writes figures to `output/figures/chapter_06/` and raw NPZ+JSON
sidecars to `output/data/chapter_06/`.

## Programmatic usage

```python
from active_inference import (
    DynamicProcess, DynamicStateSpaceModel, LinearFunction,
    simulate_dynamic_process, generalized_filter, gf_fixed_point_linear,
)
import numpy as np

# Environment: point attractor at 10, observation y = x* - 3.
proc = DynamicProcess(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0))
tr = simulate_dynamic_process(proc, x0=5.0, n_steps=1000, dt=0.01,
                              rng=np.random.default_rng(0))

# Agent: high sensory precision (λ_y=50) ≫ prior precision (λ_x=0.2) → it tracks.
model = DynamicStateSpaceModel(f=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0),
                               s2_x=5.0, sigma2_y=0.02)
res = generalized_filter(model, tr.ys, dt=0.01, kappa=0.1, mu0=15.0)
print(res.tracking_error(tr.xs, burn_in=300))     # ≈ 0.08 — close tracking
print(gf_fixed_point_linear(model, tr.ys[-1]))    # closed-form steady-state ≈ x*
```
