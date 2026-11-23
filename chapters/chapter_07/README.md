# Chapter 7 — Active Generalized Filtering

Chapter 6 built *perception* (an agent that tracks a hidden state). Chapter 7 adds
**action**: the agent now also *changes the world* so that reality conforms to its
preferences. Perception and action minimize the **same** variational free energy —
perception updates the belief `μ_x`, action emits a control signal `a`. The concept map
is [`docs/chapters/chapter_07.md`](../../docs/chapters/chapter_07.md).

> **Scope.** This folder implements the univariate active generalized filter (§7.1–7.4,
> Algorithm 7.2.1) and multivariate AIF in generalized coordinates (§7.5).

## Learning objective

- Couple perception and action under a single variational free-energy objective
  so continuous control steers hidden states toward a preferred set-point.

Practice this objective with `example_7_2_active_inference.py` and
`example_7_5_multivariate_active_inference.py`.

## Scripts

| Script | Mirrors | What it shows |
|--------|---------|---------------|
| `example_7_2_active_inference.py` | Example 7.2 / §7.4 / Alg. 7.2.1 | An agent counters an exogenous current: action drives the true state to the preferred set-point `v` and holds it there (`a → −v*`). Reproduces Fig. 7.4.4/7.4.5. |
| `example_7_5_multivariate_active_inference.py` | §7.5 / Alg. 7.5.1 | A 2-D vector agent uses generalized measurements and vector action to counter an exogenous attractor. |
| `animation_7_5_multivariate_active_inference.py` | §7.5 animated | GIF of the 2-D action-perception loop, showing state, belief, action, sensory error, and free energy over time. |

## Running

```bash
uv run python chapters/chapter_07/example_7_2_active_inference.py --save
# or
./run.sh --chapter 7
```

`--save` writes figures/GIFs to `output/figures/chapter_07/` and raw NPZ+JSON
sidecars to `output/data/chapter_07/`.

## Programmatic usage

```python
from active_inference import (
    ActiveEnvironment, ActiveInferenceAgent, DynamicStateSpaceModel,
    LinearFunction, simulate_active_inference,
)
import numpy as np

# Environment: exogenous attractor at v*=10; agent prefers v=0.
env = ActiveEnvironment(drift=LinearFunction(-1.0, 10.0), g=LinearFunction(1.0, -3.0))
model = DynamicStateSpaceModel(f=LinearFunction(-1.0, 0.0), g=LinearFunction(1.0, -3.0),
                               s2_x=1.0, sigma2_y=0.05)
agent = ActiveInferenceAgent(perception_model=model, forward_model=1.0,
                             kappa_x=0.2, kappa_a=0.4)
res = simulate_active_inference(agent, env, x0=5.0, mu0=5.0, n_steps=6000,
                                action_start=2000, rng=np.random.default_rng(0))
print(res.settled_state(), res.settled_action())   # ≈ 0 (preference), ≈ −10 (counter-force)
```
