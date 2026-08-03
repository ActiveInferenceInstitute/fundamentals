# Chapter 8 — Learning, attention, and hierarchical models

Chapter 8 extends continuous-state generalized filtering with slow learning of
first-order parameters, attention-like learning of second-order log precisions,
and hierarchical message passing. It is the bridge between the continuous
recognition dynamics of Chapters 6–7 and the explicit hierarchical / discrete
architectures of Chapters 9–10. The concept map is
[`docs/chapters/chapter_08.md`](../../docs/chapters/chapter_08.md).

## Scripts

| Script | Book section | What it shows |
|---|---:|---|
| `example_8_1_learning_attention.py` | §8.1 | Triple estimation of hidden state, observation parameter, and log precision. |
| `example_8_2_hierarchical_continuous.py` | §8.2-§8.4 | Two-layer continuous hierarchy where an upper context supplies a lower-state prior. |
| `visualize_message_passing.py` | §8.5 | Forward error messages and backward prediction messages in a hierarchical model. |
| `animation_learning_attention.py` | §8.1 | GIF of state, parameter, precision, and free-energy convergence. |

## Running

```bash
uv run python scripts/run_all_figures.py --chapters 8
./run.sh --chapter 8
```

`example_8_1_learning_attention.py` accepts `--seed` and `--n-steps`.
`example_8_2_hierarchical_continuous.py` accepts `--n-steps` and `--dt`.
`animation_learning_attention.py` accepts `--seed` and `--frames`.

## Programmatic usage

```python
from active_inference import (
    LearningAttentionModel,
    LearningAttentionState,
    simulate_learning_attention,
)
import numpy as np

# One VFE objective descends state, first-order parameter, and log precision.
model = LearningAttentionModel(
    state_attractor=5.0, theta_prior_mean=0.0, zeta_prior_mean=0.0,
    sigma2_y=0.02, sigma2_theta=50.0, sigma2_zeta=50.0,
)
ys = 5.0 + np.random.default_rng(0).normal(0.0, 0.08, size=100)
res = simulate_learning_attention(
    model, ys, initial=LearningAttentionState(mu_x=1.5, mu_theta=0.0, mu_zeta=0.0),
    dt=0.03, kappa_x=0.7, kappa_theta=2.0, kappa_zeta=0.25, damping=1.2,
)
print(res.final_state.mu_x, res.final_variance_x)
```

## Reading the figures

- **Fast perception:** `mu_x` responds on every observation to reduce sensory
  prediction error.
- **Slow learning:** `mu_theta` uses a damped velocity flow so parameter learning
  is visibly slower than state tracking.
- **Attention / precision:** `mu_zeta` is a log precision, so the learned variance
  stays positive by construction.
- **Hierarchy:** the lower layer explains sensory data while the upper layer supplies
  a contextual prior; the message-passing diagram labels bottom-up errors and
  top-down predictions separately.
