# Chapter 9 — Active Inference in POMDPs

Chapters 6–8 used *continuous* states with Gaussian densities. Chapter 9 switches to the
**discrete** (categorical) formulation — the partially-observable Markov decision process
(POMDP) — which is the most widely-used form of active inference. Everything is built from
a few matrices (`A`, `B`, `C`, `D`). The concept map is
[`docs/chapters/chapter_09.md`](../../docs/chapters/chapter_09.md).

> **Scope.** This folder implements §9.1 (the `A`/`B`/`C`/`D` model + exact state inference),
> §9.2–9.3 (dynamic forward filtering and discrete variational free energy), and §9.4–9.6
> (**expected free energy**, policy/action selection, Grid World planning, and the
> exploration/exploitation trade-off).

## Learning objective

- Construct a discrete POMDP generative model with A, B, C, and D arrays to
  infer hidden states and select policies by minimizing expected free energy.

Practice this objective with `example_9_1_state_inference.py` and
`example_9_4_gridworld.py`.

## Scripts

| Script | Mirrors | What it shows |
|--------|---------|---------------|
| `example_9_1_state_inference.py` | Example 9.1 / §9.1 | Discrete Bayesian state inference: a weather agent inverts its categorical generative model (`s = σ(log Aᵀô + log D)`) to infer the state from a one-hot observation. Reproduces Fig. 9.1.3. |
| `example_9_2_dynamic_filtering.py` | §9.2–§9.3 | Hidden Markov forward filtering: `B` propagates yesterday's posterior into today's prior, `A` assimilates the new observation, and the per-step discrete VFE is evaluated at the posterior. |
| `example_9_3_discrete_vfe.py` | §9.3 | A two-state simplex sweep showing that categorical VFE is minimized exactly at the Bayes posterior and touches the surprisal bound. |
| `example_9_4_gridworld.py` | §9.4–9.5 / Alg. 9.5.1 | A discrete active-inference agent navigates a 3×3 Grid World to a goal by minimizing **expected free energy** `G = risk + ambiguity` over policies. Planning as inference. |
| `example_9_6_exploration_exploitation.py` | §9.6 | Decomposes policy EFE into **risk** and **ambiguity** so the reward-seeking vs information-seeking trade-off is visible. |
| `animation_belief_filtering.py` | §9.2 | GIF animation of the categorical belief update rhythm: predict with `B`, update with `A`. |
| `animation_efe_tradeoff.py` | §9.6 | GIF sweep showing how sharper preferences move selection from ambiguity-dominated exploration toward risk-dominated exploitation. |

## Running

```bash
uv run python chapters/chapter_09/example_9_1_state_inference.py --save
# or
./run.sh --chapter 9
```

## Programmatic usage

```python
from active_inference import POMDPModel, infer_states
import numpy as np

A = np.array([[0.80, 0.33, 0.05, 0.40],   # P(obs | state); columns = states
              [0.15, 0.33, 0.30, 0.05],
              [0.05, 0.34, 0.65, 0.55]])
model = POMDPModel(A=A, D=np.full(4, 0.25))
print(infer_states(model, 1))   # observe "hot" → [0.18, 0.40, 0.36, 0.06]  (book Eq. 15)
```
