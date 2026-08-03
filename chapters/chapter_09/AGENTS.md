# `chapters/chapter_09/` — Active Inference in POMDPs

Chapter 9 scripts implement the **discrete (categorical)** formulation of active inference.
The generative model is built from matrices `A` (likelihood `P(o|s)`), `B` (transitions
`P(s'|s,u)`), `C` (preferences over observations), `D` (state prior) — the standard POMDP /
"pymdp-style" active inference.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_9_1_state_inference.py`](example_9_1_state_inference.py) | ~70 | Exact discrete hidden-state inference (`s = σ(log Aᵀô + log D)`, Eq. 13); reproduces Fig. 9.1.3 and the Eq. 15 numbers. |
| [`example_9_2_dynamic_filtering.py`](example_9_2_dynamic_filtering.py) | ~80 | Forward filtering with `B`-propagated priors and `A`-based observation updates. |
| [`example_9_3_discrete_vfe.py`](example_9_3_discrete_vfe.py) | ~105 | Simplex sweep showing categorical VFE minimized at the exact posterior. |
| [`example_9_4_gridworld.py`](example_9_4_gridworld.py) | ~95 | Grid World planning agent (Alg. 9.5.1): navigates to a goal by minimizing expected free energy over policies. |
| [`example_9_6_exploration_exploitation.py`](example_9_6_exploration_exploitation.py) | ~90 | Risk/ambiguity decomposition of expected free energy for exploration vs exploitation. |
| [`animation_belief_filtering.py`](animation_belief_filtering.py) | ~70 | GIF of the categorical predict-update rhythm across observations. |
| [`animation_efe_tradeoff.py`](animation_efe_tradeoff.py) | ~70 | GIF showing preferences sharpening policy selection. |

## Running

```bash
python chapters/chapter_09/example_9_1_state_inference.py --save
python scripts/run_all_figures.py --chapters 9
```

## Library Usage

```python
from active_inference import POMDPModel, infer_states, softmax, one_hot, is_stochastic_matrix
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs each non-interactive script with `--save`
in the single parametrized `test_chapter_script_runs_and_exports_raw_data`
(one case per discovered chapter script).
Unit tests live in `tests/core/test_pomdp.py` (verified against the book's exact Eq. 15
posterior `[0.18, 0.40, 0.36, 0.06]`).

## Key Concepts

- **Categorical everything.** `D ∈ [0,1]^C` (state prior), `A ∈ [0,1]^{O×C}` (likelihood,
  columns sum to 1), `B ∈ [0,1]^{C×C}` per control (transitions, columns sum to 1).
- **Exact state inference** (§9.1): for a one-hot observation `ô`, the posterior is
  `s = σ(log Aᵀô + log D)` — likelihood × prior, normalized, equivalently a softmax of
  (log-likelihood + log-prior). No variational approximation needed in the static case.
- **One-hot observations** `ô ∈ {0,1}^O` pick out the observed row of `A` via `Aᵀô`.
- **Expected free energy (§9.5).** `G(π) = risk + ambiguity` (Eq. 52). **Risk** (reward-seeking,
  Eq. 60) `= o·(log o − log C)` with `o = A·s` — KL from predicted observations to preferences.
  **Ambiguity** (information-seeking, Eq. 64) `= H·s` with `H = −diag(Aᵀlog A)` (per-state
  observation entropy). Policy posterior `Q(π) = σ(−γ G)` (Eq. 55); action posterior
  `Q(u) = Σ_π δ(u=V[π,τ])·Q(π)` (Eq. 69). The agent picks the lowest-EFE action.
- **Convention note (verified):** the book's Example 9.7 risk oracle (6.337 nats) uses `C`
  **raw** with an ε-floor for `log`, *not* softmax-normalized — `efe_risk` matches this.
- **Dynamic filtering (§9.2–§9.3).** `forward_filter` carries yesterday's posterior
  forward through `B`, assimilates today's observation through `A`, and evaluates
  discrete VFE at the posterior.
- **Visualization contract.** EFE plots must keep risk, ambiguity, total `G`, and
  policy posterior visually separable so the exploration/exploitation argument is
  inspectable.
