# Chapter 10 — Learning and extensions in POMDPs

Chapter 9 assumed the POMDP matrices `A`, `B`, `D` were *known* and that there was a single
hidden state and observation. Chapter 10 lifts both assumptions: it **learns** the matrices
from data (§10.1–10.2) and extends the model to **factorial** (multiple state factors +
observation modalities, §10.3) and **hierarchical** (nested layers, §10.4) depth. The concept
map is [`docs/chapters/chapter_10.md`](../../docs/chapters/chapter_10.md).

> **Scope.** This folder implements the **complete chapter**: §10.1 — Dirichlet update rules
> for `A`/`B`/`D` (Eq. 4–6), the **parameter-novelty** term (Eq. 12–15), the learning agent
> (Algorithm 10.1.1); §10.2 — habit (baseline prior `E`) and **policy precision** `γ`,
> including learning `γ` from a Gamma prior (Eq. 20–25); §10.3 — **factorial depth** and the
> two-armed bandit (Example 10.7); §10.4 — **hierarchical depth** (Eq. 39–50). Verified
> against the book's worked Examples 10.1–10.7 plus reduction/self-consistency oracles.

## Learning objective

- Update POMDP parameters through Dirichlet concentration counts and extend
  discrete active inference with policy precision, habits, factorial state
  factors, and hierarchical depth.

Practice this objective with `example_10_2_learn_A.py` and
`example_10_7_two_armed_bandit.py`.

## Scripts

| Script | Mirrors | What it shows |
|--------|---------|---------------|
| `example_10_1_learn_D.py` | Example 10.1 / Fig 10.1.2 | Learn the state prior `D` by accumulating initial-state beliefs (`d ← d + s⁽⁰⁾`). Reproduces the book's `d = [45.1, 5.9]`, `D ≈ [0.884, 0.116]`. |
| `example_10_2_learn_A.py` | Example 10.2 / Fig 10.1.3 | Learn the likelihood `A` by counting (observation, state) pairs over trials; entries converge on the truth while confidence (pseudocounts) grows. |
| `example_10_3_learn_B.py` | Example 10.3 / Fig 10.1.4 | Learn the transition `B` by counting (next-state, current-state) pairs. |
| `example_10_4_novelty.py` | Example 10.4 / §10.1 | The parameter-novelty term `o·(Ws)` (= 0.483 oracle) and a novelty-seeking agent that learns `A` while acting (Alg. 10.1.1). |
| `example_10_5_precision.py` | Example 10.5 / Fig 10.2.2–3 | Policy posterior `σ(log E − γ G)` swept over precision `γ` for uniform vs strong habits; mass concentrates on the lowest-EFE policy. Reproduces Fig 10.2.3 exactly. |
| `example_10_6_precision_learning.py` | Example 10.6 / Fig 10.2.4 | Learn `γ` from a Gamma prior (Eq. 23–25): `F` close to `G` ⇒ high confidence, `F` far ⇒ low confidence. |
| `example_10_7_two_armed_bandit.py` `[--explore]` | Example 10.7 / Figs 10.3.6–7 | **§10.3 factorial.** The two-armed bandit: a factorial agent infers the hidden context (better machine) and exploits it. `--explore` runs the less risk-averse agent. |
| `example_10_8_hierarchical.py` `[--regime]` | §10.4 / Fig 10.4.1 | **§10.4 hierarchical.** A two-layer POMDP where a slow top regime contextualizes a fast bottom layer over nested time scales. |
| `visualize_factorial_structure.py` | §10.3 / Fig 10.3.4 | Heatmaps of the two-armed bandit's factorial likelihood `A` set (each modality conditioned on all state factors). |

## Animations

| Script | What it shows |
|--------|---------------|
| `animation_learning.py` `[--transition]` | Dirichlet learning over trials: `A` (or `B`) entries converging on the truth while pseudocounts grow (Figs 10.1.3/10.1.4 animated). |
| `animation_precision.py` `[--strong]` | The policy posterior bars redrawing as precision `γ` ramps from 0, concentrating on the lowest-EFE policy (Fig 10.2.2 animated). |
| `animation_bandit.py` `[--explore]` | The two-armed bandit context belief + policy posterior evolving step-by-step (Figs 10.3.6/7 animated, §10.3). |

## Running

```bash
uv run python chapters/chapter_10/example_10_2_learn_A.py --save
# or
./run.sh --chapter 10
```

## Programmatic usage

```python
import numpy as np
from active_inference import simulate_array_learning, novelty_matrix, parameter_novelty

# Learn a 2×2 likelihood by counting state–observation pairs (Example 10.2).
A_true = np.array([[0.7, 0.6], [0.3, 0.4]])
B_true = np.array([[0.0, 1.0], [1.0, 0.0]])
res = simulate_array_learning(A_true=A_true, B_true=B_true, learn="A",
                              n_trials=5, steps_per_trial=1000)
print(res.A_history[-1], res.final_A_error())   # → ≈ A_true, error < 0.05

# Parameter information gain (Example 10.4 oracle).
a = np.array([[2.5, 0.15], [0.8, 3.0]])
print(parameter_novelty(a, np.array([0.15, 0.85])))   # → 0.483
```

```python
from active_inference import policy_posterior_full, learn_precision

# §10.2 habit + precision: Q(π) = σ(log E − F − γ G)  (Example 10.5).
G = np.array([3.0, 2.0, 1.5, 2.2, 3.2])
print(policy_posterior_full(G, gamma=1.5))    # concentrates on policy 2 (lowest EFE)

# Learn the precision γ from data (Example 10.6): F close to G ⇒ higher γ.
res = learn_precision(G, np.array([3.5, 2.3, 2.0, 2.5, 4.0]))
print(res.gamma, res.converged)               # self-consistent fixed point (∂F/∂γ ≈ 0)
```

```python
from active_inference import make_two_armed_bandit, simulate_two_armed_bandit

# §10.3 factorial: the two-armed bandit (Example 10.7).
model = make_two_armed_bandit(reward_prefs=(0.0, -3.0, 4.0))
res = simulate_two_armed_bandit(model, true_context=1, n_steps=15)
print(res.context_belief[-1], res.n_wins)     # → ≈ [0, 1] (right-better learned), most pulls win
```

```python
import numpy as np
from active_inference import (HierarchicalPOMDP, POMDPModel, simulate_hierarchical_agent)

# §10.4 hierarchical: a slow top regime contextualizes a fast bottom layer.
top = POMDPModel(A=np.array([[0.9, 0.1], [0.1, 0.9]]), D=np.array([0.5, 0.5]))
bot = POMDPModel(A=np.eye(3), D=np.full(3, 1 / 3))
link = np.array([[0.9, 0.05], [0.05, 0.05], [0.05, 0.9]]); link /= link.sum(0)
h = HierarchicalPOMDP(layers=[bot, top], link=[link])
res = simulate_hierarchical_agent(h, true_top=1, n_macro=5, inner_steps=3)
print(res.top_belief[-1], res.bottom_priors[-1])   # top regime → bottom prior
```
