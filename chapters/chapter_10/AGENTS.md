# `chapters/chapter_10/` — Learning and extensions in POMDPs

Chapter 10 scripts (1) **learn** the discrete POMDP arrays `A`/`B`/`D` from data using
**Dirichlet** concentration parameters (§10.1–10.2), and (2) extend the model to **factorial**
(multiple state factors + observation modalities, §10.3) and **hierarchical** (nested layers,
§10.4) depth. Learning = counting co-occurrences and taking the Dirichlet expected value.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_10_1_learn_D.py`](example_10_1_learn_D.py) | ~85 | Learn `D` (Eq. 4c/7/8); reproduces `d=[45.1,5.9]`, `D≈[0.884,0.116]` (Fig 10.1.2). |
| [`example_10_2_learn_A.py`](example_10_2_learn_A.py) | ~60 | Learn `A` by counting state–observation pairs (Eq. 4a); Fig 10.1.3. |
| [`example_10_3_learn_B.py`](example_10_3_learn_B.py) | ~60 | Learn `B` by counting state→state transitions (Eq. 4b); Fig 10.1.4. |
| [`example_10_4_novelty.py`](example_10_4_novelty.py) | ~90 | Parameter novelty `o·(Ws)=0.483` (Eq. 12–19) + novelty-driven learning agent (Alg. 10.1.1). |
| [`example_10_5_precision.py`](example_10_5_precision.py) | ~70 | §10.2 policy posterior `σ(log E − γ G)` swept over precision `γ` (Fig 10.2.3, exact). |
| [`example_10_6_precision_learning.py`](example_10_6_precision_learning.py) | ~80 | §10.2 learning `γ` from a Gamma prior (Eq. 23–25); `F` close vs far from `G`. |
| [`example_10_7_two_armed_bandit.py`](example_10_7_two_armed_bandit.py) | ~60 | §10.3 factorial two-armed bandit (Example 10.7, Figs 10.3.6/7); `--explore` = less risk-averse. |
| [`example_10_8_hierarchical.py`](example_10_8_hierarchical.py) | ~75 | §10.4 hierarchical 2-layer POMDP, nested time scales (Fig 10.4.1). |
| [`visualize_factorial_structure.py`](visualize_factorial_structure.py) | ~45 | §10.3 heatmaps of the bandit's factorial `A` set (Fig 10.3.4). |
| [`animation_learning.py`](animation_learning.py) | ~60 | Animated Dirichlet learning of `A`/`B` over trials (Figs 10.1.3/10.1.4). |
| [`animation_precision.py`](animation_precision.py) | ~55 | Animated policy-precision sweep (Fig 10.2.2). |
| [`animation_bandit.py`](animation_bandit.py) | ~55 | §10.3 animated two-armed bandit (context belief + policy posterior over time). |

## Running

```bash
python chapters/chapter_10/example_10_2_learn_A.py --save
python scripts/run_all_figures.py --chapters 10
```

## Library Usage

```python
from active_inference import (
    dirichlet_expected_value, expected_A, expected_B, expected_D,
    accumulate_a_counts, accumulate_b_counts, accumulate_d_counts,
    novelty_matrix, parameter_novelty, efe_with_novelty,
    DirichletParameters, LearningResult, learn_D_vector,
    simulate_array_learning, simulate_learning_agent,
    # §10.2 — habit + precision
    policy_posterior_full, precision_gradient, learn_precision,
    PrecisionResult, precision_policy_sweep,
    # §10.3 — factorial depth
    FactorialPOMDP, factorial_infer_states, factorial_efe,
    factorial_expected_observation, factorial_likelihood_message,
    factorial_modality_ambiguity, factorial_predict_states,
    make_two_armed_bandit, simulate_two_armed_bandit, TwoArmedBanditResult,
    # §10.4 — hierarchical depth
    HierarchicalPOMDP, hierarchical_layer_vfe, hierarchical_layer_efe,
    hierarchical_policy_posterior, simulate_hierarchical_agent, HierarchicalResult,
)
from active_inference.visualizations import (
    animate_parameter_learning, animate_policy_precision, animate_two_armed_bandit,
)
from active_inference.visualizations.unified import (
    plot_two_armed_bandit, plot_factorial_likelihood, plot_hierarchical_timescales,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs every discovered Chapter 10 script with
`--save` in the single parametrized
`test_chapter_script_runs_and_exports_raw_data`. Unit tests live in
`tests/core/test_pomdp.py` (`TestDirichletLearning`, `TestHabitAndPrecision`,
`TestFactorialDepth`, `TestHierarchicalDepth`) and `tests/estimators/test_pomdp.py`
(`TestParameterLearning`, `TestPrecisionSweep`, `TestTwoArmedBandit`,
`TestHierarchicalAgent`), verified against the book's exact Example 10.1 (`D=[0.884,0.116]`),
Example 10.4 (`W`, `o·(Ws)=0.483`), and Example 10.5 (Fig 10.2.3) oracles plus
factorial/hierarchical reduction tests.

## Key Concepts

- **Dirichlet = conjugate prior of the categorical** ⇒ learning is counting. Each array gets
  pseudocounts: `a (O×C)`, `b (C×C` per control`)`, `d (C,)`.
- **Trial-based learning.** Parameters are fixed during a trial; at the end, the accumulated
  counts are added and the arrays refreshed via the Dirichlet mean (Eq. 5):
  `A = a_ij / Σ_i a_ij`, `B` likewise per slice, `D = d / Σd`.
- **Update rules (Eq. 4):** `a += Σ_τ o⁽τ⁾ ∘ s⁽τ⁾`, `b += Σ_τ s⁽τ⁾ ∘ s⁽τ⁻¹⁾`,
  `d += s⁽⁰⁾` (`∘` = outer product). For action-dependent `B`, accumulate per control slice
  weighted by the policy posterior (Eq. 6).
- **Confidence grows with counts.** Pseudocounts increase linearly with observed pairs, so the
  estimate becomes progressively harder to move — the Dirichlet's built-in confidence.
- **Parameter novelty (Eq. 12–15).** `W ≈ ½(1/a − 1/a₀)` (`a₀` = column-sum broadcast) scores
  the information gain of one more count; `novelty = o·(Ws)` with `o = A s`. The
  novelty-augmented EFE `G = risk + ambiguity − novelty` (Eq. 15) drives *exploration that
  learns* — visiting state–observation pairs the agent is least confident about.
- **Convention note (verified):** `A = E[Dir(a)]` column-normalizes (not row); the novelty
  matrix uses `a` **raw** with column-sum `a₀`. Both reproduce the book's Example 10.4 numbers
  (`A=[[0.758,0.048],[0.242,0.952]]`, `W=[[0.048,3.175],[0.473,0.008]]`, `o·(Ws)=0.483`).

### §10.2 — habit prior `E` and policy precision `γ`

- **Full policy posterior (Eq. 22):** `Q(π) = σ(log E − F − γ G)`. `E` is a baseline
  prior/habit (uniform ⇒ no bias), `F` the policy-dependent VFE evidence (`None` ⇒ EFE-only),
  `γ` the precision scaling `G`. Reproduces Example 10.5 / Fig 10.2.3 **exactly** (γ=0 ⇒ habit;
  γ↑ ⇒ concentrate on min-EFE policy).
- **Learning precision (Eq. 23–25):** `γ = 1/β`, β a Gamma rate; descend VFE in γ via
  `∂F/∂γ = (β−β₀) + (π−π₀)·(−G)` with `π₀=σ(log E − γG)`, `π=σ(log E − F − γG)`; update
  `β ← β − κ_γ ∂F/∂γ`, `γ = α/β` (α=1). The fixed point is self-consistent (`∂F/∂γ → 0`).
- **Convention note (verified by behavior):** precision learning reproduces the book's
  **qualitative** Example 10.6 result — `F` close to `G` ⇒ higher γ (more confident), `F` far
  ⇒ lower γ. The exact magnitudes (book `γ=1.357 / 0.493`) depend on the Gamma prior rate
  `β₀` / scaling from the supplemental-material derivation (not in the PDF); we verify the
  self-consistent fixed point + ordering rather than transcribe an unverifiable constant —
  the same discipline used for the loose continuous-chapter gradient signs.

### §10.3 — factorial depth (state factors + observation modalities)

- **Sets of arrays (Eq. 32–34):** `A = {A^(m)}` (one per modality, each `(O_m, C_0, …, C_N)`
  conditioned on *all* factors), `B = {B^(n)}` (one per factor), `C = {C^(m)}`, `D = {D^(n)}`.
- **Mean-field inference (Eq. 35–37):** `Q(s) = ∏_n Q(s^(n))`; each factor =
  `σ(log prior^(n) + Σ_m E_{s∖n}[log A^(m)]·o^(m))` (variational message passing). Reduces
  **exactly** to Chapter 9 when `N=M=0` (verified vs the Eq. 15 weather posterior).
- **Factorial EFE (Eq. 38a):** `G = Σ_m [o^(m)·(log o^(m) − log C^(m)) + H^(m)·s]`. Identity:
  `risk + ambiguity = (−o·log C) − I(o;s)` (pragmatic minus info-gain). `C` is
  **softmax-normalized** per the book (p. 620), unlike the raw-`C` Ch.9 convention.
- **Two-armed bandit (Example 10.7):** context factor (fixed, identity `B`) + choice factor
  (controllable); hint/reward/choice modalities. The agent learns the hidden context and
  exploits the better arm; the hint carries `ln 2` nats of context info-gain.

### §10.4 — hierarchical depth (nested POMDP layers)

- **Top-down linking (Eq. 42):** a higher layer's state sets the lower layer's initial-state
  prior, `prior = link·s^[l+1]` (`link` column-stochastic). Slow high level contextualizes
  fast low level — nested time scales.
- **Per-layer VFE/EFE (Eq. 43/50)** and the layer-wise policy posterior (Eq. 49) are identical
  in form to the flat case — `hierarchical_layer_vfe`/`_efe` reduce to `discrete_vfe` /
  `expected_free_energy` (verified). The book gives no numerical oracle for §10.4, so
  verification is by reduction + a constructed top-down-control demonstration.
