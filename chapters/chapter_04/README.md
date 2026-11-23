# Chapter 4 — Variational Bayesian Inference

Chapter 4 is the hinge of the book. Where Chapters 1–3 computed posteriors in
closed form or by point estimation, Chapter 4 asks what to do when the posterior
has **no** closed form, and answers with **variational Bayesian inference**: turn
posterior estimation into optimization by minimizing **variational free energy**
(VFE, `𝓕`). The deeper concept map is [`docs/chapters/chapter_04.md`](../../docs/chapters/chapter_04.md).

This folder contains a thin orchestrator per concept, one animation, three
diagnostic/intuition visualizations, and one interactive explorer. Every script
imports configurable building blocks from `active_inference` and runs on the same
linear-Gaussian example as Chapter 3 (`y = β₀ + β₁x + noise`, prior `x ~ N(4, 0.25)`),
so the exact grid posterior `N(2.4, 0.05)` is available as an oracle.

## Learning objective

- Derive variational free energy as an upper bound on surprisal and minimize it
  with mean-field CAVI and fixed-form variational inference to approximate a
  posterior.

Practice this objective with `example_4_3_vfe_forms.py` and
`example_4_7_fixed_form.py`.

## Scripts

### Numbered examples

| Script | Mirrors | What it shows |
|--------|---------|---------------|
| `example_4_1_coordinate_search.py` | Example 4.1 / Alg. 4.2.1 | Zero-order coordinate search descends the VFE surface (Fig. 4.2.2). |
| `example_4_2_surprisal.py`         | §4.3        | Surprisal `−log p(y)` vs `y` and vs `p(y)` (Fig. 4.3.1). |
| `example_4_3_vfe_forms.py`         | §4.4        | All four VFE forms (G/D/C/E) agree; G/C/E decompositions over a descent (Fig. 4.4.1). |
| `example_4_6_free_form_cavi.py`    | Example 4.6 / Alg. 4.5.1 | Free-form mean-field CAVI on `(x, β₀, β₁)`; VFE monotone non-increasing. |
| `example_4_7_fixed_form.py`        | Example 4.7 / Alg. 4.6.1 | Fixed-form gradient VI converges to the exact posterior `N(2.4, 0.05)` (Fig. 4.6.1/4.6.2). |

### Animation (GIF)

| Script | What it shows |
|--------|---------------|
| `animation_vfe_descent.py` | `q(x)` tightening onto the posterior as VFE falls to the surprisal bound `−log p(y)`. |

### Visualizations

| Script | What it shows |
|--------|---------------|
| `visualize_kl_loss.py`          | The KL loss surface vs the VFE surface (same minimum, no posterior needed). |
| `visualize_vfe_intuition.py`    | G-form intuition: `q(x)`, `p(x, y)`, and the posterior (Fig. 4.2.3). |
| `visualize_model_comparison.py` | Model evidence of a good vs bad model against the true input (Fig. 4.3.2/4.3.3). |

### Interactive (GUI / web-launchable)

| Script | What it shows |
|--------|---------------|
| `interactive_vfe_explorer.py` | Sliders for `μ_x` and `σ_x²` move the variational density `q(x) = N(μ_x, σ_x²)` over the exact posterior; the live bar chart shows the VFE decomposition (free energy, divergence, complexity, accuracy) bottoming out exactly when `q` sits on the posterior. |

## Running

```bash
# from the repo root
uv sync                                                   # one-time setup
uv run python chapters/chapter_04/example_4_7_fixed_form.py --save

# or via the top-level menu
./run.sh --chapter 4
```

Pass `--save` to dump figures (and the GIF) into `output/figures/chapter_04/`.
Without it the scripts open interactive windows.

## Programmatic usage

```python
from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    fixed_form_vi,
    variational_free_energy,
)
import numpy as np

model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25)
grid = np.linspace(-6.0, 12.0, 2001)
res = fixed_form_vi(model, 7.0, grid, n_iter=2000)
print(res.belief)             # GaussianBelief(mu≈2.4, var≈0.05) — the exact posterior
print(res.final_free_energy)  # ≈ −log p(y): the bound is tight at the posterior
```
