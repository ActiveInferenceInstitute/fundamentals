# `chapters/chapter_04/` — Variational Bayesian Inference

Chapter 4 scripts turn posterior estimation into optimization by minimizing
**variational free energy** (VFE). Three algorithms minimize the same objective:
zero-order coordinate search, free-form mean-field CAVI, and fixed-form gradient
VI. All run on the linear-Gaussian example whose exact grid posterior
`N(2.4, 0.05)` and log-evidence serve as the verification oracle.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_4_1_coordinate_search.py`](example_4_1_coordinate_search.py) | ~95 | Zero-order coordinate search descends the VFE surface (Fig. 4.2.2). |
| [`example_4_2_surprisal.py`](example_4_2_surprisal.py) | ~90 | Surprisal `−log p(y)` vs `y` and vs `p(y)` (Fig. 4.3.1). |
| [`example_4_3_vfe_forms.py`](example_4_3_vfe_forms.py) | ~85 | All four VFE forms (G/D/C/E) agree; G/C/E decompositions over a descent. |
| [`example_4_6_free_form_cavi.py`](example_4_6_free_form_cavi.py) | ~85 | Mean-field CAVI on `(x, β₀, β₁)`; VFE monotone, closed-form updates. |
| [`example_4_7_fixed_form.py`](example_4_7_fixed_form.py) | ~100 | Fixed-form gradient VI → exact posterior `N(2.4, 0.05)`, bound tight. |
| [`animation_vfe_descent.py`](animation_vfe_descent.py) | ~65 | GIF: `q(x)` tightening onto the posterior as VFE falls to the bound. |
| [`visualize_kl_loss.py`](visualize_kl_loss.py) | ~95 | KL loss surface vs VFE surface (same minimum). |
| [`visualize_vfe_intuition.py`](visualize_vfe_intuition.py) | ~80 | G-form intuition: `q(x)`, `p(x, y)`, posterior. |
| [`visualize_model_comparison.py`](visualize_model_comparison.py) | ~95 | Model evidence of a good vs bad model. |
| [`interactive_vfe_explorer.py`](interactive_vfe_explorer.py) | ~30 | **Interactive** (GUI / web-launchable): `μ_x` / `σ_x²` sliders drag `q(x)` onto the posterior while a live bar chart tracks the VFE decomposition (free energy, divergence, complexity, accuracy). |

## Running

```bash
# Single script (headless — saves to output/figures/chapter_04/)
python chapters/chapter_04/example_4_7_fixed_form.py --save

# Every Chapter 4 script at once
python scripts/run_all_figures.py --chapters 4
```

Each script accepts `--save` for headless rendering (except
`interactive_vfe_explorer.py`, which always opens a GUI window).

## Library Usage

```python
from active_inference import (
    GaussianBelief, LinearGaussianModel, GridBayesianInference,
    variational_free_energy, vfe_g_form, vfe_d_form, vfe_c_form,
    coordinate_search_vfe, fixed_form_vi, free_form_cavi, MeanFieldConfig,
    log_model_evidence, surprisal,
)
```

`interactive_vfe_explorer.py` is a thin wrapper around
`active_inference.visualizations.interactive_variational_free_energy`.

## Smoke Tests

`tests/chapters/test_smoke.py` runs every discovered chapter script via `subprocess`
with `--save` and asserts exit code 0, in the single parametrized
`test_chapter_script_runs_and_exports_raw_data` (one case per script, including
every Chapter 4 orchestrator). `interactive_vfe_explorer.py` has no `--save`
path and is exercised separately by `tests/visualizations/test_interactive.py`.
Unit tests for the VFE machinery live in `tests/core/test_variational.py` and
`tests/estimators/test_variational.py`.

## Key Concepts

- **One objective, five forms.** `variational_free_energy(...)` computes the four
  full-`q` decompositions at once (generative, divergence, complexity, energy);
  `VFEComponents.check()` asserts they agree to grid precision and that the
  divergence is non-negative (Gibbs). The remaining two forms, MAP and MLE, are
  point-mass special cases reached separately via `vfe_map_form`/`vfe_mle_form`
  (a scalar `mu`, not a full `q`), not part of the same `VFEComponents` trace.
- **`𝓕 ≥ −log p(y)`** for any `q`, tight exactly at the posterior — the oracle that
  anchors every example here.
- **No PyTorch.** The book used autodiff for fixed-form VI; this companion stays
  dependency-light with central finite differences — same method, frozen deps.
