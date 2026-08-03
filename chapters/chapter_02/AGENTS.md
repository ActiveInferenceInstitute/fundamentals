# AGENTS.md — Chapter 02: Hidden State Estimation

## Overview

Chapter 2 turns the conceptual ideas of Chapter 1 into the canonical
linear-Gaussian Bayesian inference recipe and walks through ten numbered
examples (2.1–2.10) exploring variations of it.

Directory: `/chapters/chapter_02/`

## Scripts

| File | Book Section | Key Concept |
|---|---|---|
| `example_2_1_linear_deterministic.py` | Example 2.1 | Bayesian inversion of a noiseless linear sensor |
| `example_2_2_linear_probabilistic.py` | Example 2.2 | Standard Gaussian likelihood × Gaussian prior |
| `example_2_3_precision.py` | Example 2.3 | Sweep prior vs likelihood precision |
| `example_2_4_nonlinear_deterministic.py` | Example 2.4 | Quadratic generator → bi-modal posterior |
| `example_2_5_nonlinear_probabilistic.py` | Example 2.5 | Nonlinear generator with Gaussian noise |
| `example_2_6_imperfect_model.py` | Example 2.6 | Mismatch between generative process and model |
| `example_2_7_multiple_samples.py` | Example 2.7 | Sequential vs batch inference over N i.i.d. samples |
| `example_2_8_mle_analytic.py` | Example 2.8 | Closed-form MLE compared to grid-Bayesian mode |
| `example_2_9_map_analytic.py` | Example 2.9 | Closed-form MAP and prior-strength sweep |
| `example_2_10_gradient_descent.py` | §2.5.2 | Iterative MLE/MAP via gradient descent |
| `visualize_generative_model.py` | §2.4 | Joint density `p(x, y)` as heatmap + 3-D surface |
| `interactive_explorer.py` | bonus | Slider-driven exploration (two modes: `full`, `precision`) |
| `interactive_gradient_descent.py` | §2.5.2 | **Interactive** (GUI / web-launchable): `log10(learning rate)` / `iteration` sliders scrub the gradient-descent trajectory on the MLE loss surface — readout reports the iterate, loss, step size, and converging/diverging status |
| `animation_sequential.py` | bonus | GIF: posterior collapsing as i.i.d. samples arrive |
| `animation_gradient_descent.py` | bonus | GIF: MLE/MAP iterate rolling down the loss curve |

Common CLI flags across non-interactive scripts: `--save` (headless); scripts
that sample data also accept `--seed`. Some scripts accept `--x-true`,
`--y-obs`, `--n-samples`, `--n` (the two `animation_*.py` scripts — number of
frames/grid points, not samples), `--epsilon2`, `--lr`, `--max-iter` — see
individual `--help` output. `interactive_explorer.py` is the one script with
a behavior-changing flag: `--mode {full,precision}` (default `full`) switches
between the four-slider explorer and the single precision-ratio slider.

## Running

```bash
# Single script
python chapters/chapter_02/example_2_3_precision.py --save

# Interactive slider explorer
python chapters/chapter_02/interactive_explorer.py            # full mode
python chapters/chapter_02/interactive_explorer.py --mode precision

# Interactive gradient-descent trajectory scrubber
python chapters/chapter_02/interactive_gradient_descent.py

# Batch render all chapter 2 figures (excludes interactive_explorer.py and interactive_gradient_descent.py)
python scripts/run_all_figures.py --chapters 2
# or:
./scripts/run_all_chapter_02.sh
```

## Imports

These scripts import exclusively from:
- `active_inference` — all public API classes and functions
- `active_inference.estimators.mle` / `.map` — for scripts using raw gradient functions
- `active_inference.utils.io` — `default_figure_dir`, `ensure_dir`
- Standard library plus `numpy` / `matplotlib` for CLI and rendering glue

## Testing

Smoke tests in `tests/chapters/test_smoke.py` run every non-interactive
script in this folder (currently 10 numbered examples, 1 visualization,
and 2 animations = 13 files) with `--save` under `MPLBACKEND=Agg` and
assert exit code 0. Discovery is glob-driven — adding a new file in this
folder automatically wires it into the smoke suite, the
`scripts/run_all_figures.py` batch runner, and the `run.sh` text/web
menus.

## Notes

- The `interactive_explorer.py` and `interactive_gradient_descent.py` scripts
  are **not** run in smoke tests because they block on `plt.show()`.
- Example 2.7 includes a `sequential_inference` helper function (defined in the
  script itself) that manually reproduces the batch result step-by-step,
  demonstrating exchangeability.
- Example 2.10 runs both MLE and MAP gradient descent and produces a combined
  comparison plot plus individual loss/iterate panels.