# `chapters/chapter_08/` — Learning, attention, and hierarchy

Chapter 8 extends continuous-state active inference with first-order parameter
learning, second-order precision learning, hierarchical generative models, and
message passing.

## Scripts

| Script | Section | Role |
|---|---:|---|
| `example_8_1_learning_attention.py` | §8.1 | Triple estimation of `mu_x`, `mu_theta`, and log precision `mu_zeta`. |
| `example_8_2_hierarchical_continuous.py` | §8.2–§8.4 | Two-layer hierarchy with an upper contextual prior over the lower state. |
| `visualize_message_passing.py` | §8.5 | Diagram of forward sensory messages, bottom-up errors, top-down predictions, and backward errors. |
| `animation_learning_attention.py` | §8.1 | GIF of state, parameter, precision, variance, and VFE convergence. |

## Rules

- Scripts import only from `active_inference`, `numpy`, `matplotlib`, and the
  Python standard library.
- Every script accepts `--save`; stochastic scripts also accept `--seed`.
- Save static figures to `output/figures/chapter_08/`; save animations as GIFs
  through `active_inference.visualizations.save_animation`.
- Keep reusable logic in `src/active_inference/core/continuous_learning.py`,
  `estimators/continuous_learning.py`, or `visualizations/unified.py`.

## Educational checks

- Show the time-scale separation explicitly: hidden-state updates should move
  faster than parameter and precision learning.
- Label log precision and variance together; readers need to see why optimizing
  `zeta` cannot make a negative variance.
- In hierarchy figures, keep prediction nodes, error nodes, and messages visually
  distinct.

## Verification

```bash
python scripts/run_all_figures.py --chapters 8
uv run pytest tests/chapters -v
```

`tests/chapters/test_smoke.py` runs every discovered chapter script (including
all four Chapter 8 orchestrators) with `--save` in the single parametrized
`test_chapter_script_runs_and_exports_raw_data`.
