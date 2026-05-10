# `output/figures/chapter_02/` — Chapter 2 figures

PNG and GIF outputs from the Chapter 2 orchestrators
(`chapters/chapter_02/`). **Ephemeral and gitignored** — regenerate with:

```bash
./scripts/run_all_chapter_02.sh
```

## Expected files

| File | Producing script |
|---|---|
| `example_2_1_linear_deterministic.png` | `example_2_1_linear_deterministic.py` |
| `example_2_2_curve.png` / `example_2_2_posterior.png` | `example_2_2_linear_probabilistic.py` |
| `example_2_3_precision.png` | `example_2_3_precision.py` |
| `example_2_4_curve.png` / `example_2_4_posterior.png` | `example_2_4_nonlinear_deterministic.py` |
| `example_2_5_curve.png` / `example_2_5_posterior.png` | `example_2_5_nonlinear_probabilistic.py` |
| `example_2_6_imperfect_model.png` | `example_2_6_imperfect_model.py` |
| `example_2_7_ridge.png` / `example_2_7_posterior.png` / `example_2_7_convergence.png` | `example_2_7_multiple_samples.py` |
| `example_2_8_mle.png` | `example_2_8_mle_analytic.py` |
| `example_2_9_map.png` | `example_2_9_map_analytic.py` |
| `example_2_10_mle_descent.png` / `example_2_10_map_descent.png` / `example_2_10_comparison.png` | `example_2_10_gradient_descent.py` |
| `joint_uniform.png` / `joint_gaussian.png` / `joint_surface.png` | `visualize_generative_model.py` |
| `animation_sequential.gif` | `animation_sequential.py` |
| `animation_gradient_descent.gif` | `animation_gradient_descent.py` |
