# `output/figures/chapter_05/` — Chapter 5 figures

PNG and GIF outputs from the Chapter 5 orchestrators
(`chapters/chapter_05/`). **Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 5
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_05/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

One default render of every Chapter 5 script — `python
scripts/run_all_figures.py --chapters 5` — writes these artifacts:

| File | Producing script |
|---|---|
| `example_5_1_mle_free_energy.png` / `example_5_1_prediction_errors.png` | `example_5_1_prediction_errors.py` |
| `example_5_2_precision.png` | `example_5_2_precision.py` |
| `example_5_3_multivariate.png` | `example_5_3_multivariate.py` |
| `example_5_4_recognition_nonlinear.png` | `example_5_4_recognition_dynamics.py` |
| `example_5_6_parameterized_recover.png` | `example_5_6_parameterized.py` |
| `example_5_7_hierarchical.png` | `example_5_7_hierarchical.py` |
| `animation_recognition_linear.gif` | `animation_recognition_descent.py` |
| `animation_hierarchical.gif` | `animation_hierarchical.py` |

## Flag variants

These replace the default artifact when the flagged option is passed, so they
are not part of a default render:

| Flag | File | Producing script |
|---|---|---|
| `--linear` | `example_5_4_recognition_linear.png` | `example_5_4_recognition_dynamics.py` |
| `--regime informative` | `example_5_6_parameterized_informative.png` | `example_5_6_parameterized.py` |
| `--nonlinear` | `animation_recognition_nonlinear.gif` | `animation_recognition_descent.py` |

> `run_all_figures.py` runs `example_5_4` without `--linear`, producing the
> *nonlinear* PNG; pass `--linear` manually for the cross-chapter oracle figure.
