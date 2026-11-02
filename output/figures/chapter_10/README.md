# `output/figures/chapter_10/` — Chapter 10 figures

PNG outputs from the Chapter 10 orchestrators (`chapters/chapter_10/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 10
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_10/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

One default render of every Chapter 10 script — `python
scripts/run_all_figures.py --chapters 10` — writes these artifacts:

| File | Producing script |
|---|---|
| `example_10_1_learn_D.png` | `example_10_1_learn_D.py` |
| `example_10_2_learn_A.png` | `example_10_2_learn_A.py` |
| `example_10_3_learn_B.png` | `example_10_3_learn_B.py` |
| `example_10_4_novelty.png` | `example_10_4_novelty.py` |
| `example_10_5_precision.png` | `example_10_5_precision.py` |
| `example_10_6_precision_learning.png` | `example_10_6_precision_learning.py` |
| `example_10_7_bandit.png` | `example_10_7_two_armed_bandit.py` |
| `example_10_8_hierarchical.png` | `example_10_8_hierarchical.py` |
| `factorial_likelihood_structure.png` | `visualize_factorial_structure.py` |
| `animation_learning_A.gif` | `animation_learning.py` |
| `animation_precision.gif` | `animation_precision.py` |
| `animation_bandit.gif` | `animation_bandit.py` |

## Flag variants

These replace the default artifact when the flagged option is passed, so they
are not part of a default render:

| Flag | File | Producing script |
|---|---|---|
| `--explore` | `example_10_7_bandit_explore.png` | `example_10_7_two_armed_bandit.py` |
| `--explore` | `animation_bandit_explore.gif` | `animation_bandit.py` |
| `--strong` | `animation_precision_strong.gif` | `animation_precision.py` |
| `--transition` | `animation_learning_B.gif` | `animation_learning.py` |
