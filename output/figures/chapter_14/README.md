# `output/figures/chapter_14/` — Chapter 14 figures

PNG outputs from the Chapter 14 orchestrators (`chapters/chapter_14/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 14
```

Each saved artifact has a matching raw-data sidecar under
`output/data/chapter_14/`: compressed `NPZ` arrays plus a `JSON` manifest with
script provenance, figure path, array shape/dtype contracts, and summary
statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_14_1_ergodic_density.png` | `example_14_1_ergodic_density.py` |
| `example_14_4_bayesian_mechanics.png` | `example_14_4_bayesian_mechanics.py` |
