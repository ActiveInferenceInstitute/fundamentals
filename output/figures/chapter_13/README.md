# `output/figures/chapter_13/` — Chapter 13 figures

PNG outputs from the Chapter 13 orchestrators (`chapters/chapter_13/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 13
```

Each saved artifact has a matching raw-data sidecar under
`output/data/chapter_13/`: compressed `NPZ` arrays plus a `JSON` manifest with
script provenance, figure path, array shape/dtype contracts, and summary
statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_13_1_robotics_navigation.png` | `example_13_1_robotics_navigation.py` |
| `example_13_3_social_robotics.png` | `example_13_3_social_robotics.py` |
