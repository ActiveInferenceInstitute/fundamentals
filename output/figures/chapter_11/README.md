# `output/figures/chapter_11/` — Chapter 11 figures

PNG outputs from the Chapter 11 orchestrators (`chapters/chapter_11/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 11
```

Each saved artifact has a matching raw-data sidecar under
`output/data/chapter_11/`: compressed `NPZ` arrays plus a `JSON` manifest with
script provenance, figure path, array shape/dtype contracts, and summary
statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_11_1_free_energy_variants.png` | `example_11_1_free_energy_variants.py` |
| `example_11_2_sophisticated_planning.png` | `example_11_2_sophisticated_planning.py` |
