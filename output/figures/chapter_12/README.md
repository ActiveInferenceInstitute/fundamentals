# `output/figures/chapter_12/` — Chapter 12 figures

PNG outputs from the Chapter 12 orchestrators (`chapters/chapter_12/`).
**Ephemeral and gitignored** — regenerate with:

```bash
python scripts/run_all_figures.py --chapters 12
```

Each saved artifact has a matching raw-data sidecar under
`output/data/chapter_12/`: compressed `NPZ` arrays plus a `JSON` manifest with
script provenance, figure path, array shape/dtype contracts, and summary
statistics.

## Expected files

| File | Producing script |
|---|---|
| `example_12_1_factor_graph_messages.png` | `example_12_1_factor_graph_messages.py` |
| `example_12_4_hybrid_message_bridge.png` | `example_12_4_hybrid_message_bridge.py` |
