# `output/figures/chapter_01/` — Chapter 1 figures

PNG and GIF outputs from the Chapter 1 orchestrators (`chapters/chapter_01/`).
**Ephemeral and gitignored** — regenerate with:

```bash
./scripts/run_all_chapter_01.sh
```

Each saved artifact has a matching raw-data sidecar under `output/data/chapter_01/`: compressed `NPZ` arrays plus a `JSON` manifest with script provenance, figure path, array shape/dtype contracts, and summary statistics.

## Expected files

| File | Producing script |
|---|---|
| `01_box_scenario_generator.png` | `01_box_scenario.py` |
| `01_box_scenario_stream.png` | `01_box_scenario.py` |
| `02_three_perspectives.png` | `02_three_perspectives.py` |
| `03_bayes_three_panel.png` | `03_bayes_intuition.py` |
| `03_bayes_overlay.png` | `03_bayes_intuition.py` |
| `03_bayes_evidence.png` | `03_bayes_intuition.py` |
| `04_inverse_curve.png` | `04_inverse_problem.py` |
| `04_inverse_posterior.png` | `04_inverse_problem.py` |
| `04_inverse_overlay.png` | `04_inverse_problem.py` |
| `05_belief_from_stream.gif` | `05_belief_from_stream.py` |
