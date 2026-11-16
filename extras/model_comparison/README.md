# Model Comparison

Evidence and Bayes-factor style comparison across models.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.4, C.11.1

## Scripts

- `visualize_model_comparison.py` - Static visualization using shared library APIs.
- `simulate_model_comparison.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_model_comparison.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/model_comparison/` and `output/data/extras/model_comparison/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_model_comparison.py --save` | `visualize_model_comparison.png` | `visualize_model_comparison.npz`, `visualize_model_comparison.json` |
| `simulate_model_comparison.py --save` | `simulate_model_comparison.png` | `simulate_model_comparison.npz`, `simulate_model_comparison.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/model_comparison/visualize_model_comparison.py --save
uv run python extras/model_comparison/simulate_model_comparison.py --save
uv run python extras/model_comparison/interactive_model_comparison.py
```

