# Hybrid Generative Models

Continuous and discrete state-space components in one model.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11, 12
- Sections: 11.3, 12.6

## Scripts

- `visualize_hybrid_generative_models.py` - Static visualization using shared library APIs.
- `simulate_hybrid_generative_models.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_hybrid_generative_models.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/hybrid_generative_models/` and `output/data/extras/hybrid_generative_models/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_hybrid_generative_models.py --save` | `visualize_hybrid_generative_models.png` | `visualize_hybrid_generative_models.npz`, `visualize_hybrid_generative_models.json` |
| `simulate_hybrid_generative_models.py --save` | `simulate_hybrid_generative_models.png` | `simulate_hybrid_generative_models.npz`, `simulate_hybrid_generative_models.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/hybrid_generative_models/visualize_hybrid_generative_models.py --save
uv run python extras/hybrid_generative_models/simulate_hybrid_generative_models.py --save
uv run python extras/hybrid_generative_models/interactive_hybrid_generative_models.py
```

