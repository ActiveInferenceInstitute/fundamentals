# Mean-Field Variational Inference

Factorized approximations and coordinate updates.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.5, 4.6

## Scripts

- `visualize_mean_field_variational_inference.py` - Static visualization using shared library APIs.
- `simulate_mean_field_variational_inference.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_mean_field_variational_inference.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/mean_field_variational_inference/` and `output/data/extras/mean_field_variational_inference/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_mean_field_variational_inference.py --save` | `visualize_mean_field_variational_inference.png` | `visualize_mean_field_variational_inference.npz`, `visualize_mean_field_variational_inference.json` |
| `simulate_mean_field_variational_inference.py --save` | `simulate_mean_field_variational_inference.png` | `simulate_mean_field_variational_inference.npz`, `simulate_mean_field_variational_inference.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/mean_field_variational_inference/visualize_mean_field_variational_inference.py --save
uv run python extras/mean_field_variational_inference/simulate_mean_field_variational_inference.py --save
uv run python extras/mean_field_variational_inference/interactive_mean_field_variational_inference.py
```

