# Parameter Uncertainty

Forgetting rates and uncertainty on learned parameters.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.6, 11.2.7

## Scripts

- `visualize_parameter_uncertainty.py` - Static visualization using shared library APIs.
- `simulate_parameter_uncertainty.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_parameter_uncertainty.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/parameter_uncertainty/` and `output/data/extras/parameter_uncertainty/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_parameter_uncertainty.py --save` | `visualize_parameter_uncertainty.png` | `visualize_parameter_uncertainty.npz`, `visualize_parameter_uncertainty.json` |
| `simulate_parameter_uncertainty.py --save` | `simulate_parameter_uncertainty.png` | `simulate_parameter_uncertainty.npz`, `simulate_parameter_uncertainty.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/parameter_uncertainty/visualize_parameter_uncertainty.py --save
uv run python extras/parameter_uncertainty/simulate_parameter_uncertainty.py --save
uv run python extras/parameter_uncertainty/interactive_parameter_uncertainty.py
```

