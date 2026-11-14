# Generalized Coordinates

State, velocity, and higher-order embedding orders.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 6
- Sections: 6.3, 6.5, 6.6

## Scripts

- `visualize_generalized_coordinates.py` - Static visualization using shared library APIs.
- `simulate_generalized_coordinates.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_generalized_coordinates.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/generalized_coordinates/` and `output/data/extras/generalized_coordinates/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_generalized_coordinates.py --save` | `visualize_generalized_coordinates.png` | `visualize_generalized_coordinates.npz`, `visualize_generalized_coordinates.json` |
| `simulate_generalized_coordinates.py --save` | `simulate_generalized_coordinates.png` | `simulate_generalized_coordinates.npz`, `simulate_generalized_coordinates.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/generalized_coordinates/visualize_generalized_coordinates.py --save
uv run python extras/generalized_coordinates/simulate_generalized_coordinates.py --save
uv run python extras/generalized_coordinates/interactive_generalized_coordinates.py
```

