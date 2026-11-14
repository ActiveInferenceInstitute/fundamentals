# Generalized Filtering

Dynamic state inference with generalized filtering.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 6
- Sections: 6.1, 6.2

## Scripts

- `visualize_generalized_filtering.py` - Static visualization using shared library APIs.
- `simulate_generalized_filtering.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_generalized_filtering.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_generalized_filtering.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/generalized_filtering/` and `output/data/extras/generalized_filtering/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_generalized_filtering.py --save` | `visualize_generalized_filtering.png` | `visualize_generalized_filtering.npz`, `visualize_generalized_filtering.json` |
| `simulate_generalized_filtering.py --save` | `simulate_generalized_filtering.png` | `simulate_generalized_filtering.npz`, `simulate_generalized_filtering.json` |
| `animation_generalized_filtering.py --save` | `animation_generalized_filtering.gif` | `animation_generalized_filtering.npz`, `animation_generalized_filtering.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/generalized_filtering/visualize_generalized_filtering.py --save
uv run python extras/generalized_filtering/simulate_generalized_filtering.py --save
uv run python extras/generalized_filtering/interactive_generalized_filtering.py
uv run python extras/generalized_filtering/animation_generalized_filtering.py --save
```

