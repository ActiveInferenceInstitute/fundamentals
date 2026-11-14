# Active Generalized Filtering

Action and perception as coupled free-energy descent.

## Book Mapping

- Family: Active Inference Core
- Chapters: 7
- Sections: 7.2, 7.4, 7.5

## Scripts

- `visualize_active_generalized_filtering.py` - Static visualization using shared library APIs.
- `simulate_active_generalized_filtering.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_active_generalized_filtering.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_active_generalized_filtering.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/active_generalized_filtering/` and `output/data/extras/active_generalized_filtering/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_active_generalized_filtering.py --save` | `visualize_active_generalized_filtering.png` | `visualize_active_generalized_filtering.npz`, `visualize_active_generalized_filtering.json` |
| `simulate_active_generalized_filtering.py --save` | `simulate_active_generalized_filtering.png` | `simulate_active_generalized_filtering.npz`, `simulate_active_generalized_filtering.json` |
| `animation_active_generalized_filtering.py --save` | `animation_active_generalized_filtering.gif` | `animation_active_generalized_filtering.npz`, `animation_active_generalized_filtering.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/active_generalized_filtering/visualize_active_generalized_filtering.py --save
uv run python extras/active_generalized_filtering/simulate_active_generalized_filtering.py --save
uv run python extras/active_generalized_filtering/interactive_active_generalized_filtering.py
uv run python extras/active_generalized_filtering/animation_active_generalized_filtering.py --save
```

