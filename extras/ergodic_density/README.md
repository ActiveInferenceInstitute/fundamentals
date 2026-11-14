# Ergodic Density

Long-run occupancy as a density over viable states.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 14
- Sections: 14.1, 14.2

## Scripts

- `visualize_ergodic_density.py` - Static visualization using shared library APIs.
- `simulate_ergodic_density.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_ergodic_density.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_ergodic_density.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/ergodic_density/` and `output/data/extras/ergodic_density/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_ergodic_density.py --save` | `visualize_ergodic_density.png` | `visualize_ergodic_density.npz`, `visualize_ergodic_density.json` |
| `simulate_ergodic_density.py --save` | `simulate_ergodic_density.png` | `simulate_ergodic_density.npz`, `simulate_ergodic_density.json` |
| `animation_ergodic_density.py --save` | `animation_ergodic_density.gif` | `animation_ergodic_density.npz`, `animation_ergodic_density.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/ergodic_density/visualize_ergodic_density.py --save
uv run python extras/ergodic_density/simulate_ergodic_density.py --save
uv run python extras/ergodic_density/interactive_ergodic_density.py
uv run python extras/ergodic_density/animation_ergodic_density.py --save
```

