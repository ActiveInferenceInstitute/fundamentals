# Gridworld Control

Planning as inference in controllable grid-world transitions.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.4, 9.5

## Scripts

- `visualize_gridworld_control.py` - Static visualization using shared library APIs.
- `simulate_gridworld_control.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_gridworld_control.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_gridworld_control.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/gridworld_control/` and `output/data/extras/gridworld_control/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_gridworld_control.py --save` | `visualize_gridworld_control.png` | `visualize_gridworld_control.npz`, `visualize_gridworld_control.json` |
| `simulate_gridworld_control.py --save` | `simulate_gridworld_control.png` | `simulate_gridworld_control.npz`, `simulate_gridworld_control.json` |
| `animation_gridworld_control.py --save` | `animation_gridworld_control.gif` | `animation_gridworld_control.npz`, `animation_gridworld_control.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/gridworld_control/visualize_gridworld_control.py --save
uv run python extras/gridworld_control/simulate_gridworld_control.py --save
uv run python extras/gridworld_control/interactive_gridworld_control.py
uv run python extras/gridworld_control/animation_gridworld_control.py --save
```

