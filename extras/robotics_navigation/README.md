# Robotics Navigation

Navigation and control as preference-seeking active inference.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 13
- Sections: 13.1, 13.2

## Scripts

- `visualize_robotics_navigation.py` - Static visualization using shared library APIs.
- `simulate_robotics_navigation.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_robotics_navigation.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_robotics_navigation.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/robotics_navigation/` and `output/data/extras/robotics_navigation/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_robotics_navigation.py --save` | `visualize_robotics_navigation.png` | `visualize_robotics_navigation.npz`, `visualize_robotics_navigation.json` |
| `simulate_robotics_navigation.py --save` | `simulate_robotics_navigation.png` | `simulate_robotics_navigation.npz`, `simulate_robotics_navigation.json` |
| `animation_robotics_navigation.py --save` | `animation_robotics_navigation.gif` | `animation_robotics_navigation.npz`, `animation_robotics_navigation.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/robotics_navigation/visualize_robotics_navigation.py --save
uv run python extras/robotics_navigation/simulate_robotics_navigation.py --save
uv run python extras/robotics_navigation/interactive_robotics_navigation.py
uv run python extras/robotics_navigation/animation_robotics_navigation.py --save
```

