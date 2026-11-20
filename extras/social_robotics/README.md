# Social Robotics

Belief updates over another agent's hidden intention.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 13
- Sections: 13.3

## Scripts

- `visualize_social_robotics.py` - Static visualization using shared library APIs.
- `simulate_social_robotics.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_social_robotics.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/social_robotics/` and `output/data/extras/social_robotics/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_social_robotics.py --save` | `visualize_social_robotics.png` | `visualize_social_robotics.npz`, `visualize_social_robotics.json` |
| `simulate_social_robotics.py --save` | `simulate_social_robotics.png` | `simulate_social_robotics.npz`, `simulate_social_robotics.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/social_robotics/visualize_social_robotics.py --save
uv run python extras/social_robotics/simulate_social_robotics.py --save
uv run python extras/social_robotics/interactive_social_robotics.py
```

