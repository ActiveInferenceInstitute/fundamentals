# Belief Propagation

Sum-product messages for state-space models.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 12
- Sections: 12.2, 12.3

## Scripts

- `visualize_belief_propagation.py` - Static visualization using shared library APIs.
- `simulate_belief_propagation.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_belief_propagation.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_belief_propagation.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/belief_propagation/` and `output/data/extras/belief_propagation/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_belief_propagation.py --save` | `visualize_belief_propagation.png` | `visualize_belief_propagation.npz`, `visualize_belief_propagation.json` |
| `simulate_belief_propagation.py --save` | `simulate_belief_propagation.png` | `simulate_belief_propagation.npz`, `simulate_belief_propagation.json` |
| `animation_belief_propagation.py --save` | `animation_belief_propagation.gif` | `animation_belief_propagation.npz`, `animation_belief_propagation.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/belief_propagation/visualize_belief_propagation.py --save
uv run python extras/belief_propagation/simulate_belief_propagation.py --save
uv run python extras/belief_propagation/interactive_belief_propagation.py
uv run python extras/belief_propagation/animation_belief_propagation.py --save
```

