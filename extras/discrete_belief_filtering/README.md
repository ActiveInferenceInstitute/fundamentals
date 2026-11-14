# Discrete Belief Filtering

Dynamic categorical belief updates over time.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.2

## Scripts

- `visualize_discrete_belief_filtering.py` - Static visualization using shared library APIs.
- `simulate_discrete_belief_filtering.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_discrete_belief_filtering.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_discrete_belief_filtering.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/discrete_belief_filtering/` and `output/data/extras/discrete_belief_filtering/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_discrete_belief_filtering.py --save` | `visualize_discrete_belief_filtering.png` | `visualize_discrete_belief_filtering.npz`, `visualize_discrete_belief_filtering.json` |
| `simulate_discrete_belief_filtering.py --save` | `simulate_discrete_belief_filtering.png` | `simulate_discrete_belief_filtering.npz`, `simulate_discrete_belief_filtering.json` |
| `animation_discrete_belief_filtering.py --save` | `animation_discrete_belief_filtering.gif` | `animation_discrete_belief_filtering.npz`, `animation_discrete_belief_filtering.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/discrete_belief_filtering/visualize_discrete_belief_filtering.py --save
uv run python extras/discrete_belief_filtering/simulate_discrete_belief_filtering.py --save
uv run python extras/discrete_belief_filtering/interactive_discrete_belief_filtering.py
uv run python extras/discrete_belief_filtering/animation_discrete_belief_filtering.py --save
```

