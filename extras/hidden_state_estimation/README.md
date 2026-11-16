# Hidden State Estimation

Sequential inference over latent states.

## Book Mapping

- Family: Foundations
- Chapters: 2
- Sections: 2.2, 2.3

## Scripts

- `visualize_hidden_state_estimation.py` - Static visualization using shared library APIs.
- `simulate_hidden_state_estimation.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_hidden_state_estimation.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_hidden_state_estimation.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/hidden_state_estimation/` and `output/data/extras/hidden_state_estimation/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_hidden_state_estimation.py --save` | `visualize_hidden_state_estimation.png` | `visualize_hidden_state_estimation.npz`, `visualize_hidden_state_estimation.json` |
| `simulate_hidden_state_estimation.py --save` | `simulate_hidden_state_estimation.png` | `simulate_hidden_state_estimation.npz`, `simulate_hidden_state_estimation.json` |
| `animation_hidden_state_estimation.py --save` | `animation_hidden_state_estimation.gif` | `animation_hidden_state_estimation.npz`, `animation_hidden_state_estimation.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/hidden_state_estimation/visualize_hidden_state_estimation.py --save
uv run python extras/hidden_state_estimation/simulate_hidden_state_estimation.py --save
uv run python extras/hidden_state_estimation/interactive_hidden_state_estimation.py
uv run python extras/hidden_state_estimation/animation_hidden_state_estimation.py --save
```

