# Backward Smoothing

Backward messages that refine earlier state beliefs.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11, 12
- Sections: 11.2.9, 12.3

## Scripts

- `visualize_backward_smoothing.py` - Static visualization using shared library APIs.
- `simulate_backward_smoothing.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_backward_smoothing.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_backward_smoothing.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/backward_smoothing/` and `output/data/extras/backward_smoothing/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_backward_smoothing.py --save` | `visualize_backward_smoothing.png` | `visualize_backward_smoothing.npz`, `visualize_backward_smoothing.json` |
| `simulate_backward_smoothing.py --save` | `simulate_backward_smoothing.png` | `simulate_backward_smoothing.npz`, `simulate_backward_smoothing.json` |
| `animation_backward_smoothing.py --save` | `animation_backward_smoothing.gif` | `animation_backward_smoothing.npz`, `animation_backward_smoothing.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/backward_smoothing/visualize_backward_smoothing.py --save
uv run python extras/backward_smoothing/simulate_backward_smoothing.py --save
uv run python extras/backward_smoothing/interactive_backward_smoothing.py
uv run python extras/backward_smoothing/animation_backward_smoothing.py --save
```

