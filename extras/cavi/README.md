# CAVI

Coordinate-ascent updates as repeated local message refinement.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4, 12
- Sections: 4.5, 12.4

## Scripts

- `visualize_cavi.py` - Static visualization using shared library APIs.
- `simulate_cavi.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_cavi.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_cavi.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/cavi/` and `output/data/extras/cavi/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_cavi.py --save` | `visualize_cavi.png` | `visualize_cavi.npz`, `visualize_cavi.json` |
| `simulate_cavi.py --save` | `simulate_cavi.png` | `simulate_cavi.npz`, `simulate_cavi.json` |
| `animation_cavi.py --save` | `animation_cavi.gif` | `animation_cavi.npz`, `animation_cavi.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/cavi/visualize_cavi.py --save
uv run python extras/cavi/simulate_cavi.py --save
uv run python extras/cavi/interactive_cavi.py
uv run python extras/cavi/animation_cavi.py --save
```

