# Factorial Depth

Multiple state factors and observation modalities.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10
- Sections: 10.3, 12.6

## Scripts

- `visualize_factorial_depth.py` - Static visualization using shared library APIs.
- `simulate_factorial_depth.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_factorial_depth.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/factorial_depth/` and `output/data/extras/factorial_depth/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_factorial_depth.py --save` | `visualize_factorial_depth.png` | `visualize_factorial_depth.npz`, `visualize_factorial_depth.json` |
| `simulate_factorial_depth.py --save` | `simulate_factorial_depth.png` | `simulate_factorial_depth.npz`, `simulate_factorial_depth.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/factorial_depth/visualize_factorial_depth.py --save
uv run python extras/factorial_depth/simulate_factorial_depth.py --save
uv run python extras/factorial_depth/interactive_factorial_depth.py
```

