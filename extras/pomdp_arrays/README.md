# POMDP Arrays

D, A, B, C, and E arrays as discrete generative-model components.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9, 10
- Sections: 9.1, B.10

## Scripts

- `visualize_pomdp_arrays.py` - Static visualization using shared library APIs.
- `simulate_pomdp_arrays.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_pomdp_arrays.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/pomdp_arrays/` and `output/data/extras/pomdp_arrays/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_pomdp_arrays.py --save` | `visualize_pomdp_arrays.png` | `visualize_pomdp_arrays.npz`, `visualize_pomdp_arrays.json` |
| `simulate_pomdp_arrays.py --save` | `simulate_pomdp_arrays.png` | `simulate_pomdp_arrays.npz`, `simulate_pomdp_arrays.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/pomdp_arrays/visualize_pomdp_arrays.py --save
uv run python extras/pomdp_arrays/simulate_pomdp_arrays.py --save
uv run python extras/pomdp_arrays/interactive_pomdp_arrays.py
```

