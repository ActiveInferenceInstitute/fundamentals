# Sophisticated Inference

Planning with beliefs over future belief updates.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.1

## Scripts

- `visualize_sophisticated_inference.py` - Static visualization using shared library APIs.
- `simulate_sophisticated_inference.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_sophisticated_inference.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/sophisticated_inference/` and `output/data/extras/sophisticated_inference/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_sophisticated_inference.py --save` | `visualize_sophisticated_inference.png` | `visualize_sophisticated_inference.npz`, `visualize_sophisticated_inference.json` |
| `simulate_sophisticated_inference.py --save` | `simulate_sophisticated_inference.png` | `simulate_sophisticated_inference.npz`, `simulate_sophisticated_inference.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/sophisticated_inference/visualize_sophisticated_inference.py --save
uv run python extras/sophisticated_inference/simulate_sophisticated_inference.py --save
uv run python extras/sophisticated_inference/interactive_sophisticated_inference.py
```

