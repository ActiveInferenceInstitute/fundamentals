# Precision Weighting

Variance, precision, and their control over belief updates.

## Book Mapping

- Family: Foundations
- Chapters: 2, 5, 10
- Sections: 2.2, 5.1, 10.2

## Scripts

- `visualize_precision_weighting.py` - Static visualization using shared library APIs.
- `simulate_precision_weighting.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_precision_weighting.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/precision_weighting/` and `output/data/extras/precision_weighting/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_precision_weighting.py --save` | `visualize_precision_weighting.png` | `visualize_precision_weighting.npz`, `visualize_precision_weighting.json` |
| `simulate_precision_weighting.py --save` | `simulate_precision_weighting.png` | `simulate_precision_weighting.npz`, `simulate_precision_weighting.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/precision_weighting/visualize_precision_weighting.py --save
uv run python extras/precision_weighting/simulate_precision_weighting.py --save
uv run python extras/precision_weighting/interactive_precision_weighting.py
```

