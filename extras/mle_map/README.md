# MLE And MAP

Likelihood-only and prior-regularized point estimates.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 2
- Sections: 2.5.1

## Scripts

- `visualize_mle_map.py` - Static visualization using shared library APIs.
- `simulate_mle_map.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_mle_map.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/mle_map/` and `output/data/extras/mle_map/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_mle_map.py --save` | `visualize_mle_map.png` | `visualize_mle_map.npz`, `visualize_mle_map.json` |
| `simulate_mle_map.py --save` | `simulate_mle_map.png` | `simulate_mle_map.npz`, `simulate_mle_map.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/mle_map/visualize_mle_map.py --save
uv run python extras/mle_map/simulate_mle_map.py --save
uv run python extras/mle_map/interactive_mle_map.py
```

