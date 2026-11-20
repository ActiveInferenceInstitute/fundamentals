# Linear Regression

Deterministic parameter estimation and residual geometry.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.1, 3.2

## Scripts

- `visualize_linear_regression.py` - Static visualization using shared library APIs.
- `simulate_linear_regression.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_linear_regression.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/linear_regression/` and `output/data/extras/linear_regression/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_linear_regression.py --save` | `visualize_linear_regression.png` | `visualize_linear_regression.npz`, `visualize_linear_regression.json` |
| `simulate_linear_regression.py --save` | `simulate_linear_regression.png` | `simulate_linear_regression.npz`, `simulate_linear_regression.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/linear_regression/visualize_linear_regression.py --save
uv run python extras/linear_regression/simulate_linear_regression.py --save
uv run python extras/linear_regression/interactive_linear_regression.py
```

