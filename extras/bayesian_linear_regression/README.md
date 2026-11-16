# Bayesian Linear Regression

Posterior parameter uncertainty and predictive bands.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.3

## Scripts

- `visualize_bayesian_linear_regression.py` - Static visualization using shared library APIs.
- `simulate_bayesian_linear_regression.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_bayesian_linear_regression.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_bayesian_linear_regression.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/bayesian_linear_regression/` and `output/data/extras/bayesian_linear_regression/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_bayesian_linear_regression.py --save` | `visualize_bayesian_linear_regression.png` | `visualize_bayesian_linear_regression.npz`, `visualize_bayesian_linear_regression.json` |
| `simulate_bayesian_linear_regression.py --save` | `simulate_bayesian_linear_regression.png` | `simulate_bayesian_linear_regression.npz`, `simulate_bayesian_linear_regression.json` |
| `animation_bayesian_linear_regression.py --save` | `animation_bayesian_linear_regression.gif` | `animation_bayesian_linear_regression.npz`, `animation_bayesian_linear_regression.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/bayesian_linear_regression/visualize_bayesian_linear_regression.py --save
uv run python extras/bayesian_linear_regression/simulate_bayesian_linear_regression.py --save
uv run python extras/bayesian_linear_regression/interactive_bayesian_linear_regression.py
uv run python extras/bayesian_linear_regression/animation_bayesian_linear_regression.py --save
```

