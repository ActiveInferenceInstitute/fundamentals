# Multivariate Gaussians

Covariance geometry, entropy, and KL in multiple dimensions.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3, 6
- Sections: 3.4, C.10

## Scripts

- `visualize_multivariate_gaussians.py` - Static visualization using shared library APIs.
- `simulate_multivariate_gaussians.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_multivariate_gaussians.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/multivariate_gaussians/` and `output/data/extras/multivariate_gaussians/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_multivariate_gaussians.py --save` | `visualize_multivariate_gaussians.png` | `visualize_multivariate_gaussians.npz`, `visualize_multivariate_gaussians.json` |
| `simulate_multivariate_gaussians.py --save` | `simulate_multivariate_gaussians.png` | `simulate_multivariate_gaussians.npz`, `simulate_multivariate_gaussians.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/multivariate_gaussians/visualize_multivariate_gaussians.py --save
uv run python extras/multivariate_gaussians/simulate_multivariate_gaussians.py --save
uv run python extras/multivariate_gaussians/interactive_multivariate_gaussians.py
```

