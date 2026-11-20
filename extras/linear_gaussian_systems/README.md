# Linear Gaussian Systems

State-space prediction and filtering under linear Gaussian assumptions.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.4

## Scripts

- `visualize_linear_gaussian_systems.py` - Static visualization using shared library APIs.
- `simulate_linear_gaussian_systems.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_linear_gaussian_systems.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_linear_gaussian_systems.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/linear_gaussian_systems/` and `output/data/extras/linear_gaussian_systems/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_linear_gaussian_systems.py --save` | `visualize_linear_gaussian_systems.png` | `visualize_linear_gaussian_systems.npz`, `visualize_linear_gaussian_systems.json` |
| `simulate_linear_gaussian_systems.py --save` | `simulate_linear_gaussian_systems.png` | `simulate_linear_gaussian_systems.npz`, `simulate_linear_gaussian_systems.json` |
| `animation_linear_gaussian_systems.py --save` | `animation_linear_gaussian_systems.gif` | `animation_linear_gaussian_systems.npz`, `animation_linear_gaussian_systems.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/linear_gaussian_systems/visualize_linear_gaussian_systems.py --save
uv run python extras/linear_gaussian_systems/simulate_linear_gaussian_systems.py --save
uv run python extras/linear_gaussian_systems/interactive_linear_gaussian_systems.py
uv run python extras/linear_gaussian_systems/animation_linear_gaussian_systems.py --save
```

