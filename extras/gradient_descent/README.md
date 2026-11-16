# Gradient Descent

Iterative descent on a differentiable loss surface.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 2, 3
- Sections: 2.5.2, 3.1

## Scripts

- `visualize_gradient_descent.py` - Static visualization using shared library APIs.
- `simulate_gradient_descent.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_gradient_descent.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_gradient_descent.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/gradient_descent/` and `output/data/extras/gradient_descent/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_gradient_descent.py --save` | `visualize_gradient_descent.png` | `visualize_gradient_descent.npz`, `visualize_gradient_descent.json` |
| `simulate_gradient_descent.py --save` | `simulate_gradient_descent.png` | `simulate_gradient_descent.npz`, `simulate_gradient_descent.json` |
| `animation_gradient_descent.py --save` | `animation_gradient_descent.gif` | `animation_gradient_descent.npz`, `animation_gradient_descent.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/gradient_descent/visualize_gradient_descent.py --save
uv run python extras/gradient_descent/simulate_gradient_descent.py --save
uv run python extras/gradient_descent/interactive_gradient_descent.py
uv run python extras/gradient_descent/animation_gradient_descent.py --save
```

