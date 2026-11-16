# Expectation Maximization

Alternating latent expectation and parameter maximization.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.5

## Scripts

- `visualize_expectation_maximization.py` - Static visualization using shared library APIs.
- `simulate_expectation_maximization.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_expectation_maximization.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_expectation_maximization.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/expectation_maximization/` and `output/data/extras/expectation_maximization/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_expectation_maximization.py --save` | `visualize_expectation_maximization.png` | `visualize_expectation_maximization.npz`, `visualize_expectation_maximization.json` |
| `simulate_expectation_maximization.py --save` | `simulate_expectation_maximization.png` | `simulate_expectation_maximization.npz`, `simulate_expectation_maximization.json` |
| `animation_expectation_maximization.py --save` | `animation_expectation_maximization.gif` | `animation_expectation_maximization.npz`, `animation_expectation_maximization.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/expectation_maximization/visualize_expectation_maximization.py --save
uv run python extras/expectation_maximization/simulate_expectation_maximization.py --save
uv run python extras/expectation_maximization/interactive_expectation_maximization.py
uv run python extras/expectation_maximization/animation_expectation_maximization.py --save
```

