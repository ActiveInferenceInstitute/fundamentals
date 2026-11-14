# Predictive Coding

Prediction errors and recognition dynamics.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 5
- Sections: 5.1, 5.2

## Scripts

- `visualize_predictive_coding.py` - Static visualization using shared library APIs.
- `simulate_predictive_coding.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_predictive_coding.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_predictive_coding.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/predictive_coding/` and `output/data/extras/predictive_coding/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_predictive_coding.py --save` | `visualize_predictive_coding.png` | `visualize_predictive_coding.npz`, `visualize_predictive_coding.json` |
| `simulate_predictive_coding.py --save` | `simulate_predictive_coding.png` | `simulate_predictive_coding.npz`, `simulate_predictive_coding.json` |
| `animation_predictive_coding.py --save` | `animation_predictive_coding.gif` | `animation_predictive_coding.npz`, `animation_predictive_coding.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/predictive_coding/visualize_predictive_coding.py --save
uv run python extras/predictive_coding/simulate_predictive_coding.py --save
uv run python extras/predictive_coding/interactive_predictive_coding.py
uv run python extras/predictive_coding/animation_predictive_coding.py --save
```

