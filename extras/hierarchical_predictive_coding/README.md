# Hierarchical Predictive Coding

Layered prediction-error propagation.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 5, 8
- Sections: 5.4, 8.3

## Scripts

- `visualize_hierarchical_predictive_coding.py` - Static visualization using shared library APIs.
- `simulate_hierarchical_predictive_coding.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_hierarchical_predictive_coding.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_hierarchical_predictive_coding.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/hierarchical_predictive_coding/` and `output/data/extras/hierarchical_predictive_coding/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_hierarchical_predictive_coding.py --save` | `visualize_hierarchical_predictive_coding.png` | `visualize_hierarchical_predictive_coding.npz`, `visualize_hierarchical_predictive_coding.json` |
| `simulate_hierarchical_predictive_coding.py --save` | `simulate_hierarchical_predictive_coding.png` | `simulate_hierarchical_predictive_coding.npz`, `simulate_hierarchical_predictive_coding.json` |
| `animation_hierarchical_predictive_coding.py --save` | `animation_hierarchical_predictive_coding.gif` | `animation_hierarchical_predictive_coding.npz`, `animation_hierarchical_predictive_coding.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/hierarchical_predictive_coding/visualize_hierarchical_predictive_coding.py --save
uv run python extras/hierarchical_predictive_coding/simulate_hierarchical_predictive_coding.py --save
uv run python extras/hierarchical_predictive_coding/interactive_hierarchical_predictive_coding.py
uv run python extras/hierarchical_predictive_coding/animation_hierarchical_predictive_coding.py --save
```

