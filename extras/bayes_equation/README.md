# Bayes Equation

Prior, likelihood, evidence, posterior, and normalization.

## Book Mapping

- Family: Foundations
- Chapters: 1, 2
- Sections: 1.3, 2.1.4, C.2.4

## Scripts

- `visualize_bayes_equation.py` - Static visualization using shared library APIs.
- `simulate_bayes_equation.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_bayes_equation.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/bayes_equation/` and `output/data/extras/bayes_equation/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_bayes_equation.py --save` | `visualize_bayes_equation.png` | `visualize_bayes_equation.npz`, `visualize_bayes_equation.json` |
| `simulate_bayes_equation.py --save` | `simulate_bayes_equation.png` | `simulate_bayes_equation.npz`, `simulate_bayes_equation.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/bayes_equation/visualize_bayes_equation.py --save
uv run python extras/bayes_equation/simulate_bayes_equation.py --save
uv run python extras/bayes_equation/interactive_bayes_equation.py
```

