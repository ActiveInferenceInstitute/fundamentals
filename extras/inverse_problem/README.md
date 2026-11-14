# Inverse Problem

Recovering hidden states from observations under a generative model.

## Book Mapping

- Family: Foundations
- Chapters: 1, 2
- Sections: 1.4, 2.1.4

## Scripts

- `visualize_inverse_problem.py` - Static visualization using shared library APIs.
- `simulate_inverse_problem.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_inverse_problem.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/inverse_problem/` and `output/data/extras/inverse_problem/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_inverse_problem.py --save` | `visualize_inverse_problem.png` | `visualize_inverse_problem.npz`, `visualize_inverse_problem.json` |
| `simulate_inverse_problem.py --save` | `simulate_inverse_problem.png` | `simulate_inverse_problem.npz`, `simulate_inverse_problem.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/inverse_problem/visualize_inverse_problem.py --save
uv run python extras/inverse_problem/simulate_inverse_problem.py --save
uv run python extras/inverse_problem/interactive_inverse_problem.py
```

