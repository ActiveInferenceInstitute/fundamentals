# Factor Graphs

Forney factor graphs as model diagrams for message passing.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 12
- Sections: 12.1, 12.5, B.12

## Scripts

- `visualize_factor_graphs.py` - Static visualization using shared library APIs.
- `simulate_factor_graphs.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_factor_graphs.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/factor_graphs/` and `output/data/extras/factor_graphs/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_factor_graphs.py --save` | `visualize_factor_graphs.png` | `visualize_factor_graphs.npz`, `visualize_factor_graphs.json` |
| `simulate_factor_graphs.py --save` | `simulate_factor_graphs.png` | `simulate_factor_graphs.npz`, `simulate_factor_graphs.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/factor_graphs/visualize_factor_graphs.py --save
uv run python extras/factor_graphs/simulate_factor_graphs.py --save
uv run python extras/factor_graphs/interactive_factor_graphs.py
```

