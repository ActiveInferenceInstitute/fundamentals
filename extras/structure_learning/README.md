# Structure Learning

Comparing candidate model structures through evidence-like scores.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.5

## Scripts

- `visualize_structure_learning.py` - Static visualization using shared library APIs.
- `simulate_structure_learning.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_structure_learning.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/structure_learning/` and `output/data/extras/structure_learning/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_structure_learning.py --save` | `visualize_structure_learning.png` | `visualize_structure_learning.npz`, `visualize_structure_learning.json` |
| `simulate_structure_learning.py --save` | `simulate_structure_learning.png` | `simulate_structure_learning.npz`, `simulate_structure_learning.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/structure_learning/visualize_structure_learning.py --save
uv run python extras/structure_learning/simulate_structure_learning.py --save
uv run python extras/structure_learning/interactive_structure_learning.py
```

