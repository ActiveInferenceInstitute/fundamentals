# Dirichlet Learning

Pseudocount accumulation for POMDP parameter learning.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10
- Sections: 10.1

## Scripts

- `visualize_dirichlet_learning.py` - Static visualization using shared library APIs.
- `simulate_dirichlet_learning.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_dirichlet_learning.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_dirichlet_learning.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/dirichlet_learning/` and `output/data/extras/dirichlet_learning/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_dirichlet_learning.py --save` | `visualize_dirichlet_learning.png` | `visualize_dirichlet_learning.npz`, `visualize_dirichlet_learning.json` |
| `simulate_dirichlet_learning.py --save` | `simulate_dirichlet_learning.png` | `simulate_dirichlet_learning.npz`, `simulate_dirichlet_learning.json` |
| `animation_dirichlet_learning.py --save` | `animation_dirichlet_learning.gif` | `animation_dirichlet_learning.npz`, `animation_dirichlet_learning.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/dirichlet_learning/visualize_dirichlet_learning.py --save
uv run python extras/dirichlet_learning/simulate_dirichlet_learning.py --save
uv run python extras/dirichlet_learning/interactive_dirichlet_learning.py
uv run python extras/dirichlet_learning/animation_dirichlet_learning.py --save
```

