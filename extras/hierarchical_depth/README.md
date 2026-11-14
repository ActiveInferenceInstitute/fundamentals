# Hierarchical Depth

Nested policies and slower contextual layers.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10, 12
- Sections: 10.4, 12.6

## Scripts

- `visualize_hierarchical_depth.py` - Static visualization using shared library APIs.
- `simulate_hierarchical_depth.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_hierarchical_depth.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/hierarchical_depth/` and `output/data/extras/hierarchical_depth/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_hierarchical_depth.py --save` | `visualize_hierarchical_depth.png` | `visualize_hierarchical_depth.npz`, `visualize_hierarchical_depth.json` |
| `simulate_hierarchical_depth.py --save` | `simulate_hierarchical_depth.png` | `simulate_hierarchical_depth.npz`, `simulate_hierarchical_depth.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/hierarchical_depth/visualize_hierarchical_depth.py --save
uv run python extras/hierarchical_depth/simulate_hierarchical_depth.py --save
uv run python extras/hierarchical_depth/interactive_hierarchical_depth.py
```

