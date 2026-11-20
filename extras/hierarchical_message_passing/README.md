# Hierarchical Message Passing

Forward and backward messages across hierarchical layers.

## Book Mapping

- Family: Active Inference Core
- Chapters: 8, 12
- Sections: 8.5, 12.5, 12.7

## Scripts

- `visualize_hierarchical_message_passing.py` - Static visualization using shared library APIs.
- `simulate_hierarchical_message_passing.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_hierarchical_message_passing.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_hierarchical_message_passing.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/hierarchical_message_passing/` and `output/data/extras/hierarchical_message_passing/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_hierarchical_message_passing.py --save` | `visualize_hierarchical_message_passing.png` | `visualize_hierarchical_message_passing.npz`, `visualize_hierarchical_message_passing.json` |
| `simulate_hierarchical_message_passing.py --save` | `simulate_hierarchical_message_passing.png` | `simulate_hierarchical_message_passing.npz`, `simulate_hierarchical_message_passing.json` |
| `animation_hierarchical_message_passing.py --save` | `animation_hierarchical_message_passing.gif` | `animation_hierarchical_message_passing.npz`, `animation_hierarchical_message_passing.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/hierarchical_message_passing/visualize_hierarchical_message_passing.py --save
uv run python extras/hierarchical_message_passing/simulate_hierarchical_message_passing.py --save
uv run python extras/hierarchical_message_passing/interactive_hierarchical_message_passing.py
uv run python extras/hierarchical_message_passing/animation_hierarchical_message_passing.py --save
```

