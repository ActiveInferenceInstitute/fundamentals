# Variational Message Passing

Mean-field updates expressed as local messages.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 12
- Sections: 12.4, 12.4.1

## Scripts

- `visualize_variational_message_passing.py` - Static visualization using shared library APIs.
- `simulate_variational_message_passing.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_variational_message_passing.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/variational_message_passing/` and `output/data/extras/variational_message_passing/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_variational_message_passing.py --save` | `visualize_variational_message_passing.png` | `visualize_variational_message_passing.npz`, `visualize_variational_message_passing.json` |
| `simulate_variational_message_passing.py --save` | `simulate_variational_message_passing.png` | `simulate_variational_message_passing.npz`, `simulate_variational_message_passing.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/variational_message_passing/visualize_variational_message_passing.py --save
uv run python extras/variational_message_passing/simulate_variational_message_passing.py --save
uv run python extras/variational_message_passing/interactive_variational_message_passing.py
```

