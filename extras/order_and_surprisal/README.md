# Order And Surprisal

Order as low-surprisal occupancy over viable states.

## Book Mapping

- Family: Foundations
- Chapters: 1, 14
- Sections: 1.2, 14.3

## Scripts

- `visualize_order_and_surprisal.py` - Static visualization using shared library APIs.
- `simulate_order_and_surprisal.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_order_and_surprisal.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/order_and_surprisal/` and `output/data/extras/order_and_surprisal/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_order_and_surprisal.py --save` | `visualize_order_and_surprisal.png` | `visualize_order_and_surprisal.npz`, `visualize_order_and_surprisal.json` |
| `simulate_order_and_surprisal.py --save` | `simulate_order_and_surprisal.png` | `simulate_order_and_surprisal.npz`, `simulate_order_and_surprisal.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/order_and_surprisal/visualize_order_and_surprisal.py --save
uv run python extras/order_and_surprisal/simulate_order_and_surprisal.py --save
uv run python extras/order_and_surprisal/interactive_order_and_surprisal.py
```

