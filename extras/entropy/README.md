# Entropy

Discrete and differential entropy, including negative differential cases.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4, 14
- Sections: C.10.4, 14.3

## Scripts

- `visualize_entropy.py` - Static visualization using shared library APIs.
- `simulate_entropy.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_entropy.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/entropy/` and `output/data/extras/entropy/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_entropy.py --save` | `visualize_entropy.png` | `visualize_entropy.npz`, `visualize_entropy.json` |
| `simulate_entropy.py --save` | `simulate_entropy.png` | `simulate_entropy.npz`, `simulate_entropy.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/entropy/visualize_entropy.py --save
uv run python extras/entropy/simulate_entropy.py --save
uv run python extras/entropy/interactive_entropy.py
```

