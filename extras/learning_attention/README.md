# Learning And Attention

Learning first- and second-order parameters through precision.

## Book Mapping

- Family: Active Inference Core
- Chapters: 8
- Sections: 8.1

## Scripts

- `visualize_learning_attention.py` - Static visualization using shared library APIs.
- `simulate_learning_attention.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_learning_attention.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_learning_attention.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/learning_attention/` and `output/data/extras/learning_attention/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_learning_attention.py --save` | `visualize_learning_attention.png` | `visualize_learning_attention.npz`, `visualize_learning_attention.json` |
| `simulate_learning_attention.py --save` | `simulate_learning_attention.png` | `simulate_learning_attention.npz`, `simulate_learning_attention.json` |
| `animation_learning_attention.py --save` | `animation_learning_attention.gif` | `animation_learning_attention.npz`, `animation_learning_attention.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/learning_attention/visualize_learning_attention.py --save
uv run python extras/learning_attention/simulate_learning_attention.py --save
uv run python extras/learning_attention/interactive_learning_attention.py
uv run python extras/learning_attention/animation_learning_attention.py --save
```

