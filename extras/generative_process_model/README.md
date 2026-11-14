# Generative Process And Model

Contrasting the data-generating process with the agent's model.

## Book Mapping

- Family: Foundations
- Chapters: 2
- Sections: 2.1.2, 2.1.3, B.4

## Scripts

- `visualize_generative_process_model.py` - Static visualization using shared library APIs.
- `simulate_generative_process_model.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_generative_process_model.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/generative_process_model/` and `output/data/extras/generative_process_model/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_generative_process_model.py --save` | `visualize_generative_process_model.png` | `visualize_generative_process_model.npz`, `visualize_generative_process_model.json` |
| `simulate_generative_process_model.py --save` | `simulate_generative_process_model.png` | `simulate_generative_process_model.npz`, `simulate_generative_process_model.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/generative_process_model/visualize_generative_process_model.py --save
uv run python extras/generative_process_model/simulate_generative_process_model.py --save
uv run python extras/generative_process_model/interactive_generative_process_model.py
```

