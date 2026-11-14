# Surprisal And Evidence

Evidence, negative log evidence, and bound gaps.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.2, 4.3

## Scripts

- `visualize_surprisal_evidence.py` - Static visualization using shared library APIs.
- `simulate_surprisal_evidence.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_surprisal_evidence.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/surprisal_evidence/` and `output/data/extras/surprisal_evidence/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_surprisal_evidence.py --save` | `visualize_surprisal_evidence.png` | `visualize_surprisal_evidence.npz`, `visualize_surprisal_evidence.json` |
| `simulate_surprisal_evidence.py --save` | `simulate_surprisal_evidence.png` | `simulate_surprisal_evidence.npz`, `simulate_surprisal_evidence.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/surprisal_evidence/visualize_surprisal_evidence.py --save
uv run python extras/surprisal_evidence/simulate_surprisal_evidence.py --save
uv run python extras/surprisal_evidence/interactive_surprisal_evidence.py
```

