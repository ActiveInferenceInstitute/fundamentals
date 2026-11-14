# FEP Entropy Bounds

Entropy and VFE bounds for self-organizing systems.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 14
- Sections: 14.3

## Scripts

- `visualize_fep_entropy_bounds.py` - Static visualization using shared library APIs.
- `simulate_fep_entropy_bounds.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_fep_entropy_bounds.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/fep_entropy_bounds/` and `output/data/extras/fep_entropy_bounds/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_fep_entropy_bounds.py --save` | `visualize_fep_entropy_bounds.png` | `visualize_fep_entropy_bounds.npz`, `visualize_fep_entropy_bounds.json` |
| `simulate_fep_entropy_bounds.py --save` | `simulate_fep_entropy_bounds.png` | `simulate_fep_entropy_bounds.npz`, `simulate_fep_entropy_bounds.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/fep_entropy_bounds/visualize_fep_entropy_bounds.py --save
uv run python extras/fep_entropy_bounds/simulate_fep_entropy_bounds.py --save
uv run python extras/fep_entropy_bounds/interactive_fep_entropy_bounds.py
```

