# Enthalpy

H = U + pV and G = H - T S as explicit analogy-layer quantities.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 4, 14
- Sections: D.4

## Scripts

- `visualize_enthalpy.py` - Static visualization using shared library APIs.
- `simulate_enthalpy.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_enthalpy.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/enthalpy/` and `output/data/extras/enthalpy/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_enthalpy.py --save` | `visualize_enthalpy.png` | `visualize_enthalpy.npz`, `visualize_enthalpy.json` |
| `simulate_enthalpy.py --save` | `simulate_enthalpy.png` | `simulate_enthalpy.npz`, `simulate_enthalpy.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/enthalpy/visualize_enthalpy.py --save
uv run python extras/enthalpy/simulate_enthalpy.py --save
uv run python extras/enthalpy/interactive_enthalpy.py
```

