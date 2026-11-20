# Temperature

Temperature-scaled canonical probabilities and U - T S.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 4, 14
- Sections: D.3, 14.3

## Scripts

- `visualize_temperature.py` - Static visualization using shared library APIs.
- `simulate_temperature.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_temperature.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/temperature/` and `output/data/extras/temperature/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_temperature.py --save` | `visualize_temperature.png` | `visualize_temperature.npz`, `visualize_temperature.json` |
| `simulate_temperature.py --save` | `simulate_temperature.png` | `simulate_temperature.npz`, `simulate_temperature.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/temperature/visualize_temperature.py --save
uv run python extras/temperature/simulate_temperature.py --save
uv run python extras/temperature/interactive_temperature.py
```

