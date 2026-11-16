# Bayesian Mechanics Bridge

A careful bridge between active inference, FEP, and Bayesian mechanics.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 14
- Sections: 14.1, 14.4

## Scripts

- `visualize_bayesian_mechanics_bridge.py` - Static visualization using shared library APIs.
- `simulate_bayesian_mechanics_bridge.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_bayesian_mechanics_bridge.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/bayesian_mechanics_bridge/` and `output/data/extras/bayesian_mechanics_bridge/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_bayesian_mechanics_bridge.py --save` | `visualize_bayesian_mechanics_bridge.png` | `visualize_bayesian_mechanics_bridge.npz`, `visualize_bayesian_mechanics_bridge.json` |
| `simulate_bayesian_mechanics_bridge.py --save` | `simulate_bayesian_mechanics_bridge.png` | `simulate_bayesian_mechanics_bridge.npz`, `simulate_bayesian_mechanics_bridge.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/bayesian_mechanics_bridge/visualize_bayesian_mechanics_bridge.py --save
uv run python extras/bayesian_mechanics_bridge/simulate_bayesian_mechanics_bridge.py --save
uv run python extras/bayesian_mechanics_bridge/interactive_bayesian_mechanics_bridge.py
```

