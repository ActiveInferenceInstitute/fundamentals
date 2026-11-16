# Discrete VFE

Discrete free energy for hidden-state estimation.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.3

## Scripts

- `visualize_discrete_vfe.py` - Static visualization using shared library APIs.
- `simulate_discrete_vfe.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_discrete_vfe.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/discrete_vfe/` and `output/data/extras/discrete_vfe/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_discrete_vfe.py --save` | `visualize_discrete_vfe.png` | `visualize_discrete_vfe.npz`, `visualize_discrete_vfe.json` |
| `simulate_discrete_vfe.py --save` | `simulate_discrete_vfe.png` | `simulate_discrete_vfe.npz`, `simulate_discrete_vfe.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/discrete_vfe/visualize_discrete_vfe.py --save
uv run python extras/discrete_vfe/simulate_discrete_vfe.py --save
uv run python extras/discrete_vfe/interactive_discrete_vfe.py
```

