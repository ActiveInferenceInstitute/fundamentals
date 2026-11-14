# Variational Free Energy

Energy, entropy, KL, and surprisal decompositions of VFE.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.2, 4.3, D.1

## Scripts

- `visualize_variational_free_energy.py` - Static visualization using shared library APIs.
- `simulate_variational_free_energy.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_variational_free_energy.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_variational_free_energy.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/variational_free_energy/` and `output/data/extras/variational_free_energy/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_variational_free_energy.py --save` | `visualize_variational_free_energy.png` | `visualize_variational_free_energy.npz`, `visualize_variational_free_energy.json` |
| `simulate_variational_free_energy.py --save` | `simulate_variational_free_energy.png` | `simulate_variational_free_energy.npz`, `simulate_variational_free_energy.json` |
| `animation_variational_free_energy.py --save` | `animation_variational_free_energy.gif` | `animation_variational_free_energy.npz`, `animation_variational_free_energy.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/variational_free_energy/visualize_variational_free_energy.py --save
uv run python extras/variational_free_energy/simulate_variational_free_energy.py --save
uv run python extras/variational_free_energy/interactive_variational_free_energy.py
uv run python extras/variational_free_energy/animation_variational_free_energy.py --save
```

