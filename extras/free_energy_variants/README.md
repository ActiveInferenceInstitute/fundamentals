# Free-Energy Variants

FEF, OFE, PFE, FEEF, generalized, Bethe, and Renyi forms.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.1, 11.1.1-11.1.8, D.4

## Scripts

- `visualize_free_energy_variants.py` - Static visualization using shared library APIs.
- `simulate_free_energy_variants.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_free_energy_variants.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/free_energy_variants/` and `output/data/extras/free_energy_variants/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_free_energy_variants.py --save` | `visualize_free_energy_variants.png` | `visualize_free_energy_variants.npz`, `visualize_free_energy_variants.json` |
| `simulate_free_energy_variants.py --save` | `simulate_free_energy_variants.png` | `simulate_free_energy_variants.npz`, `simulate_free_energy_variants.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/free_energy_variants/visualize_free_energy_variants.py --save
uv run python extras/free_energy_variants/simulate_free_energy_variants.py --save
uv run python extras/free_energy_variants/interactive_free_energy_variants.py
```

