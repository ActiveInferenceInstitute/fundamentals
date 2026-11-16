# Expected Free Energy

Risk, ambiguity, and epistemic value in policy scoring.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.5, D.3

## Scripts

- `visualize_expected_free_energy.py` - Static visualization using shared library APIs.
- `simulate_expected_free_energy.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_expected_free_energy.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/expected_free_energy/` and `output/data/extras/expected_free_energy/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_expected_free_energy.py --save` | `visualize_expected_free_energy.png` | `visualize_expected_free_energy.npz`, `visualize_expected_free_energy.json` |
| `simulate_expected_free_energy.py --save` | `simulate_expected_free_energy.png` | `simulate_expected_free_energy.npz`, `simulate_expected_free_energy.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/expected_free_energy/visualize_expected_free_energy.py --save
uv run python extras/expected_free_energy/simulate_expected_free_energy.py --save
uv run python extras/expected_free_energy/interactive_expected_free_energy.py
```

