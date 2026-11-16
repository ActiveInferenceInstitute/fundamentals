# Multiple Samples

How repeated observations tighten posterior beliefs.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 2
- Sections: 2.3

## Scripts

- `visualize_multiple_samples.py` - Static visualization using shared library APIs.
- `simulate_multiple_samples.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_multiple_samples.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/multiple_samples/` and `output/data/extras/multiple_samples/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_multiple_samples.py --save` | `visualize_multiple_samples.png` | `visualize_multiple_samples.npz`, `visualize_multiple_samples.json` |
| `simulate_multiple_samples.py --save` | `simulate_multiple_samples.png` | `simulate_multiple_samples.npz`, `simulate_multiple_samples.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/multiple_samples/visualize_multiple_samples.py --save
uv run python extras/multiple_samples/simulate_multiple_samples.py --save
uv run python extras/multiple_samples/interactive_multiple_samples.py
```

