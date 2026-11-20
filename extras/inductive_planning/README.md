# Inductive Planning

Policy search that reuses substructure across paths.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.2

## Scripts

- `visualize_inductive_planning.py` - Static visualization using shared library APIs.
- `simulate_inductive_planning.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_inductive_planning.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/inductive_planning/` and `output/data/extras/inductive_planning/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_inductive_planning.py --save` | `visualize_inductive_planning.png` | `visualize_inductive_planning.npz`, `visualize_inductive_planning.json` |
| `simulate_inductive_planning.py --save` | `simulate_inductive_planning.png` | `simulate_inductive_planning.npz`, `simulate_inductive_planning.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/inductive_planning/visualize_inductive_planning.py --save
uv run python extras/inductive_planning/simulate_inductive_planning.py --save
uv run python extras/inductive_planning/interactive_inductive_planning.py
```

