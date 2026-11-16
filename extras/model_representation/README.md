# Model And Representation

How internal representations stand in for hidden causes.

## Book Mapping

- Family: Foundations
- Chapters: 1
- Sections: 1.1

## Scripts

- `visualize_model_representation.py` - Static visualization using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/model_representation/` and `output/data/extras/model_representation/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_model_representation.py --save` | `visualize_model_representation.png` | `visualize_model_representation.npz`, `visualize_model_representation.json` |

```bash
uv run python extras/model_representation/visualize_model_representation.py --save
```

