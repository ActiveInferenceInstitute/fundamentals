# KL Divergence

Asymmetric divergence and posterior approximation loss.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.1, C.10.5

## Scripts

- `visualize_kl_divergence.py` - Static visualization using shared library APIs.
- `simulate_kl_divergence.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_kl_divergence.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/kl_divergence/` and `output/data/extras/kl_divergence/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_kl_divergence.py --save` | `visualize_kl_divergence.png` | `visualize_kl_divergence.npz`, `visualize_kl_divergence.json` |
| `simulate_kl_divergence.py --save` | `simulate_kl_divergence.png` | `simulate_kl_divergence.npz`, `simulate_kl_divergence.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/kl_divergence/visualize_kl_divergence.py --save
uv run python extras/kl_divergence/simulate_kl_divergence.py --save
uv run python extras/kl_divergence/interactive_kl_divergence.py
```

