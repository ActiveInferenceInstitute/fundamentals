# FEP Entropy Bounds

Entropy and VFE bounds for self-organizing systems.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 14
- Sections: 14.3

## Scripts

- `visualize_fep_entropy_bounds.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_fep_entropy_bounds.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_fep_entropy_bounds.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/fep_entropy_bounds` and raw-data sidecars under `output/data/extras/fep_entropy_bounds`.

```bash
uv run python extras/fep_entropy_bounds/visualize_fep_entropy_bounds.py --save
uv run python extras/fep_entropy_bounds/simulate_fep_entropy_bounds.py --save
uv run python extras/fep_entropy_bounds/interactive_fep_entropy_bounds.py
```
