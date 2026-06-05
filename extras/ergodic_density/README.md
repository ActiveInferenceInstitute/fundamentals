# Ergodic Density

Long-run occupancy as a density over viable states.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 14
- Sections: 14.1, 14.2

## Scripts

- `visualize_ergodic_density.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_ergodic_density.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_ergodic_density.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_ergodic_density.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/ergodic_density` and raw-data sidecars under `output/data/extras/ergodic_density`.

```bash
uv run python extras/ergodic_density/visualize_ergodic_density.py --save
uv run python extras/ergodic_density/simulate_ergodic_density.py --save
uv run python extras/ergodic_density/interactive_ergodic_density.py
uv run python extras/ergodic_density/animation_ergodic_density.py --save
```
