# Generalized Coordinates

State, velocity, and higher-order embedding orders.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 6
- Sections: 6.3, 6.5, 6.6

## Scripts

- `visualize_generalized_coordinates.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_generalized_coordinates.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_generalized_coordinates.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/generalized_coordinates` and raw-data sidecars under `output/data/extras/generalized_coordinates`.

```bash
uv run python extras/generalized_coordinates/visualize_generalized_coordinates.py --save
uv run python extras/generalized_coordinates/simulate_generalized_coordinates.py --save
uv run python extras/generalized_coordinates/interactive_generalized_coordinates.py
```
