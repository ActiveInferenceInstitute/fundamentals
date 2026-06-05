# Generalized Filtering

Dynamic state inference with generalized filtering.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 6
- Sections: 6.1, 6.2

## Scripts

- `visualize_generalized_filtering.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_generalized_filtering.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_generalized_filtering.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_generalized_filtering.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/generalized_filtering` and raw-data sidecars under `output/data/extras/generalized_filtering`.

```bash
uv run python extras/generalized_filtering/visualize_generalized_filtering.py --save
uv run python extras/generalized_filtering/simulate_generalized_filtering.py --save
uv run python extras/generalized_filtering/interactive_generalized_filtering.py
uv run python extras/generalized_filtering/animation_generalized_filtering.py --save
```
