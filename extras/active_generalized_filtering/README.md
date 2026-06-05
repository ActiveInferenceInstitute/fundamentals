# Active Generalized Filtering

Action and perception as coupled free-energy descent.

## Book Mapping

- Family: Active Inference Core
- Chapters: 7
- Sections: 7.2, 7.4, 7.5

## Scripts

- `visualize_active_generalized_filtering.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_active_generalized_filtering.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_active_generalized_filtering.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_active_generalized_filtering.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/active_generalized_filtering` and raw-data sidecars under `output/data/extras/active_generalized_filtering`.

```bash
uv run python extras/active_generalized_filtering/visualize_active_generalized_filtering.py --save
uv run python extras/active_generalized_filtering/simulate_active_generalized_filtering.py --save
uv run python extras/active_generalized_filtering/interactive_active_generalized_filtering.py
uv run python extras/active_generalized_filtering/animation_active_generalized_filtering.py --save
```
