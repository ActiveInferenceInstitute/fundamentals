# Discrete Belief Filtering

Dynamic categorical belief updates over time.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.2

## Scripts

- `visualize_discrete_belief_filtering.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_discrete_belief_filtering.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_discrete_belief_filtering.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_discrete_belief_filtering.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/discrete_belief_filtering` and raw-data sidecars under `output/data/extras/discrete_belief_filtering`.

```bash
uv run python extras/discrete_belief_filtering/visualize_discrete_belief_filtering.py --save
uv run python extras/discrete_belief_filtering/simulate_discrete_belief_filtering.py --save
uv run python extras/discrete_belief_filtering/interactive_discrete_belief_filtering.py
uv run python extras/discrete_belief_filtering/animation_discrete_belief_filtering.py --save
```
