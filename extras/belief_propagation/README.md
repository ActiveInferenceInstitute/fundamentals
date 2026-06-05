# Belief Propagation

Sum-product messages for state-space models.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 12
- Sections: 12.2, 12.3

## Scripts

- `visualize_belief_propagation.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_belief_propagation.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_belief_propagation.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_belief_propagation.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/belief_propagation` and raw-data sidecars under `output/data/extras/belief_propagation`.

```bash
uv run python extras/belief_propagation/visualize_belief_propagation.py --save
uv run python extras/belief_propagation/simulate_belief_propagation.py --save
uv run python extras/belief_propagation/interactive_belief_propagation.py
uv run python extras/belief_propagation/animation_belief_propagation.py --save
```
