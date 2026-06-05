# Hidden State Estimation

Sequential inference over latent states.

## Book Mapping

- Family: Foundations
- Chapters: 2
- Sections: 2.2, 2.3

## Scripts

- `visualize_hidden_state_estimation.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_hidden_state_estimation.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_hidden_state_estimation.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_hidden_state_estimation.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/hidden_state_estimation` and raw-data sidecars under `output/data/extras/hidden_state_estimation`.

```bash
uv run python extras/hidden_state_estimation/visualize_hidden_state_estimation.py --save
uv run python extras/hidden_state_estimation/simulate_hidden_state_estimation.py --save
uv run python extras/hidden_state_estimation/interactive_hidden_state_estimation.py
uv run python extras/hidden_state_estimation/animation_hidden_state_estimation.py --save
```
