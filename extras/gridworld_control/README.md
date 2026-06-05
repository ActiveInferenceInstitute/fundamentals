# Gridworld Control

Planning as inference in controllable grid-world transitions.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.4, 9.5

## Scripts

- `visualize_gridworld_control.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_gridworld_control.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_gridworld_control.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_gridworld_control.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/gridworld_control` and raw-data sidecars under `output/data/extras/gridworld_control`.

```bash
uv run python extras/gridworld_control/visualize_gridworld_control.py --save
uv run python extras/gridworld_control/simulate_gridworld_control.py --save
uv run python extras/gridworld_control/interactive_gridworld_control.py
uv run python extras/gridworld_control/animation_gridworld_control.py --save
```
