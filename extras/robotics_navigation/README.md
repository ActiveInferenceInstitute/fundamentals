# Robotics Navigation

Navigation and control as preference-seeking active inference.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 13
- Sections: 13.1, 13.2

## Scripts

- `visualize_robotics_navigation.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_robotics_navigation.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_robotics_navigation.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_robotics_navigation.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/robotics_navigation` and raw-data sidecars under `output/data/extras/robotics_navigation`.

```bash
uv run python extras/robotics_navigation/visualize_robotics_navigation.py --save
uv run python extras/robotics_navigation/simulate_robotics_navigation.py --save
uv run python extras/robotics_navigation/interactive_robotics_navigation.py
uv run python extras/robotics_navigation/animation_robotics_navigation.py --save
```
