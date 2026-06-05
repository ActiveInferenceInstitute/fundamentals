# Social Robotics

Belief updates over another agent's hidden intention.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 13
- Sections: 13.3

## Scripts

- `visualize_social_robotics.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_social_robotics.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_social_robotics.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/social_robotics` and raw-data sidecars under `output/data/extras/social_robotics`.

```bash
uv run python extras/social_robotics/visualize_social_robotics.py --save
uv run python extras/social_robotics/simulate_social_robotics.py --save
uv run python extras/social_robotics/interactive_social_robotics.py
```
