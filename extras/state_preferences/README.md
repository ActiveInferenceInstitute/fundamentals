# State Preferences

Preferences over states and time-dependent preference schedules.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.3, 11.2.5

## Scripts

- `visualize_state_preferences.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_state_preferences.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_state_preferences.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/state_preferences` and raw-data sidecars under `output/data/extras/state_preferences`.

```bash
uv run python extras/state_preferences/visualize_state_preferences.py --save
uv run python extras/state_preferences/simulate_state_preferences.py --save
uv run python extras/state_preferences/interactive_state_preferences.py
```
