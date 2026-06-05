# Policy Precision And Habits

Policy precision and baseline habits in action selection.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10
- Sections: 10.2

## Scripts

- `visualize_policy_precision_habits.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_policy_precision_habits.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_policy_precision_habits.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_policy_precision_habits.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/policy_precision_habits` and raw-data sidecars under `output/data/extras/policy_precision_habits`.

```bash
uv run python extras/policy_precision_habits/visualize_policy_precision_habits.py --save
uv run python extras/policy_precision_habits/simulate_policy_precision_habits.py --save
uv run python extras/policy_precision_habits/interactive_policy_precision_habits.py
uv run python extras/policy_precision_habits/animation_policy_precision_habits.py --save
```
