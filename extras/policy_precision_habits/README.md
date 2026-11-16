# Policy Precision And Habits

Policy precision and baseline habits in action selection.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10
- Sections: 10.2

## Scripts

- `visualize_policy_precision_habits.py` - Static visualization using shared library APIs.
- `simulate_policy_precision_habits.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_policy_precision_habits.py` - Host GUI sliders; no saved media or data sidecars.
- `animation_policy_precision_habits.py` - Animated trajectory using shared library APIs.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/policy_precision_habits/` and `output/data/extras/policy_precision_habits/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_policy_precision_habits.py --save` | `visualize_policy_precision_habits.png` | `visualize_policy_precision_habits.npz`, `visualize_policy_precision_habits.json` |
| `simulate_policy_precision_habits.py --save` | `simulate_policy_precision_habits.png` | `simulate_policy_precision_habits.npz`, `simulate_policy_precision_habits.json` |
| `animation_policy_precision_habits.py --save` | `animation_policy_precision_habits.gif` | `animation_policy_precision_habits.npz`, `animation_policy_precision_habits.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/policy_precision_habits/visualize_policy_precision_habits.py --save
uv run python extras/policy_precision_habits/simulate_policy_precision_habits.py --save
uv run python extras/policy_precision_habits/interactive_policy_precision_habits.py
uv run python extras/policy_precision_habits/animation_policy_precision_habits.py --save
```

