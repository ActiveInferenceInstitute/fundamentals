# State Preferences

Preferences over states and time-dependent preference schedules.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.3, 11.2.5

## Scripts

- `visualize_state_preferences.py` - Static visualization using shared library APIs.
- `simulate_state_preferences.py` - Parameter-sweep simulation using shared library APIs.
- `interactive_state_preferences.py` - Host GUI sliders; no saved media or data sidecars.

## Outputs

Run non-interactive scripts with `--save`. The default artifact paths are
`output/figures/extras/state_preferences/` and `output/data/extras/state_preferences/`.
These are declared outputs, not evidence that rendering has run.

| Script | Figure | Raw data and manifest |
|---|---|---|
| `visualize_state_preferences.py --save` | `visualize_state_preferences.png` | `visualize_state_preferences.npz`, `visualize_state_preferences.json` |
| `simulate_state_preferences.py --save` | `simulate_state_preferences.png` | `simulate_state_preferences.npz`, `simulate_state_preferences.json` |

Interactive mode requires a host display. Its `--save` option closes
the GUI without writing artifacts; use the commands below as shown.

```bash
uv run python extras/state_preferences/visualize_state_preferences.py --save
uv run python extras/state_preferences/simulate_state_preferences.py --save
uv run python extras/state_preferences/interactive_state_preferences.py
```

