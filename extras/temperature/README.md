# Temperature

Temperature-scaled canonical probabilities and U - T S.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 4, 14
- Sections: D, 14.3

## Scripts

- `visualize_temperature.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_temperature.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_temperature.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/temperature` and raw-data sidecars under `output/data/extras/temperature`.

```bash
uv run python extras/temperature/visualize_temperature.py --save
uv run python extras/temperature/simulate_temperature.py --save
uv run python extras/temperature/interactive_temperature.py
```
