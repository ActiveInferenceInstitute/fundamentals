# Parameter Uncertainty

Forgetting rates and uncertainty on learned parameters.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.6, 11.2.7

## Scripts

- `visualize_parameter_uncertainty.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_parameter_uncertainty.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_parameter_uncertainty.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/parameter_uncertainty` and raw-data sidecars under `output/data/extras/parameter_uncertainty`.

```bash
uv run python extras/parameter_uncertainty/visualize_parameter_uncertainty.py --save
uv run python extras/parameter_uncertainty/simulate_parameter_uncertainty.py --save
uv run python extras/parameter_uncertainty/interactive_parameter_uncertainty.py
```
