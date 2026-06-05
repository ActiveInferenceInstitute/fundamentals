# Mean-Field Variational Inference

Factorized approximations and coordinate updates.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.5, 4.6

## Scripts

- `visualize_mean_field_variational_inference.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_mean_field_variational_inference.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_mean_field_variational_inference.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/mean_field_variational_inference` and raw-data sidecars under `output/data/extras/mean_field_variational_inference`.

```bash
uv run python extras/mean_field_variational_inference/visualize_mean_field_variational_inference.py --save
uv run python extras/mean_field_variational_inference/simulate_mean_field_variational_inference.py --save
uv run python extras/mean_field_variational_inference/interactive_mean_field_variational_inference.py
```
