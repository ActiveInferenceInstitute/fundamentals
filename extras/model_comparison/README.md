# Model Comparison

Evidence and Bayes-factor style comparison across models.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.4, C.11.1

## Scripts

- `visualize_model_comparison.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_model_comparison.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_model_comparison.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/model_comparison` and raw-data sidecars under `output/data/extras/model_comparison`.

```bash
uv run python extras/model_comparison/visualize_model_comparison.py --save
uv run python extras/model_comparison/simulate_model_comparison.py --save
uv run python extras/model_comparison/interactive_model_comparison.py
```
