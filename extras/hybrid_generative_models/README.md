# Hybrid Generative Models

Continuous and discrete state-space components in one model.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11, 12
- Sections: 11.3, 12.6

## Scripts

- `visualize_hybrid_generative_models.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_hybrid_generative_models.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_hybrid_generative_models.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/hybrid_generative_models` and raw-data sidecars under `output/data/extras/hybrid_generative_models`.

```bash
uv run python extras/hybrid_generative_models/visualize_hybrid_generative_models.py --save
uv run python extras/hybrid_generative_models/simulate_hybrid_generative_models.py --save
uv run python extras/hybrid_generative_models/interactive_hybrid_generative_models.py
```
