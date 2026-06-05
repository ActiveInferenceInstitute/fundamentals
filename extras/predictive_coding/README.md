# Predictive Coding

Prediction errors and recognition dynamics.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 5
- Sections: 5.1, 5.2

## Scripts

- `visualize_predictive_coding.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_predictive_coding.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_predictive_coding.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_predictive_coding.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/predictive_coding` and raw-data sidecars under `output/data/extras/predictive_coding`.

```bash
uv run python extras/predictive_coding/visualize_predictive_coding.py --save
uv run python extras/predictive_coding/simulate_predictive_coding.py --save
uv run python extras/predictive_coding/interactive_predictive_coding.py
uv run python extras/predictive_coding/animation_predictive_coding.py --save
```
