# Hierarchical Predictive Coding

Layered prediction-error propagation.

## Book Mapping

- Family: Predictive Coding And Continuous Dynamics
- Chapters: 5, 8
- Sections: 5.4, 8.3

## Scripts

- `visualize_hierarchical_predictive_coding.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_hierarchical_predictive_coding.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_hierarchical_predictive_coding.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_hierarchical_predictive_coding.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/hierarchical_predictive_coding` and raw-data sidecars under `output/data/extras/hierarchical_predictive_coding`.

```bash
uv run python extras/hierarchical_predictive_coding/visualize_hierarchical_predictive_coding.py --save
uv run python extras/hierarchical_predictive_coding/simulate_hierarchical_predictive_coding.py --save
uv run python extras/hierarchical_predictive_coding/interactive_hierarchical_predictive_coding.py
uv run python extras/hierarchical_predictive_coding/animation_hierarchical_predictive_coding.py --save
```
