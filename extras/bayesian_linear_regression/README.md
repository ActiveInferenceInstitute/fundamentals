# Bayesian Linear Regression

Posterior parameter uncertainty and predictive bands.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.3

## Scripts

- `visualize_bayesian_linear_regression.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_bayesian_linear_regression.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_bayesian_linear_regression.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_bayesian_linear_regression.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/bayesian_linear_regression` and raw-data sidecars under `output/data/extras/bayesian_linear_regression`.

```bash
uv run python extras/bayesian_linear_regression/visualize_bayesian_linear_regression.py --save
uv run python extras/bayesian_linear_regression/simulate_bayesian_linear_regression.py --save
uv run python extras/bayesian_linear_regression/interactive_bayesian_linear_regression.py
uv run python extras/bayesian_linear_regression/animation_bayesian_linear_regression.py --save
```
