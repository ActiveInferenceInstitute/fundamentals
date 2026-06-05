# Linear Regression

Deterministic parameter estimation and residual geometry.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.1, 3.2

## Scripts

- `visualize_linear_regression.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_linear_regression.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_linear_regression.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/linear_regression` and raw-data sidecars under `output/data/extras/linear_regression`.

```bash
uv run python extras/linear_regression/visualize_linear_regression.py --save
uv run python extras/linear_regression/simulate_linear_regression.py --save
uv run python extras/linear_regression/interactive_linear_regression.py
```
