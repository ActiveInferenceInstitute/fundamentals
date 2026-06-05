# Gradient Descent

Iterative descent on a differentiable loss surface.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 2, 3
- Sections: 2.5.2, 3.1

## Scripts

- `visualize_gradient_descent.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_gradient_descent.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_gradient_descent.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_gradient_descent.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/gradient_descent` and raw-data sidecars under `output/data/extras/gradient_descent`.

```bash
uv run python extras/gradient_descent/visualize_gradient_descent.py --save
uv run python extras/gradient_descent/simulate_gradient_descent.py --save
uv run python extras/gradient_descent/interactive_gradient_descent.py
uv run python extras/gradient_descent/animation_gradient_descent.py --save
```
