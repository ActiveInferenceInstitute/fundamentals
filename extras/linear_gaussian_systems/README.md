# Linear Gaussian Systems

State-space prediction and filtering under linear Gaussian assumptions.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.4

## Scripts

- `visualize_linear_gaussian_systems.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_linear_gaussian_systems.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_linear_gaussian_systems.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_linear_gaussian_systems.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/linear_gaussian_systems` and raw-data sidecars under `output/data/extras/linear_gaussian_systems`.

```bash
uv run python extras/linear_gaussian_systems/visualize_linear_gaussian_systems.py --save
uv run python extras/linear_gaussian_systems/simulate_linear_gaussian_systems.py --save
uv run python extras/linear_gaussian_systems/interactive_linear_gaussian_systems.py
uv run python extras/linear_gaussian_systems/animation_linear_gaussian_systems.py --save
```
