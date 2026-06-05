# Multivariate Gaussians

Covariance geometry, entropy, and KL in multiple dimensions.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3, 6
- Sections: 3.4, C.10

## Scripts

- `visualize_multivariate_gaussians.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_multivariate_gaussians.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_multivariate_gaussians.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/multivariate_gaussians` and raw-data sidecars under `output/data/extras/multivariate_gaussians`.

```bash
uv run python extras/multivariate_gaussians/visualize_multivariate_gaussians.py --save
uv run python extras/multivariate_gaussians/simulate_multivariate_gaussians.py --save
uv run python extras/multivariate_gaussians/interactive_multivariate_gaussians.py
```
