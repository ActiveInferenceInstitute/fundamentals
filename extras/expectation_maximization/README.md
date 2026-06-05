# Expectation Maximization

Alternating latent expectation and parameter maximization.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 3
- Sections: 3.5

## Scripts

- `visualize_expectation_maximization.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_expectation_maximization.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_expectation_maximization.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_expectation_maximization.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/expectation_maximization` and raw-data sidecars under `output/data/extras/expectation_maximization`.

```bash
uv run python extras/expectation_maximization/visualize_expectation_maximization.py --save
uv run python extras/expectation_maximization/simulate_expectation_maximization.py --save
uv run python extras/expectation_maximization/interactive_expectation_maximization.py
uv run python extras/expectation_maximization/animation_expectation_maximization.py --save
```
