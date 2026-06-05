# MLE And MAP

Likelihood-only and prior-regularized point estimates.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 2
- Sections: 2.5.1

## Scripts

- `visualize_mle_map.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_mle_map.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_mle_map.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/mle_map` and raw-data sidecars under `output/data/extras/mle_map`.

```bash
uv run python extras/mle_map/visualize_mle_map.py --save
uv run python extras/mle_map/simulate_mle_map.py --save
uv run python extras/mle_map/interactive_mle_map.py
```
