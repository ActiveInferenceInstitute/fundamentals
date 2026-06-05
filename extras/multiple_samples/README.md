# Multiple Samples

How repeated observations tighten posterior beliefs.

## Book Mapping

- Family: Statistical Estimation
- Chapters: 2
- Sections: 2.3

## Scripts

- `visualize_multiple_samples.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_multiple_samples.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_multiple_samples.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/multiple_samples` and raw-data sidecars under `output/data/extras/multiple_samples`.

```bash
uv run python extras/multiple_samples/visualize_multiple_samples.py --save
uv run python extras/multiple_samples/simulate_multiple_samples.py --save
uv run python extras/multiple_samples/interactive_multiple_samples.py
```
