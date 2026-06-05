# Precision Weighting

Variance, precision, and their control over belief updates.

## Book Mapping

- Family: Foundations
- Chapters: 2, 5, 10
- Sections: 2.2, 5.1, 10.2

## Scripts

- `visualize_precision_weighting.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_precision_weighting.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_precision_weighting.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/precision_weighting` and raw-data sidecars under `output/data/extras/precision_weighting`.

```bash
uv run python extras/precision_weighting/visualize_precision_weighting.py --save
uv run python extras/precision_weighting/simulate_precision_weighting.py --save
uv run python extras/precision_weighting/interactive_precision_weighting.py
```
