# Factorial Depth

Multiple state factors and observation modalities.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10
- Sections: 10.3, 12.6

## Scripts

- `visualize_factorial_depth.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_factorial_depth.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_factorial_depth.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/factorial_depth` and raw-data sidecars under `output/data/extras/factorial_depth`.

```bash
uv run python extras/factorial_depth/visualize_factorial_depth.py --save
uv run python extras/factorial_depth/simulate_factorial_depth.py --save
uv run python extras/factorial_depth/interactive_factorial_depth.py
```
