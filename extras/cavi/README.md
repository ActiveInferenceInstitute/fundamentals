# CAVI

Coordinate-ascent updates as repeated local message refinement.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4, 12
- Sections: 4.5, 12.4

## Scripts

- `visualize_cavi.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_cavi.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_cavi.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_cavi.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/cavi` and raw-data sidecars under `output/data/extras/cavi`.

```bash
uv run python extras/cavi/visualize_cavi.py --save
uv run python extras/cavi/simulate_cavi.py --save
uv run python extras/cavi/interactive_cavi.py
uv run python extras/cavi/animation_cavi.py --save
```
