# Backward Smoothing

Backward messages that refine earlier state beliefs.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11, 12
- Sections: 11.2.9, 12.3

## Scripts

- `visualize_backward_smoothing.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_backward_smoothing.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_backward_smoothing.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_backward_smoothing.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/backward_smoothing` and raw-data sidecars under `output/data/extras/backward_smoothing`.

```bash
uv run python extras/backward_smoothing/visualize_backward_smoothing.py --save
uv run python extras/backward_smoothing/simulate_backward_smoothing.py --save
uv run python extras/backward_smoothing/interactive_backward_smoothing.py
uv run python extras/backward_smoothing/animation_backward_smoothing.py --save
```
