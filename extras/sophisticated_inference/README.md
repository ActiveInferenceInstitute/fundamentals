# Sophisticated Inference

Planning with beliefs over future belief updates.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.1

## Scripts

- `visualize_sophisticated_inference.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_sophisticated_inference.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_sophisticated_inference.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/sophisticated_inference` and raw-data sidecars under `output/data/extras/sophisticated_inference`.

```bash
uv run python extras/sophisticated_inference/visualize_sophisticated_inference.py --save
uv run python extras/sophisticated_inference/simulate_sophisticated_inference.py --save
uv run python extras/sophisticated_inference/interactive_sophisticated_inference.py
```
