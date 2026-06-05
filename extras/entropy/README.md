# Entropy

Discrete and differential entropy, including negative differential cases.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4, 14
- Sections: C.10.4, 14.3

## Scripts

- `visualize_entropy.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_entropy.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_entropy.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/entropy` and raw-data sidecars under `output/data/extras/entropy`.

```bash
uv run python extras/entropy/visualize_entropy.py --save
uv run python extras/entropy/simulate_entropy.py --save
uv run python extras/entropy/interactive_entropy.py
```
