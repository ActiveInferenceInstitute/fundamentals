# Learning And Attention

Learning first- and second-order parameters through precision.

## Book Mapping

- Family: Active Inference Core
- Chapters: 8
- Sections: 8.1

## Scripts

- `visualize_learning_attention.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_learning_attention.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_learning_attention.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_learning_attention.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/learning_attention` and raw-data sidecars under `output/data/extras/learning_attention`.

```bash
uv run python extras/learning_attention/visualize_learning_attention.py --save
uv run python extras/learning_attention/simulate_learning_attention.py --save
uv run python extras/learning_attention/interactive_learning_attention.py
uv run python extras/learning_attention/animation_learning_attention.py --save
```
