# Hierarchical Message Passing

Forward and backward messages across hierarchical layers.

## Book Mapping

- Family: Active Inference Core
- Chapters: 8, 12
- Sections: 8.5, 12.5

## Scripts

- `visualize_hierarchical_message_passing.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_hierarchical_message_passing.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_hierarchical_message_passing.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_hierarchical_message_passing.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/hierarchical_message_passing` and raw-data sidecars under `output/data/extras/hierarchical_message_passing`.

```bash
uv run python extras/hierarchical_message_passing/visualize_hierarchical_message_passing.py --save
uv run python extras/hierarchical_message_passing/simulate_hierarchical_message_passing.py --save
uv run python extras/hierarchical_message_passing/interactive_hierarchical_message_passing.py
uv run python extras/hierarchical_message_passing/animation_hierarchical_message_passing.py --save
```
