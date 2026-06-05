# Variational Message Passing

Mean-field updates expressed as local messages.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 12
- Sections: 12.4

## Scripts

- `visualize_variational_message_passing.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_variational_message_passing.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_variational_message_passing.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/variational_message_passing` and raw-data sidecars under `output/data/extras/variational_message_passing`.

```bash
uv run python extras/variational_message_passing/visualize_variational_message_passing.py --save
uv run python extras/variational_message_passing/simulate_variational_message_passing.py --save
uv run python extras/variational_message_passing/interactive_variational_message_passing.py
```
