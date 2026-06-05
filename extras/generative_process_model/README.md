# Generative Process And Model

Contrasting the data-generating process with the agent's model.

## Book Mapping

- Family: Foundations
- Chapters: 2
- Sections: 2.1.2, 2.1.3, B.4

## Scripts

- `visualize_generative_process_model.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_generative_process_model.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_generative_process_model.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/generative_process_model` and raw-data sidecars under `output/data/extras/generative_process_model`.

```bash
uv run python extras/generative_process_model/visualize_generative_process_model.py --save
uv run python extras/generative_process_model/simulate_generative_process_model.py --save
uv run python extras/generative_process_model/interactive_generative_process_model.py
```
