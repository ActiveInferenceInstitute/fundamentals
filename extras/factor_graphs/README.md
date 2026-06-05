# Factor Graphs

Forney factor graphs as model diagrams for message passing.

## Book Mapping

- Family: Factor Graphs And Applications
- Chapters: 12
- Sections: 12.1, 12.5

## Scripts

- `visualize_factor_graphs.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_factor_graphs.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_factor_graphs.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/factor_graphs` and raw-data sidecars under `output/data/extras/factor_graphs`.

```bash
uv run python extras/factor_graphs/visualize_factor_graphs.py --save
uv run python extras/factor_graphs/simulate_factor_graphs.py --save
uv run python extras/factor_graphs/interactive_factor_graphs.py
```
