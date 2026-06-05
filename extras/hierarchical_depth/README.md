# Hierarchical Depth

Nested policies and slower contextual layers.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10, 12
- Sections: 10.4, 12.6

## Scripts

- `visualize_hierarchical_depth.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_hierarchical_depth.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_hierarchical_depth.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/hierarchical_depth` and raw-data sidecars under `output/data/extras/hierarchical_depth`.

```bash
uv run python extras/hierarchical_depth/visualize_hierarchical_depth.py --save
uv run python extras/hierarchical_depth/simulate_hierarchical_depth.py --save
uv run python extras/hierarchical_depth/interactive_hierarchical_depth.py
```
