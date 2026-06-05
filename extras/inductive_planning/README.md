# Inductive Planning

Policy search that reuses substructure across paths.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.2.2

## Scripts

- `visualize_inductive_planning.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_inductive_planning.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_inductive_planning.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/inductive_planning` and raw-data sidecars under `output/data/extras/inductive_planning`.

```bash
uv run python extras/inductive_planning/visualize_inductive_planning.py --save
uv run python extras/inductive_planning/simulate_inductive_planning.py --save
uv run python extras/inductive_planning/interactive_inductive_planning.py
```
