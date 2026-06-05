# Structure Learning

Comparing candidate model structures through evidence-like scores.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.5

## Scripts

- `visualize_structure_learning.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_structure_learning.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_structure_learning.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/structure_learning` and raw-data sidecars under `output/data/extras/structure_learning`.

```bash
uv run python extras/structure_learning/visualize_structure_learning.py --save
uv run python extras/structure_learning/simulate_structure_learning.py --save
uv run python extras/structure_learning/interactive_structure_learning.py
```
