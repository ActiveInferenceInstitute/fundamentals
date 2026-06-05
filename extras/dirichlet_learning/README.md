# Dirichlet Learning

Pseudocount accumulation for POMDP parameter learning.

## Book Mapping

- Family: Learning And Depth
- Chapters: 10
- Sections: 10.1

## Scripts

- `visualize_dirichlet_learning.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_dirichlet_learning.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_dirichlet_learning.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_dirichlet_learning.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/dirichlet_learning` and raw-data sidecars under `output/data/extras/dirichlet_learning`.

```bash
uv run python extras/dirichlet_learning/visualize_dirichlet_learning.py --save
uv run python extras/dirichlet_learning/simulate_dirichlet_learning.py --save
uv run python extras/dirichlet_learning/interactive_dirichlet_learning.py
uv run python extras/dirichlet_learning/animation_dirichlet_learning.py --save
```
