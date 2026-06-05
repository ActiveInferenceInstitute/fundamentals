# Inverse Problem

Recovering hidden states from observations under a generative model.

## Book Mapping

- Family: Foundations
- Chapters: 1, 2
- Sections: 1.4, 2.1.4

## Scripts

- `visualize_inverse_problem.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_inverse_problem.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_inverse_problem.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/inverse_problem` and raw-data sidecars under `output/data/extras/inverse_problem`.

```bash
uv run python extras/inverse_problem/visualize_inverse_problem.py --save
uv run python extras/inverse_problem/simulate_inverse_problem.py --save
uv run python extras/inverse_problem/interactive_inverse_problem.py
```
