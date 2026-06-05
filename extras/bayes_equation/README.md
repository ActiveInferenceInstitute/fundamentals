# Bayes Equation

Prior, likelihood, evidence, posterior, and normalization.

## Book Mapping

- Family: Foundations
- Chapters: 1, 2
- Sections: 1.3, 2.1.4, C.2.4

## Scripts

- `visualize_bayes_equation.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_bayes_equation.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_bayes_equation.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/bayes_equation` and raw-data sidecars under `output/data/extras/bayes_equation`.

```bash
uv run python extras/bayes_equation/visualize_bayes_equation.py --save
uv run python extras/bayes_equation/simulate_bayes_equation.py --save
uv run python extras/bayes_equation/interactive_bayes_equation.py
```
