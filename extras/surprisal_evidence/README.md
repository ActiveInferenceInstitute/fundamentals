# Surprisal And Evidence

Evidence, negative log evidence, and bound gaps.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.2, 4.3

## Scripts

- `visualize_surprisal_evidence.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_surprisal_evidence.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_surprisal_evidence.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/surprisal_evidence` and raw-data sidecars under `output/data/extras/surprisal_evidence`.

```bash
uv run python extras/surprisal_evidence/visualize_surprisal_evidence.py --save
uv run python extras/surprisal_evidence/simulate_surprisal_evidence.py --save
uv run python extras/surprisal_evidence/interactive_surprisal_evidence.py
```
