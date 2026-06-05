# Expected Free Energy

Risk, ambiguity, and epistemic value in policy scoring.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.5, D.3

## Scripts

- `visualize_expected_free_energy.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_expected_free_energy.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_expected_free_energy.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/expected_free_energy` and raw-data sidecars under `output/data/extras/expected_free_energy`.

```bash
uv run python extras/expected_free_energy/visualize_expected_free_energy.py --save
uv run python extras/expected_free_energy/simulate_expected_free_energy.py --save
uv run python extras/expected_free_energy/interactive_expected_free_energy.py
```
