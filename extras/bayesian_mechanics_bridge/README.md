# Bayesian Mechanics Bridge

A careful bridge between active inference, FEP, and Bayesian mechanics.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 14
- Sections: 14.1, 14.4, A

## Scripts

- `visualize_bayesian_mechanics_bridge.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_bayesian_mechanics_bridge.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_bayesian_mechanics_bridge.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/bayesian_mechanics_bridge` and raw-data sidecars under `output/data/extras/bayesian_mechanics_bridge`.

```bash
uv run python extras/bayesian_mechanics_bridge/visualize_bayesian_mechanics_bridge.py --save
uv run python extras/bayesian_mechanics_bridge/simulate_bayesian_mechanics_bridge.py --save
uv run python extras/bayesian_mechanics_bridge/interactive_bayesian_mechanics_bridge.py
```
