# Discrete VFE

Discrete free energy for hidden-state estimation.

## Book Mapping

- Family: Discrete POMDP Active Inference
- Chapters: 9
- Sections: 9.3

## Scripts

- `visualize_discrete_vfe.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_discrete_vfe.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_discrete_vfe.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/discrete_vfe` and raw-data sidecars under `output/data/extras/discrete_vfe`.

```bash
uv run python extras/discrete_vfe/visualize_discrete_vfe.py --save
uv run python extras/discrete_vfe/simulate_discrete_vfe.py --save
uv run python extras/discrete_vfe/interactive_discrete_vfe.py
```
