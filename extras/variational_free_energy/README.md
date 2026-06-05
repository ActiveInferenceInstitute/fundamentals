# Variational Free Energy

Energy, entropy, KL, and surprisal decompositions of VFE.

## Book Mapping

- Family: Information And Variational Inference
- Chapters: 4
- Sections: 4.2, 4.3, D.1

## Scripts

- `visualize_variational_free_energy.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_variational_free_energy.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_variational_free_energy.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.
- `animation_variational_free_energy.py` - animation orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/variational_free_energy` and raw-data sidecars under `output/data/extras/variational_free_energy`.

```bash
uv run python extras/variational_free_energy/visualize_variational_free_energy.py --save
uv run python extras/variational_free_energy/simulate_variational_free_energy.py --save
uv run python extras/variational_free_energy/interactive_variational_free_energy.py
uv run python extras/variational_free_energy/animation_variational_free_energy.py --save
```
