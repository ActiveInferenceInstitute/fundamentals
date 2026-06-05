# Enthalpy

H = U + pV and G = H - T S as explicit analogy-layer quantities.

## Book Mapping

- Family: Thermodynamic/FEP Bridge
- Chapters: 4, 14
- Sections: D

## Scripts

- `visualize_enthalpy.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_enthalpy.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_enthalpy.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/enthalpy` and raw-data sidecars under `output/data/extras/enthalpy`.

```bash
uv run python extras/enthalpy/visualize_enthalpy.py --save
uv run python extras/enthalpy/simulate_enthalpy.py --save
uv run python extras/enthalpy/interactive_enthalpy.py
```
