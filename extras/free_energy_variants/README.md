# Free-Energy Variants

FEF, OFE, PFE, FEEF, generalized, Bethe, and Renyi forms.

## Book Mapping

- Family: Part III Extensions
- Chapters: 11
- Sections: 11.1, 11.1.1-11.1.8, D.4

## Scripts

- `visualize_free_energy_variants.py` - visualize orchestrator over `active_inference.extra_topics` and core APIs.
- `simulate_free_energy_variants.py` - simulate orchestrator over `active_inference.extra_topics` and core APIs.
- `interactive_free_energy_variants.py` - interactive slider orchestrator over `active_inference.extra_topics` and core APIs.

## Outputs

Run any script with `--save` to render artifacts under `output/figures/extras/free_energy_variants` and raw-data sidecars under `output/data/extras/free_energy_variants`.

```bash
uv run python extras/free_energy_variants/visualize_free_energy_variants.py --save
uv run python extras/free_energy_variants/simulate_free_energy_variants.py --save
uv run python extras/free_energy_variants/interactive_free_energy_variants.py
```
