# Chapter 14 - FEP and Bayesian mechanics bridge

Chapter 14 connects active inference to ergodic density, entropy bounds,
Markov blankets, and Bayesian mechanics.

| Script | Mirrors | What it shows |
|---|---:|---|
| `example_14_1_ergodic_density.py` | §14.1-§14.3 | An ergodic trajectory, normalized occupancy density, entropy, and a VFE-like upper bound. |
| `example_14_2_survival_viability.py` | §14.2 | Viability thresholds and survival probability over a trajectory. |
| `example_14_3_entropy_vfe_bounds.py` | §14.3 | Entropy/VFE bound curves and residual gaps. |
| `example_14_4_bayesian_mechanics.py` | §14.4 | Coupled external, blanket, and internal states plus their correlation structure. |

## Running

```bash
uv run python chapters/chapter_14/example_14_1_ergodic_density.py --save
uv run python chapters/chapter_14/example_14_2_survival_viability.py --save
uv run python chapters/chapter_14/example_14_3_entropy_vfe_bounds.py --save
uv run python chapters/chapter_14/example_14_4_bayesian_mechanics.py --save
```
