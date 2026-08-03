# Chapter 14 — Bayesian mechanics and Markov blankets

Chapter 14 connects active inference to ergodic density, entropy bounds,
Markov blankets, and Bayesian mechanics. The concept map is
[`docs/chapters/chapter_14.md`](../../docs/chapters/chapter_14.md).

## Scripts

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

`--save` writes figures to `output/figures/chapter_14/` and raw NPZ+JSON
sidecars to `output/data/chapter_14/`.

## Programmatic usage

```python
from active_inference import (
    bayesian_mechanics_summary,
    ergodic_ou_trajectory,
)

trajectory = ergodic_ou_trajectory(n_steps=600, drift=0.06, diffusion=0.22)
summary = bayesian_mechanics_summary(trajectory, bins=70, vfe_margin=0.4)
print(summary.entropy, summary.upper_bound, summary.gap)
```

The ergodic/Bayesian-mechanics helpers live in `core.ergodic` and
`core.bayesian_mechanics`; the scripts wrap them with the book's §14.1–§14.4
teaching examples.
