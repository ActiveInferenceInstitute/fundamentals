# `chapters/chapter_14/` — agent guide

Chapter 14 scripts cover ergodic density, entropy bounds, Markov blankets, and
the Bayesian-mechanics bridge.

## Rules

- Keep bridge helpers in `active_inference.core.bayesian_mechanics` or
  `active_inference.core.ergodic`.
- Every saved script writes `save_chapter_data(14, ...)` sidecars.
- Avoid unsupported metaphysical claims; keep figures tied to explicit arrays.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_14_1_ergodic_density.py`](example_14_1_ergodic_density.py) | ~75 | An ergodic trajectory, normalized occupancy density, entropy, and a VFE-like upper bound. |
| [`example_14_2_survival_viability.py`](example_14_2_survival_viability.py) | ~50 | Viability thresholds and survival probability over a trajectory. |
| [`example_14_3_entropy_vfe_bounds.py`](example_14_3_entropy_vfe_bounds.py) | ~50 | Entropy/VFE bound curves and residual gaps. |
| [`example_14_4_bayesian_mechanics.py`](example_14_4_bayesian_mechanics.py) | ~70 | Coupled external, blanket, and internal states plus their correlation structure. |

## Library Usage

```python
from active_inference import (
    bayesian_mechanics_summary,
    ergodic_ou_trajectory,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs every discovered chapter script with
`--save` in the single parametrized `test_chapter_script_runs_and_exports_raw_data`.
Unit tests for the ergodic/Bayesian-mechanics helpers live in
`tests/core/test_ergodic.py` and `tests/core/test_bayesian_mechanics.py`.
