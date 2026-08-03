# `chapters/chapter_11/` — agent guide

Chapter 11 scripts promote Part III planning extensions from extras sketches into
thin, tested chapter orchestrators.

## Rules

- Keep scripts as wrappers around `active_inference` APIs.
- Preserve `--save`; stochastic additions must also accept `--seed`.
- Pair every saved figure with `save_chapter_data(11, ...)`.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_11_1_free_energy_variants.py`](example_11_1_free_energy_variants.py) | ~90 | EFE / FEF / GFE / Renyi-style certainty-equivalent teaching curves over policies. |
| [`example_11_2_sophisticated_planning.py`](example_11_2_sophisticated_planning.py) | ~90 | Bounded tree policy search, future-belief entropy, time-dependent preferences, forgetting, and structure-posterior diagnostics. |
| [`example_11_3_preference_habit_learning.py`](example_11_3_preference_habit_learning.py) | ~70 | Preference pseudocount learning and habit-prior construction. |
| [`example_11_4_hybrid_tree_structure.py`](example_11_4_hybrid_tree_structure.py) | ~70 | Hybrid model evidence, path-policy scores, and structure posteriors. |

## Library Usage

```python
from active_inference import (
    simulate_parameter_forgetting,
    simulate_sophisticated_planning,
    simulate_state_preference_schedule,
    simulate_structure_learning,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs every discovered chapter script with
`--save` in the single parametrized `test_chapter_script_runs_and_exports_raw_data`.
Unit tests for the Part III helpers live in `tests/core/test_pomdp_extensions.py`
and `tests/estimators/test_pomdp_extensions.py`.
