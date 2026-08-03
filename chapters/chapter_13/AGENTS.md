# `chapters/chapter_13/` — agent guide

Chapter 13 contains application-level active-inference demos for robotics and
social inference.

## Rules

- Keep application simulation logic in `active_inference.estimators.applications`.
- Every saved script must write raw sidecars through `save_chapter_data(13, ...)`.
- Prefer deterministic examples unless a stochastic seed is required.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_13_1_robotics_navigation.py`](example_13_1_robotics_navigation.py) | ~70 | A goal-directed trajectory, shrinking distance-to-goal, and increasing preference satisfaction. |
| [`example_13_2_fault_tolerant_control.py`](example_13_2_fault_tolerant_control.py) | ~50 | Fault-compensated control under actuator efficacy loss. |
| [`example_13_3_social_robotics.py`](example_13_3_social_robotics.py) | ~70 | Belief updates over another agent's hidden intention from communicative observations. |
| [`example_13_4_robotics_theory.py`](example_13_4_robotics_theory.py) | ~50 | Controllability, epistemic value, preference satisfaction, and combined robotics-theory score. |

## Library Usage

```python
from active_inference import (
    simulate_fault_tolerant_control,
    simulate_robot_navigation,
    simulate_social_inference,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs every discovered chapter script with
`--save` in the single parametrized `test_chapter_script_runs_and_exports_raw_data`.
Unit tests for the application simulations live in `tests/estimators/test_applications.py`.
