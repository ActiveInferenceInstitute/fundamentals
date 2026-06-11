# Chapter 13 — concept map

Chapter 13 shifts from abstract algorithms to application demonstrations. The
companion starts this spine with deterministic robotics navigation and social
inference examples that are simple enough to test exactly while preserving the
active-inference pattern of preferences, beliefs, and action-oriented traces.

> **Implemented:** PDF-grounded Chapter 13 companion coverage with
> application-level estimator helpers, four thin orchestrators, rendered PNG
> outputs, and raw NPZ/JSON sidecars.

## Script inventory

| File | Role |
|---|---|
| `example_13_1_robotics_navigation.py` | A 2-D robot trajectory approaches a goal while the preference trace increases. |
| `example_13_2_fault_tolerant_control.py` | Deterministic control compensation under actuator efficacy loss. |
| `example_13_3_social_robotics.py` | Binary intention inference from communicative observations in a social-robotics toy example. |
| `example_13_4_robotics_theory.py` | Small robotics-theory landscape linking controllability, epistemic value, and preference satisfaction. |

## Library surface

`simulate_robot_navigation` returns a `NavigationResult` with path, goal,
distance, and preference traces. `simulate_fault_tolerant_control` returns a
`FaultTolerantControlResult` with desired, efficacy, action, compensated
output, and error traces. `simulate_social_inference` returns a
`SocialInferenceResult` with observations, belief history, and final inferred
intention. `robotics_theory_landscape` returns a `RoboticsTheoryResult` for
theory-level tradeoff plots.

## Where the book takes this next

Chapter 14 returns to foundations: ergodic density, entropy bounds,
thermodynamic/FEP bridges, Bayesian mechanics, and Markov blankets.
