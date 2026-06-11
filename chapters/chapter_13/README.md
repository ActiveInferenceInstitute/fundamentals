# Chapter 13 - Applications

Chapter 13 applies active-inference primitives to robotics navigation and social
inference examples.

| Script | Mirrors | What it shows |
|---|---:|---|
| `example_13_1_robotics_navigation.py` | §13.1-§13.2 | A goal-directed trajectory, shrinking distance-to-goal, and increasing preference satisfaction. |
| `example_13_2_fault_tolerant_control.py` | §13.2 | Fault-compensated control under actuator efficacy loss. |
| `example_13_3_social_robotics.py` | §13.3 | Belief updates over another agent's hidden intention from communicative observations. |
| `example_13_4_robotics_theory.py` | §13.4 | Controllability, epistemic value, preference satisfaction, and combined robotics-theory score. |

## Running

```bash
uv run python chapters/chapter_13/example_13_1_robotics_navigation.py --save
uv run python chapters/chapter_13/example_13_2_fault_tolerant_control.py --save
uv run python chapters/chapter_13/example_13_3_social_robotics.py --save
uv run python chapters/chapter_13/example_13_4_robotics_theory.py --save
```
