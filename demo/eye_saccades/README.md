# Eye saccades demo

Discrete **expected free energy** planning on a retinotopic fixation grid (Chapter 9 POMDP machinery). Each grid cell is a candidate fixation; actions are saccade directions toward a salient target.

## Run

```bash
python demo/eye_saccades/visualize_eye_saccades.py --save
```

## Library APIs

- `make_gridworld`, `simulate_pomdp_agent` — `estimators/pomdp.py`
- `evaluate_policy` — `core/pomdp.py`

## Outputs

- Figure: `output/figures/demo/eye_saccades/visualize_eye_saccades.png`
- Data: `output/data/demo/eye_saccades/visualize_eye_saccades.{npz,json}`

## Limitations (v1)

No foveated likelihood, saccade kinematics, or eye–head plant. The grid is a pedagogical stand-in for fixation selection.
