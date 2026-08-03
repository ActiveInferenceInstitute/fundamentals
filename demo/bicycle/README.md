# Riding a bicycle demo

**Balance metaphor** using multivariate active inference (Chapter 7.5 vector plant) to drive a 2-D state toward an upright preference, plus a fault-tolerant control trace (Chapter 13) as compensation after a wobble.

## Run

```bash
python demo/bicycle/visualize_bicycle.py --save
```

## Library APIs

- `build_multivariate_active_agent_env` — `orchestrator_workflows.py`
- `simulate_multivariate_active_inference` — `estimators/active_inference.py`
- `simulate_fault_tolerant_control` — `estimators/applications.py`

## Outputs

- Figure: `output/figures/demo/bicycle/visualize_bicycle.png`
- Data: `output/data/demo/bicycle/visualize_bicycle.{npz,json}`

## Limitations (v1)

Not a bicycle or inverted-pendulum plant; the Hooke-style vector dynamics stand in for roll/lean stabilization.
