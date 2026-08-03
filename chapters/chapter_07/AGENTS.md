# `chapters/chapter_07/` — Active Generalized Filtering

Chapter 7 scripts add **action** to the Chapter 6 generalized filter, giving the
continuous-state formulation of active inference. The agent perceives *and* acts; both
descend the same variational free energy. The folder now runs through §7.5, including
multivariate active generalized filtering in generalized coordinates.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_7_2_active_inference.py`](example_7_2_active_inference.py) | ~95 | The coupled action-perception loop (Alg. 7.2.1): action drives the true state to the agent's preferred set-point against an exogenous force; reproduces Fig. 7.4.5. |
| [`example_7_5_multivariate_active_inference.py`](example_7_5_multivariate_active_inference.py) | ~60 | Thin wrapper over the `orchestrator_workflows` §7.5 builder: a 2-D agent uses generalized measurements and vector action to cancel an exogenous attractor. |
| [`animation_7_5_multivariate_active_inference.py`](animation_7_5_multivariate_active_inference.py) | ~50 | Thin wrapper over the `orchestrator_workflows` §7.5 animation builder: GIF showing 2-D path, belief, action vector, sensory error, and free-energy evolution. |

## Running

```bash
python chapters/chapter_07/example_7_2_active_inference.py --save
python scripts/run_all_figures.py --chapters 7
```

## Library Usage

```python
from active_inference import (
    ActiveEnvironment, ActiveInferenceAgent, DynamicStateSpaceModel, LinearFunction,
    action_gradient, perception_gradient, simulate_active_inference,
    MultivariateActiveEnvironment, MultivariateActiveInferenceAgent,
    multivariate_action_gradient, simulate_multivariate_active_inference,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs each script with `--save` and asserts exit 0,
in the single parametrized `test_chapter_script_runs_and_exports_raw_data`
(one case per discovered chapter script). Unit tests live in
`tests/core/test_active_inference.py` and
`tests/estimators/test_active_inference.py`.

## Key Concepts

- **Action through the sensory channel.** Free energy is not a function of `a` directly
  (action isn't in the generative model), so action descends `F` via the chain rule:
  `ȧ = −κ_a·(∂y/∂a)·λ_y·ε_y` (Eq. 9/11/17). `∂y/∂a` is the **forward model** — the agent's
  belief about the sensory consequence of acting (simplest: a constant gain/sign).
- **Preference prior / set-point.** The autonomous state `v=η` is encoded as the point
  attractor of the state-transition model `f_M(μ)=v−μ`. The agent *expects* to be at `v`
  and acts until its sensations match that expectation (a Bayesian thermostat).
- **Action-perception cycle.** The agent's action feeds back into the generative process,
  so environment and agent are simulated together (`simulate_active_inference`), unlike
  Chapter 6's pre-generated stream.
- **The defining property** (verified): with action, the true state is driven to the
  preferred set-point `v`; with action off (`κ_a=0`) it stays at the uncontrolled
  exogenous attractor `v*`. *Perception changes the model to match the world; action
  changes the world to match the model.*
- **Sign note.** The book's printed `f_E` action-coupling signs are OCR-ambiguous; the
  forward-model sign here (+1 for the `drift+a` coupling) is the internally-consistent
  choice, verified by the simulation converging to the set-point.
- **Vector action (§7.5).** `MultivariateActiveInferenceAgent` wraps a Chapter 6
  `GeneralizedVectorModel` with a forward model `∂ỹ/∂a`. The perception flow is
  `D μ̃ − κ_x∂F/∂μ̃`; the action flow is `−κ_a forward_model.T Π_y ε_y`.
  `simulate_multivariate_active_inference` keeps an action-off baseline so the figure
  can show action changing the world toward the preference rather than merely changing
  the belief.
