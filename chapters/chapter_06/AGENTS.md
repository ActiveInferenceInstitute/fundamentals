# `chapters/chapter_06/` — Generalized Filtering for Perception

Chapter 6 scripts implement **online generalized filtering** (Part II): the static
predictive-coding belief of Chapter 5 becomes a dynamic state filtered in real time.
The agent's prior mean `m_x` is replaced by a **state-transition function** `f_M` that
predicts the state's flow; recognition is one Euler step of `μ̇_x = −κ ∂F/∂μ_x` per
time point. The folder now runs through §6.6, including correlated embedding-order
precision `Π̃(γ)` and Example 6.7 vector generalized filtering.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_6_1_generalized_filter.py`](example_6_1_generalized_filter.py) | ~90 | Univariate generalized filter (Alg. 6.1.1) tracks a moving hidden state from noisy obs; reproduces Fig. 6.1.3. |
| [`example_6_2_multivariate_filter.py`](example_6_2_multivariate_filter.py) | ~95 | Multivariate filter (§6.2) on a Hooke's-law oscillator; vector belief tracks the orbit with perception lag; reproduces Fig. 6.2.3. |
| [`example_6_6_generalized_coordinates.py`](example_6_6_generalized_coordinates.py) | ~95 | Generalized filtering in generalized coordinates (§6.5, Alg. 6.5.1); recovers position + velocity; the `D μ̃ − κ∂F` recognition flow. |
| [`visualize_6_6_correlated_embedding_orders.py`](visualize_6_6_correlated_embedding_orders.py) | ~75 | Correlated embedding-order precision (§6.6, Fig. 6.6.2); shows how `S(γ)^-1 ⊗ Π` changes with smoothness. |
| [`example_6_7_multivariate_generalized_coordinates.py`](example_6_7_multivariate_generalized_coordinates.py) | ~60 | Thin wrapper over the `orchestrator_workflows` §6.6 builder: vector generalized-coordinate filter with full correlated precision, compared against the ordinary multivariate filter. |

## Running

```bash
python chapters/chapter_06/example_6_1_generalized_filter.py --save
python scripts/run_all_figures.py --chapters 6
```

## Library Usage

```python
from active_inference import (
    DynamicProcess, DynamicStateSpaceModel, LinearFunction,
    simulate_dynamic_process, generalized_filter,
    gf_free_energy, gf_free_energy_grad, gf_fixed_point_linear,
    gaussian_temporal_covariance, correlated_embedding_precision,
    GeneralizedVectorModel, generalized_vector_filter,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs each script via `subprocess` with `--save`
and asserts exit code 0, in the single parametrized
`test_chapter_script_runs_and_exports_raw_data` (one case per discovered
chapter script). Unit tests for the methods live
in `tests/core/test_generalized_filtering.py` and
`tests/estimators/test_generalized_filtering.py`.

## Key Concepts

- **Dynamic generative model** (Eq. 10): `f_M(μ)=θ_x−μ` (state-transition / flow),
  `g_M(μ)=μ−θ_y` (observation), with state/sensory precisions `λ_x`, `λ_y`.
- **Derived gradient** `∂F/∂μ = λ_x ε_x(1−f_M'(μ)) − λ_y ε_y g_M'(μ)` — derived by
  chain rule (the book's printed Eq. 7b/14 sign convention is loose), finite-difference
  verified.
- **Why it tracks.** With `λ_y ≫ λ_x` the sensory term dominates, so the recognition
  fixed point `gf_fixed_point_linear` ≈ the state that explains `y` (i.e. `x*`). With
  equal precisions the estimate is *biased* — precision weighting is the whole story.
- **Euler-method recognition.** Hidden-state flow `μ̇_x = −κ ∂F/∂μ_x` integrated by
  Euler's method (Eq. 8); the online loop interleaves an environment step (transition +
  observe) and an agent step (one descent step).
- **Generalized coordinates (§6.3–§6.5).** The belief `μ̃_x = [μ_x, μ_x', μ_x'', …]` carries
  higher-order motions. The **derivative shift operator** `D` (`shift_operator`, Eq. 37)
  gives `Dμ̃ = μ̃'`. The generalized state error `ε̃_x = Dμ̃ − f̃(μ̃)` compares the actual
  generalized motion to the predicted flow — resolving §6.1's awkwardness. Recognition is
  `μ̃̇ = Dμ̃ − κ∂F/∂μ̃` (Eq. 51), gradient descent in a moving reference frame; at
  equilibrium `μ̃̇ = Dμ̃` (motion of the expectation = expectation of the motion). The
  gradient (Eq. 50a) uses the local-linearity approximation — *exact* for linear `f`/`g`,
  FD-verified. The payoff: the belief recovers the state's velocity/acceleration, not just
  its position.
- **Correlated embedding orders (§6.6).** `gaussian_temporal_covariance(M, gamma)`
  builds the temporal covariance `S(γ)` from the Gaussian derivative formula, and
  `correlated_embedding_precision(..., layout=...)` turns it into the book/order-major
  or repo state-major full precision. Example 6.7 combines this with a
  `GeneralizedVectorModel` so prediction errors, gradients, and exported traces use
  reconstructable `(time, embedding_order, variable)` arrays.
