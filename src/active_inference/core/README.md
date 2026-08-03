# `core/` — mathematical primitives

Distributions, generative process / model classes, exact grid Bayesian
inference, and closed-form Linear Gaussian System posteriors. Everything in
the rest of the package builds on this layer.

## Files

| File | What it defines |
|---|---|
| [`distributions.py`](distributions.py) | Univariate (`gaussian_pdf`, `gaussian_log_pdf`, `uniform_pdf`, `dirac_like_pdf`, `normalize_density`) and multivariate (`mvn_pdf`, `mvn_log_pdf`, `mvn_sample`, `mahalanobis_squared`, `isotropic_cov`, `diagonal_cov`) helpers. |
| [`generative_process.py`](generative_process.py) | `GenerativeProcess` (abstract), `LinearGaussianProcess` (univariate), `LinearGaussianMVProcess` (multivariate). |
| [`generative_model.py`](generative_model.py) | `GenerativeModel` (abstract), `LinearGaussianModel` (univariate), `LinearGaussianMVModel` (multivariate). |
| [`inference.py`](inference.py) | `GridBayesianInference` + `InferenceResult` — exact posterior on a 1-D grid with trapezoid normalization. |
| [`lgs.py`](lgs.py) | `LinearGaussianSystem` + `LGSPosterior` — closed-form multivariate posterior. |
| [`variational.py`](variational.py) | **(Ch.4)** `GaussianBelief`, `variational_free_energy` → `VFEComponents` (the five VFE forms), `vfe_g_form`/`vfe_d_form`/`vfe_c_form`/`vfe_e_form`/`vfe_map_form`/`vfe_mle_form`, `log_model_evidence`, `surprisal`, `free_energy_bound_gap`. |
| [`thermodynamics.py`](thermodynamics.py) | `ThermodynamicState`, canonical probabilities, entropy/energy potentials, and `vfe_thermodynamic_state` for the explicit FEP thermodynamic bridge used by extras. |
| [`predictive_coding.py`](predictive_coding.py) | **(Ch.5)** `PredictiveCodingModel`, `GenerativeFunction`/`LinearFunction`/`QuadraticFunction`/`GenericFunction`, `PCFreeEnergy`, `predictive_coding_free_energy`, `pc_free_energy_grad`(`_fd`), `pc_linear_fixed_point`, `pc_curvature_linear`, `sensory_prediction_error`, `state_prediction_error`. |
| [`generalized_filtering.py`](generalized_filtering.py) | **(Ch.6)** §6.1–§6.6 generalized filtering in ordinary and generalized coordinates: `DynamicStateSpaceModel`, `gf_*` filters, `shift_operator` / `embed_flow`, `GeneralizedModel`, `generalized_*_error`, and `correlated_embedding_precision`. |
| [`active_inference.py`](active_inference.py) | **(Ch.7)** `ActiveInferenceAgent`, `action_gradient`, `perception_gradient`; `MultivariateActiveInferenceAgent`, `multivariate_action_gradient` for active generalized filtering in generalized coordinates. |
| [`continuous_learning.py`](continuous_learning.py) | **(Ch.8)** `LearningAttentionModel`, `LearningAttentionState`, `learning_attention_free_energy`, `learning_attention_grad`(`_fd`), log-precision transforms, `HierarchicalContinuousModel`, and message-passing terms. |
| [`pomdp.py`](pomdp.py) | **(Ch.9–10)** `POMDPModel`, exact state inference, EFE components, policy/action posteriors, forward filtering, discrete VFE (Ch.9); Dirichlet `A`/`B`/`D` learning, novelty, habit/policy precision, and factorial/hierarchical depth (Ch.10). |
| [`pomdp_extensions.py`](pomdp_extensions.py) | **(Ch.11)** Part III planning extensions: tree policy search, sophisticated inference traces, time-dependent preferences, parameter forgetting, structure learning, and path-based policy scores. |
| [`diagnostics.py`](diagnostics.py) | Statistical diagnostics — calibration/coverage/CRPS/log-score, entropies & KLs, `posterior_predictive_check`, and the Ch.5 validation helpers `gradient_check`, `convergence_report`, `oracle_agreement`. |
| [`free_energy_forms.py`](free_energy_forms.py) | Pedagogical free-energy variants: EFE, FEF, observed/predicted FE, GFE, Bethe-style FE, Renyi bound, and comparison tables. |
| [`factor_graph.py`](factor_graph.py) | Categorical factor-graph message helpers: message normalization, sum-product factor messages, chain filtering, and VMP-style updates. |
| [`ergodic.py`](ergodic.py) | Ergodic density, differential entropy, entropy-bound residuals, and a deterministic OU teaching trajectory. |
| [`bayesian_mechanics.py`](bayesian_mechanics.py) | **(Ch.14)** `BayesianMechanicsSummary`, `MarkovBlanketFlow`, blanket coupling, viability/survival indicators, entropy–VFE bounds, and the `phase1_fep_bridge`. |
| [`appendix_math.py`](appendix_math.py) | **(Appendix B/C)** `normalize_categorical`, `joint_from_likelihood_prior`, `marginalize`, `bayes_posterior_from_likelihood`, `gamma_pdf`, `dirichlet_mean`/`dirichlet_variance`, `mutual_information`, `maximum_entropy_distribution`, `euler_integrate`, `jensen_gap`, `kronecker_delta`, `dirac_delta_grid`. |
| [`model_comparison.py`](model_comparison.py) | **(Appendix C.11)** `model_posterior`, `bayes_factor`, `log_bayes_factor`, `bayesian_model_average`, `bayesian_model_reduction`, `bayesian_model_expansion`. |
| [`noise.py`](noise.py) | **(Appendix C.9)** `squared_exponential_covariance`, `colored_noise_precision`, `sample_colored_noise`, `finite_difference_derivative`. |
| [`compose.py`](compose.py) | `Pipeline` (process + model wiring), `running_stats` / `RunningPosteriorStats`. |
| [`posterior.py`](posterior.py) | `Posterior` protocol + `summarize_posterior` and accessors that dispatch across grid / LGS / BLR posteriors. |
| [`validators.py`](validators.py) | `require_*` defensive runtime checks (shapes, finiteness, ranges). |
| [`types.py`](types.py) | Shape aliases + `assert_cov` / `assert_probabilities`. |
| `__init__.py` | Re-exports the public surface. |

## Public API

```python
from active_inference.core.distributions import (
    gaussian_pdf, gaussian_log_pdf, uniform_pdf, dirac_like_pdf,
    mvn_pdf, mvn_log_pdf, mvn_sample, mahalanobis_squared,
    isotropic_cov, diagonal_cov, normalize_density,
)
from active_inference.core.generative_process import (
    GenerativeProcess, LinearGaussianProcess, LinearGaussianMVProcess,
)
from active_inference.core.generative_model import (
    GenerativeModel, LinearGaussianModel, LinearGaussianMVModel,
)
from active_inference.core.inference import GridBayesianInference, InferenceResult
from active_inference.core.lgs import LinearGaussianSystem, LGSPosterior
from active_inference.core.variational import (        # Chapter 4
    GaussianBelief, variational_free_energy, VFEComponents,
    log_model_evidence, surprisal,
)
from active_inference.core.thermodynamics import (
    ThermodynamicState, canonical_probabilities, vfe_thermodynamic_state,
)
from active_inference.core.predictive_coding import (  # Chapter 5
    PredictiveCodingModel, LinearFunction, QuadraticFunction,
    predictive_coding_free_energy, pc_free_energy_grad, pc_linear_fixed_point,
)
from active_inference.core.generalized_filtering import (  # Chapter 6
    GeneralizedVectorModel, correlated_embedding_precision,
    generalized_vector_free_energy_grad,
)
from active_inference.core.active_inference import (  # Chapter 7
    MultivariateActiveInferenceAgent, multivariate_action_gradient,
)
from active_inference.core.continuous_learning import (  # Chapter 8
    LearningAttentionModel, LearningAttentionState,
    learning_attention_free_energy, learning_attention_grad,
)
from active_inference.core.diagnostics import (
    gradient_check, convergence_report, oracle_agreement,
)
from active_inference.core.free_energy_forms import (
    expected_free_energy_form, free_energy_variant_table,
)
from active_inference.core.factor_graph import normalize_message, sum_product_chain
from active_inference.core.ergodic import ergodic_density, entropy_upper_bound_from_vfe
```

(The `core.distributions`, `compose`, `posterior`, `validators`, and `types`
helpers are also re-exported at the package top level — see
[`docs/reference/core.md`](../../../docs/reference/core.md) for the full catalogue.)

## Design Decisions

- **Vectorized.** Shapes are designed so that any of `x`, `mu`, or `sigma2`
  may broadcast — crucial for grid-based inference.
- **Variances everywhere.** Code uses `sigma2`, `s2_x`, `cov_y` (variance /
  covariance), never standard deviations.
- **Cholesky for MVN.** Multivariate density and sampling use Cholesky-based
  solves rather than explicit matrix inverses.
- **Explicit RNG.** Random generators are passed via `rng: np.random.Generator`.
- **Log-space inference.** `GridBayesianInference` works in log-space and
  subtracts the max log-density before exponentiating to avoid under/overflow.
- **Result dataclasses.** `InferenceResult` and `LGSPosterior` provide
  computed properties (mode, mean, variance, credible interval, std,
  precision) so downstream code never reimplements them.

## Dependencies

`numpy` (everywhere) + `scipy.linalg.solve_triangular` (only inside
`mvn_log_pdf` for stability).

## Testing

One test file per module under `tests/core/`: `test_distributions.py`,
`test_distributions_mvn.py`, `test_generative_process.py`,
`test_generative_model.py`, `test_inference.py`, `test_lgs.py`,
`test_variational.py`, `test_predictive_coding.py`, `test_diagnostics.py`,
`test_thermodynamics.py`, `test_free_energy_forms.py`, `test_factor_graph.py`,
`test_ergodic.py`, `test_continuous_learning.py`, `test_compose.py`,
`test_posterior.py`, `test_validators.py`, `test_types.py`.
