# Book Topic Coverage Matrix

This matrix is the audit contract for the repo-root `extras/` curriculum. It maps the attached Namjoshi manuscript source spine to live extras topic folders and reusable `src/active_inference` APIs. The grouping is concept-level: a topic can cover several neighboring book sections when they teach one pedagogical unit.

The "Artifact modes" column lists saveable rendered modes: `visualize`,
`simulate`, and `animation`. Simulation-capable topics also expose
`interactive_<topic>.py` GUI launchers through the same registry-backed source
methods; those wrappers are validated by
`scripts/validate_orchestrator_provenance.py` and tested through the shared
`interactive_topic_demo` constructor rather than by saved PNG/GIF output.

Source spine: `/Users/4d/Documents/Namjoshi_2025_v5_Fundamentals_of_Active_Inference.pdf` (1153 pages, LaTeX build dated 2025-03-20).

## Coverage By Family

### Foundations

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 1 | 1.1 | [`model_representation`](../../extras/model_representation/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize |
| 1, 14 | 1.2, 14.3 | [`order_and_surprisal`](../../extras/order_and_surprisal/) | `active_inference.core.ergodic`; `active_inference.extra_topics` | visualize, simulate |
| 1, 2 | 1.3, 2.1.4, C.2.4 | [`bayes_equation`](../../extras/bayes_equation/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 1, 2 | 1.4, 2.1.4 | [`inverse_problem`](../../extras/inverse_problem/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 2 | 2.1.2, 2.1.3, B.4 | [`generative_process_model`](../../extras/generative_process_model/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 2, 5, 10 | 2.2, 5.1, 10.2 | [`precision_weighting`](../../extras/precision_weighting/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 2 | 2.2, 2.3 | [`hidden_state_estimation`](../../extras/hidden_state_estimation/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |

### Statistical Estimation

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 2 | 2.3 | [`multiple_samples`](../../extras/multiple_samples/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 2 | 2.5.1 | [`mle_map`](../../extras/mle_map/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 2, 3 | 2.5.2, 3.1 | [`gradient_descent`](../../extras/gradient_descent/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 3 | 3.1, 3.2 | [`linear_regression`](../../extras/linear_regression/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 3 | 3.3 | [`bayesian_linear_regression`](../../extras/bayesian_linear_regression/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 3, 6 | 3.4, C.10 | [`multivariate_gaussians`](../../extras/multivariate_gaussians/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 3 | 3.4 | [`linear_gaussian_systems`](../../extras/linear_gaussian_systems/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 3 | 3.5 | [`expectation_maximization`](../../extras/expectation_maximization/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |

### Information And Variational Inference

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 4, 14 | C.10.4, 14.3 | [`entropy`](../../extras/entropy/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 4 | 4.1, C.10.5 | [`kl_divergence`](../../extras/kl_divergence/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 4 | 4.2, 4.3 | [`surprisal_evidence`](../../extras/surprisal_evidence/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 4 | 4.2, 4.3, D.1 | [`variational_free_energy`](../../extras/variational_free_energy/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 4 | 4.5, 4.6 | [`mean_field_variational_inference`](../../extras/mean_field_variational_inference/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 4, 12 | 4.5, 12.4 | [`cavi`](../../extras/cavi/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate, animation |
| 4 | 4.4, C.11.1 | [`model_comparison`](../../extras/model_comparison/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |

### Predictive Coding And Continuous Dynamics

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 5 | 5.1, 5.2 | [`predictive_coding`](../../extras/predictive_coding/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 5, 8 | 5.4, 8.3 | [`hierarchical_predictive_coding`](../../extras/hierarchical_predictive_coding/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 6 | 6.1, 6.2 | [`generalized_filtering`](../../extras/generalized_filtering/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 6 | 6.3, 6.5, 6.6 | [`generalized_coordinates`](../../extras/generalized_coordinates/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |

### Active Inference Core

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 7 | 7.2, 7.4, 7.5 | [`active_generalized_filtering`](../../extras/active_generalized_filtering/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 8 | 8.1 | [`learning_attention`](../../extras/learning_attention/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 8, 12 | 8.5, 12.5 | [`hierarchical_message_passing`](../../extras/hierarchical_message_passing/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate, animation |

### Discrete POMDP Active Inference

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 9, 10 | 9.1, B.10 | [`pomdp_arrays`](../../extras/pomdp_arrays/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 9 | 9.2 | [`discrete_belief_filtering`](../../extras/discrete_belief_filtering/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 9 | 9.3 | [`discrete_vfe`](../../extras/discrete_vfe/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 9 | 9.4, 9.5 | [`gridworld_control`](../../extras/gridworld_control/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 9 | 9.5, D.3 | [`expected_free_energy`](../../extras/expected_free_energy/) | `active_inference.core.free_energy_forms`; `active_inference.extra_topics` | visualize, simulate |
| 9, 10 | 9.6, 10.1 | [`exploration_exploitation`](../../extras/exploration_exploitation/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |

### Learning And Depth

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 10 | 10.1 | [`dirichlet_learning`](../../extras/dirichlet_learning/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 10 | 10.2 | [`policy_precision_habits`](../../extras/policy_precision_habits/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 10 | 10.3, 12.6 | [`factorial_depth`](../../extras/factorial_depth/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate |
| 10, 12 | 10.4, 12.6 | [`hierarchical_depth`](../../extras/hierarchical_depth/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate |

### Part III Extensions

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 11 | 11.1, 11.1.1-11.1.8, D.4 | [`free_energy_variants`](../../extras/free_energy_variants/) | `active_inference.core.free_energy_forms`; `active_inference.extra_topics` | visualize, simulate |
| 11 | 11.2.1 | [`sophisticated_inference`](../../extras/sophisticated_inference/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 11 | 11.2.2 | [`inductive_planning`](../../extras/inductive_planning/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 11 | 11.2.3, 11.2.5 | [`state_preferences`](../../extras/state_preferences/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 11 | 11.2.6, 11.2.7 | [`parameter_uncertainty`](../../extras/parameter_uncertainty/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 11, 12 | 11.2.9, 12.3 | [`backward_smoothing`](../../extras/backward_smoothing/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate, animation |
| 11, 12 | 11.3, 12.6 | [`hybrid_generative_models`](../../extras/hybrid_generative_models/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |
| 11 | 11.2.8, 11.4 | [`tree_policy_search`](../../extras/tree_policy_search/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 11 | 11.5 | [`structure_learning`](../../extras/structure_learning/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |

### Factor Graphs And Applications

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 12 | 12.1, 12.5 | [`factor_graphs`](../../extras/factor_graphs/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate |
| 12 | 12.2, 12.3 | [`belief_propagation`](../../extras/belief_propagation/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate, animation |
| 12 | 12.4 | [`variational_message_passing`](../../extras/variational_message_passing/) | `active_inference.core.factor_graph`; `active_inference.extra_topics` | visualize, simulate |
| 13 | 13.1, 13.2 | [`robotics_navigation`](../../extras/robotics_navigation/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate, animation |
| 13 | 13.3 | [`social_robotics`](../../extras/social_robotics/) | `active_inference.extra_topics`; `src/active_inference/core` | visualize, simulate |

### Thermodynamic/FEP Bridge

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 14 | 14.1, 14.2 | [`ergodic_density`](../../extras/ergodic_density/) | `active_inference.core.ergodic`; `active_inference.extra_topics` | visualize, simulate, animation |
| 14 | 14.3 | [`fep_entropy_bounds`](../../extras/fep_entropy_bounds/) | `active_inference.core.ergodic`; `active_inference.extra_topics` | visualize, simulate |
| 4, 14 | D.3, 14.3 | [`temperature`](../../extras/temperature/) | `active_inference.core.thermodynamics`; `active_inference.extra_topics` | visualize, simulate |
| 4, 14 | D.4 | [`enthalpy`](../../extras/enthalpy/) | `active_inference.core.thermodynamics`; `active_inference.extra_topics` | visualize, simulate |
| 14 | 14.1, 14.4 | [`bayesian_mechanics_bridge`](../../extras/bayesian_mechanics_bridge/) | `active_inference.core.ergodic`; `active_inference.extra_topics` | visualize, simulate |

## PDF Source Spine Contract

Source spine: `/Users/4d/Documents/Namjoshi_2025_v5_Fundamentals_of_Active_Inference.pdf`
has 1153 pages, LaTeX build date 2025-03-20, Chapters 1-14, and Appendices
A-D. It has no Chapter 15; `chapter_15` must remain absent unless a different
source manuscript is supplied.

### Part III Explicit Subsection Coverage

| Book chapters | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| 11 | 11.1, 11.1.1, 11.1.2, 11.1.3, 11.1.4, 11.1.5, 11.1.6, 11.1.7, 11.1.8, D.4 | [`free_energy_variants`](../../extras/free_energy_variants/) | `active_inference.core.free_energy_forms` | visualize, simulate |
| 11 | 11.2.4 | [`preference_habit_learning`](../../extras/preference_habit_learning/) | `active_inference.core.pomdp_extensions`; `active_inference.estimators.pomdp_extensions` | visualize |
| 11 | 11.2.8, 11.4 | [`tree_policy_search`](../../extras/tree_policy_search/) | `active_inference.core.pomdp_extensions`; `active_inference.estimators.pomdp_extensions` | visualize, simulate, animation |
| 12 | 12.4, 12.4.1 | [`variational_message_passing`](../../extras/variational_message_passing/) | `active_inference.core.factor_graph` | visualize, simulate |
| 12 | 12.7 | [`hierarchical_message_passing`](../../extras/hierarchical_message_passing/) | `active_inference.core.factor_graph`; `active_inference.core.continuous_learning` | visualize, simulate, animation |
| 13 | 13.4 | [`robotics_theory`](../../extras/robotics_theory/) | `active_inference.estimators.applications` | visualize |

### Appendix A Historical/Future Coverage

| Appendix | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| A | A.1, A.1.1, A.1.2, A.1.3, A.1.4, A.1.5, A.1.6, A.1.7 | [`active_inference_history`](../../extras/active_inference_history/) | `active_inference.source_spine` | visualize |
| A | A.2, A.2.5 | [`active_inference_future`](../../extras/active_inference_future/) | `active_inference.source_spine` | visualize |
| A | A.2.1 | [`deep_generative_models`](../../extras/deep_generative_models/) | `active_inference.source_spine` | visualize |
| A | A.2.2 | [`cybernetics_control`](../../extras/cybernetics_control/) | `active_inference.source_spine` | visualize |
| A | A.2.3 | [`information_theory_lineage`](../../extras/information_theory_lineage/) | `active_inference.source_spine` | visualize |
| A | A.2.4 | [`reinforcement_learning_lineage`](../../extras/reinforcement_learning_lineage/) | `active_inference.source_spine` | visualize |

### Appendix B-D Executable Math Coverage

| Appendix | Sections | Extras topic | Reusable APIs | Artifact modes |
|---|---|---|---|---|
| B | B.1, B.2, B.3, B.4, B.5, B.6, B.7, B.8, B.9, B.10, B.11, B.12 | [`appendix_notation_model_setup`](../../extras/appendix_notation_model_setup/) | `active_inference.core.appendix_math`; `active_inference.core.pomdp`; `active_inference.core.factor_graph` | visualize |
| C | C.1, C.1.1, C.1.2, C.1.3, C.1.4, C.2, C.2.1, C.2.2, C.2.2.1, C.2.2.2, C.2.2.3, C.2.2.4, C.2.2.5, C.2.3, C.2.4, C.2.5, C.2.6, C.3, C.3.1, C.3.2, C.4, C.4.1, C.4.2, C.4.3, C.4.4, C.5, C.5.1, C.5.2, C.5.3, C.5.4, C.6, C.6.1, C.6.2, C.7, C.7.1, C.7.2, C.7.3, C.8, C.8.1, C.8.2, C.8.3, C.10, C.10.1, C.10.2, C.10.3, C.10.4, C.10.5, C.10.6, C.10.7, C.12, C.13, C.13.1, C.13.2 | [`appendix_math_fundamentals`](../../extras/appendix_math_fundamentals/) | `active_inference.core.appendix_math` | visualize |
| C | C.9 | [`colored_noise`](../../extras/colored_noise/) | `active_inference.core.noise` | visualize |
| C | C.11, C.11.1, C.11.2, C.11.3, C.11.4 | [`bayesian_model_selection`](../../extras/bayesian_model_selection/) | `active_inference.core.model_comparison` | visualize |
| D | D.1, D.2, D.3, D.3.1, D.3.2, D.3.3, D.3.4, D.4 | [`appendix_free_energy_forms`](../../extras/appendix_free_energy_forms/) | `active_inference.core.free_energy_forms` | visualize |

## Validation

Run the matrix validator after changing topic registry entries or extras folders:

```bash
uv run python scripts/validate_book_topic_coverage.py
uv run python scripts/validate_book_topic_coverage.py --require-rendered
uv run python scripts/validate_source_spine.py --require-pdf
```

The default validator requires each registered topic to have a folder, README, static script, declared simulation/animation scripts, and one coverage-matrix row. The `--require-rendered` mode is the post-render gate: every declared static, simulation, and animation wrapper must also have its expected PNG/GIF artifact plus paired NPZ+JSON sidecars.
