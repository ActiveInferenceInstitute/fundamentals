# chapters/ — Chapter Orchestrators

Each subdirectory corresponds to a chapter of the book and contains thin
orchestrator scripts that reproduce its figures and numerical examples.

## Structure

```
chapters/
├── README.md                  ← this file
├── AGENTS.md
├── chapter_01/                ← The Hypothesis-Testing Brain
│   ├── 01_box_scenario.py
│   ├── 02_three_perspectives.py
│   ├── 03_bayes_intuition.py
│   └── 04_inverse_problem.py
├── chapter_02/                ← Hidden State Estimation
│   ├── example_2_1_linear_deterministic.py
│   ├── example_2_2_linear_probabilistic.py
│   ├── example_2_3_precision.py
│   ├── example_2_4_nonlinear_deterministic.py
│   ├── example_2_5_nonlinear_probabilistic.py
│   ├── example_2_6_imperfect_model.py
│   ├── example_2_7_multiple_samples.py
│   ├── example_2_8_mle_analytic.py
│   ├── example_2_9_map_analytic.py
│   ├── example_2_10_gradient_descent.py
│   ├── visualize_generative_model.py
│   ├── interactive_explorer.py
│   ├── animation_sequential.py
│   └── animation_gradient_descent.py
├── chapter_03/                ← Combining Learning and Inference
│   ├── example_3_1_linear_regression_mle.py
│   ├── example_3_2_linear_regression_gd.py
│   ├── example_3_3_multiple_regression.py
│   ├── example_3_4_multivariate_gaussian.py
│   ├── example_3_5_bayesian_linear_regression.py
│   ├── example_3_6_lgs_food_localization.py
│   ├── example_3_7_factor_analysis_em.py
│   ├── animation_bimodal_emergence.py
│   ├── animation_blr_predictive_band.py
│   ├── animation_blr_tightening.py
│   ├── animation_em_convergence.py
│   ├── animation_em_steps.py
│   ├── animation_lgs_online.py
│   ├── animation_precision_sweep.py
│   ├── animation_sufficient_statistics.py
│   ├── visualize_calibration.py
│   ├── visualize_coverage.py
│   └── visualize_posterior_predictive.py
├── chapter_04/                ← Variational Bayesian Inference
│   ├── example_4_1_coordinate_search.py
│   ├── example_4_2_surprisal.py
│   ├── example_4_3_vfe_forms.py
│   ├── example_4_6_free_form_cavi.py
│   ├── example_4_7_fixed_form.py
│   ├── animation_vfe_descent.py
│   ├── visualize_kl_loss.py
│   ├── visualize_vfe_intuition.py
│   ├── visualize_model_comparison.py
│   └── interactive_vfe_explorer.py
├── chapter_05/                ← Predictive Coding
│   ├── example_5_1_prediction_errors.py
│   ├── example_5_3_multivariate.py
│   ├── example_5_4_recognition_dynamics.py
│   ├── example_5_7_hierarchical.py
│   ├── animation_recognition_descent.py
│   └── animation_hierarchical.py
├── chapter_06/                ← Generalized Filtering for Perception (Part II)
│   ├── example_6_1_generalized_filter.py
│   ├── example_6_2_multivariate_filter.py
│   ├── example_6_6_generalized_coordinates.py
│   ├── visualize_6_6_correlated_embedding_orders.py
│   └── example_6_7_multivariate_generalized_coordinates.py
├── chapter_07/                ← Active Generalized Filtering (Part II)
│   ├── example_7_2_active_inference.py
│   ├── example_7_5_multivariate_active_inference.py
│   └── animation_7_5_multivariate_active_inference.py
├── chapter_08/                ← Learning, attention, and hierarchy (Part II)
│   ├── example_8_1_learning_attention.py
│   ├── example_8_2_hierarchical_continuous.py
│   ├── visualize_message_passing.py
│   └── animation_learning_attention.py
├── chapter_09/                ← Active Inference in POMDPs (Part II)
│   ├── example_9_1_state_inference.py … example_9_6_exploration_exploitation.py
│   └── animation_belief_filtering.py · animation_efe_tradeoff.py
├── chapter_10/                ← Learning & extensions in POMDPs (Part II)
│   ├── example_10_1_learn_D.py … example_10_6_precision_learning.py   (§10.1–10.2)
│   ├── example_10_7_two_armed_bandit.py                              (§10.3 factorial)
│   ├── example_10_8_hierarchical.py                                  (§10.4 hierarchical)
│   ├── visualize_factorial_structure.py
│   └── animation_learning.py · animation_precision.py · animation_bandit.py
├── chapter_11/                ← Planning extensions (Part III)
│   └── example_11_1_free_energy_variants.py … example_11_4_hybrid_tree_structure.py
├── chapter_12/                ← Factor graphs and message passing (Part III)
│   └── example_12_1_factor_graph_messages.py … example_12_5_active_factor_learning_attention.py
├── chapter_13/                ← Robotics and social applications (Part III)
│   └── example_13_1_robotics_navigation.py … example_13_4_robotics_theory.py
└── chapter_14/                ← Bayesian mechanics and Markov blankets (Part III)
    └── example_14_1_ergodic_density.py … example_14_4_bayesian_mechanics.py
```

## Running a single script

```bash
uv run python chapters/chapter_01/03_bayes_intuition.py --save
uv run python chapters/chapter_02/example_2_2_linear_probabilistic.py
uv run python chapters/chapter_03/example_3_5_bayesian_linear_regression.py --save
```

All non-interactive scripts accept `--save` to write figures to
`output/figures/`; stochastic scripts also accept `--seed` for
reproducibility. Chapter 2/3 scripts additionally accept `--x-true`,
`--y-obs`, `--n-samples`, etc. — see each script's docstring or `--help`.

## Running all figures

The top-level [`run.sh`](../run.sh) menu is the simplest entry point:

```bash
./run.sh                        # interactive menu
./run.sh --all                  # every chapter, every script
./run.sh --chapter 3            # one chapter
./run.sh --no-animations        # skip GIF renderers
./run.sh --keep-going           # don't abort on first failure
```

The older batch pipeline still works:

```bash
uv run python scripts/run_all_figures.py              # all discovered chapters
uv run python scripts/run_all_figures.py --chapters 1
uv run python scripts/run_all_figures.py --chapters 4 5
uv run python scripts/run_all_figures.py --clean      # remove old generated media first
```

Shell shortcuts are also available:

```bash
./scripts/run_all_chapter_01.sh
./scripts/run_all_chapter_02.sh
./scripts/run_all_chapter_03.sh
```

## Conventions

- Every script is a **thin orchestrator**: it imports from `active_inference`
  and standard library, never from sibling scripts.
- All business logic lives in `src/active_inference/`.
- Each script ends with `if __name__ == "__main__": main()`.
- Default seeds are fixed so figures are deterministic on re-run.

## Smoke Tests

Chapter scripts are exercised by `tests/chapters/test_smoke.py`, which
runs every script with `--save` and asserts exit code 0. The
[`run.sh`](../run.sh) text menu uses the same discovery rules.

## Coverage

| Chapter   | Scripts                                                                  | Status |
|-----------|--------------------------------------------------------------------------|--------|
| Chapter 1 | 4 concept orchestrators                                                  | Complete |
| Chapter 2 | 14 (10 examples + 2 auxiliary + 2 animations)                            | Complete |
| Chapter 3 | 18 (7 examples + 8 animations + 3 diagnostic visualizations)             | Complete |
| Chapter 4 | 10 (5 examples + 1 animation + 3 visualizations + 1 interactive)         | Complete |
| Chapter 5 | 6 (4 examples + 2 animations)                                            | Complete |
| Chapter 6 | 5 (§6.1 univariate + §6.2 multivariate + §6.3–6.6 generalized/correlated coordinates) | Complete through §6.6 |
| Chapter 7 | 3 (§7.1–7.5 univariate + multivariate active generalized filtering)           | Complete through §7.5 |
| Chapter 8 | 4 (2 examples + 1 visualization + 1 animation; §8.1 learning/attention · §8.2–§8.6 hierarchy/message passing) | Partial continuous increment complete; nonlinear hierarchy extensions remain |
| Chapter 9 | 7 (5 examples + 2 animations; §9.1 state inference · §9.2–§9.3 dynamic filtering/VFE · §9.4–§9.6 EFE, Grid World, exploration/exploitation) | Complete for declared scope |
| Chapter 10 | 12 (8 examples + 1 visualization + 3 animations; §10.1 learning · §10.2 habit/precision · §10.3 factorial / two-armed bandit · §10.4 hierarchical) | Complete |
| Chapter 11 | 4 (free-energy variants, sophisticated planning, preference/habit learning, hybrid tree/structure scoring) | PDF-grounded Part III companion demos |
| Chapter 12 | 5 (factor-graph messages, smoothing, VMP/marginals, hybrid bridge, active-factor learning/attention) | PDF-grounded Part III companion demos |
| Chapter 13 | 4 (robotics navigation, fault-tolerant control, social inference, robotics theory landscape) | PDF-grounded Part III companion demos |
| Chapter 14 | 4 (ergodic density, survival/viability, entropy/VFE bounds, Bayesian mechanics) | PDF-grounded Part III companion demos |
