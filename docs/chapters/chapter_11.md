# Chapter 11 — concept map

Chapter 11 extends the Chapter 9-10 POMDP spine with planning variants:
time-dependent preferences, sophisticated inference over future beliefs,
bounded policy-tree search, parameter forgetting, and lightweight structure
learning. The companion implements these as small helpers in
`core.pomdp_extensions` and deterministic demos in
`estimators.pomdp_extensions`.

> **Implemented:** PDF-grounded Chapter 11 companion coverage with
> source-backed planning helpers, four thin orchestrators, rendered PNG outputs,
> and NPZ/JSON raw-data sidecars. This is a conservative Part III increment,
> not a claim that every manuscript worked example has been transcribed.

## Script inventory

| File | Role |
|---|---|
| `example_11_1_free_energy_variants.py` | Compare EFE, free energy of the future, generalized free energy, Bethe-style terms, and Renyi bounds over policy-indexed toy traces. |
| `example_11_2_sophisticated_planning.py` | Run bounded tree search in a line-world POMDP and display preference schedules, forgetting, and structure-learning diagnostics. |
| `example_11_3_preference_habit_learning.py` | Track state/time preferences, learned preference counts, and habit priors. |
| `example_11_4_hybrid_tree_structure.py` | Combine hybrid model evidence, path-based policy scoring, and structure posterior diagnostics. |

## Library surface

The core functions are `time_dependent_preferences`,
`forget_dirichlet_counts`, `structure_log_evidence`, `structure_posterior`,
`tree_policy_search`, `sophisticated_policy_trace`, `update_preference_counts`,
`habit_prior_from_counts`, and `path_based_policy_scores`. Their estimator-level
companions are `simulate_state_preference_schedule`,
`simulate_parameter_forgetting`, `simulate_structure_learning`,
`make_line_world`, `simulate_sophisticated_planning`,
`simulate_preference_habit_learning`, and `simulate_path_policy_computation`.

## Where the book takes this next

Chapter 12 moves from policy trees to graphical structure: factor graphs,
sum-product belief propagation, smoothing, variational message passing, and
hybrid discrete/continuous bridges.
