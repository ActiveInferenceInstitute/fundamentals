# Chapter 14 — concept map

Chapter 14 links active inference to Bayesian mechanics, ergodic density, and
Markov-blanket language. The companion implementation stays numerical and
testable: estimate an ergodic density, compute an entropy/VFE-style bound, and
inspect coupled external, blanket, and internal trajectories.

> **Implemented:** PDF-grounded Chapter 14 companion coverage with
> ergodic-density, viability, entropy-bound, and Markov-blanket helpers, four
> thin orchestrators, rendered PNG outputs, and raw NPZ/JSON sidecars.

## Script inventory

| File | Role |
|---|---|
| `example_14_1_ergodic_density.py` | Estimate an ergodic density from a deterministic OU teaching trajectory and show its entropy bound. |
| `example_14_2_survival_viability.py` | Map viability thresholds to binary viability indicators and survival probabilities. |
| `example_14_3_entropy_vfe_bounds.py` | Show entropy/VFE-style bound curves and residual gaps. |
| `example_14_4_bayesian_mechanics.py` | Demonstrate coupled external, blanket, and internal trajectories plus their correlation structure. |

## Library surface

`bayesian_mechanics_summary` wraps `ergodic_density`,
`density_entropy`, and `entropy_upper_bound_from_vfe` into a single
`BayesianMechanicsSummary`. `viability_indicator`,
`survival_probability`, `entropy_vfe_bound_curve`, and `phase1_fep_bridge`
cover survival/viability and Phase-I FEP bridge demos.
`simulate_markov_blanket_flow` returns a `MarkovBlanketFlow`, and
`blanket_coupling_matrix` summarizes the coupling among external, blanket,
and internal states.

## Where the book takes this next

The inspected PDF source spine ends at Chapter 14 and Appendices A-D. Future
Part III work should replace any teaching approximations with
manuscript-numbered worked examples as those equations are audited into tests.
