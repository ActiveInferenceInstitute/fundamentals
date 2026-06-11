"""Registry-driven extras topic demos beyond the chapter spine.

The public functions here power the repo-root ``extras/`` wrappers. They keep
topic scripts thin by centralizing pedagogical data generation, plotting, and
raw-data export behavior in the tested library layer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .core.distributions import gaussian_pdf, normalize_density
from .core.ergodic import (
    density_entropy,
    entropy_upper_bound_from_vfe,
    ergodic_density,
    ergodic_ou_trajectory,
)
from .core.factor_graph import (
    categorical_factor_message,
    sum_product_chain,
    variational_message_update,
)
from .core.free_energy_forms import free_energy_variant_table, renyi_bound
from .core.appendix_math import (
    dirichlet_mean,
    gamma_pdf,
    maximum_entropy_distribution,
    mutual_information,
)
from .core.model_comparison import model_posterior
from .core.noise import sample_colored_noise, squared_exponential_covariance
from .source_spine import APPENDICES
from .core.thermodynamics import (
    boltzmann_entropy,
    canonical_probabilities,
    expected_energy,
    helmholtz_free_energy,
)
from .core.variational import GaussianBelief
from .utils import default_figure_dir, ensure_dir, save_extra_data
from .visualizations.animations import save_animation
from .visualizations.style import COLORS, annotate_stat_box


SOURCE_APIS_BY_DEMO_KIND: dict[str, tuple[str, ...]] = {
    "foundation": (
        "active_inference.core.distributions.gaussian_pdf",
        "active_inference.core.distributions.normalize_density",
        "active_inference.core.inference.GridBayesianInference",
    ),
    "estimation": (
        "active_inference.estimators.gradient_descent.gradient_descent",
        "active_inference.estimators.mle.mle_loss",
        "active_inference.estimators.map.map_loss",
    ),
    "information": (
        "active_inference.core.diagnostics.grid_entropy",
        "active_inference.core.diagnostics.grid_kl_divergence",
        "active_inference.core.thermodynamics.canonical_probabilities",
    ),
    "variational": (
        "active_inference.core.variational.GaussianBelief",
        "active_inference.core.variational.variational_free_energy",
        "active_inference.core.variational.log_model_evidence",
    ),
    "continuous": (
        "active_inference.core.predictive_coding.predictive_coding_free_energy",
        "active_inference.core.generalized_filtering.generalized_free_energy",
    ),
    "pomdp": (
        "active_inference.core.pomdp.infer_states",
        "active_inference.core.pomdp.expected_free_energy",
        "active_inference.core.pomdp.policy_posterior",
    ),
    "learning": (
        "active_inference.core.pomdp.dirichlet_expected_value",
        "active_inference.core.pomdp.parameter_novelty",
        "active_inference.estimators.pomdp.simulate_learning_agent",
    ),
    "extension": (
        "active_inference.core.free_energy_forms.free_energy_variant_table",
        "active_inference.core.free_energy_forms.renyi_bound",
    ),
    "factor_graph": (
        "active_inference.core.factor_graph.sum_product_chain",
        "active_inference.core.factor_graph.categorical_factor_message",
        "active_inference.core.factor_graph.variational_message_update",
    ),
    "application": (
        "active_inference.estimators.pomdp.make_gridworld",
        "active_inference.estimators.pomdp.simulate_pomdp_agent",
    ),
    "thermo": (
        "active_inference.core.thermodynamics.canonical_probabilities",
        "active_inference.core.thermodynamics.helmholtz_free_energy",
        "active_inference.core.thermodynamics.enthalpy",
    ),
    "ergodic": (
        "active_inference.core.ergodic.ergodic_density",
        "active_inference.core.ergodic.density_entropy",
        "active_inference.core.ergodic.entropy_upper_bound_from_vfe",
    ),
    "history": (
        "active_inference.source_spine.APPENDICES",
        "active_inference.source_spine.appendix_section_ids",
    ),
    "appendix_math": (
        "active_inference.core.appendix_math.gamma_pdf",
        "active_inference.core.appendix_math.dirichlet_mean",
        "active_inference.core.appendix_math.mutual_information",
    ),
    "model_comparison": (
        "active_inference.core.model_comparison.model_posterior",
        "active_inference.core.model_comparison.bayes_factor",
    ),
    "noise": (
        "active_inference.core.noise.squared_exponential_covariance",
        "active_inference.core.noise.sample_colored_noise",
    ),
}


@dataclass(frozen=True)
class ExtraTopicSpec:
    """Metadata contract for one extras topic folder.

    Parameters
    ----------
    slug : str
        Directory name under ``extras/``.
    title : str
        Human-readable display title.
    family : str
        Curriculum grouping used by docs, menu, and web UI.
    chapters : tuple[int, ...]
        Book chapter numbers primarily connected to the topic.
    sections : tuple[str, ...]
        Book section identifiers or appendix references.
    summary : str
        One-sentence pedagogical purpose.
    demo_kind : str
        Internal data-generation family.
    has_simulation : bool
        Whether ``simulate_<topic>.py`` should exist.
    has_animation : bool
        Whether ``animation_<topic>.py`` should exist.
    """

    slug: str
    title: str
    family: str
    chapters: tuple[int, ...]
    sections: tuple[str, ...]
    summary: str
    demo_kind: str
    has_simulation: bool = False
    has_animation: bool = False

    @property
    def source_apis(self) -> tuple[str, ...]:
        """Return tested library APIs that ground this topic's demo family."""
        return SOURCE_APIS_BY_DEMO_KIND[self.demo_kind]


@dataclass(frozen=True)
class TopicDemo:
    """Numerical arrays and plotting hints for one extras artifact."""

    spec: ExtraTopicSpec
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    line_keys: tuple[str, ...]
    heatmap_key: str | None = None
    bar_key: str | None = None


EXTRA_TOPICS: tuple[ExtraTopicSpec, ...] = (
    ExtraTopicSpec("model_representation", "Model And Representation", "Foundations", (1,), ("1.1",), "How internal representations stand in for hidden causes.", "foundation"),
    ExtraTopicSpec("order_and_surprisal", "Order And Surprisal", "Foundations", (1, 14), ("1.2", "14.3"), "Order as low-surprisal occupancy over viable states.", "ergodic", True),
    ExtraTopicSpec("bayes_equation", "Bayes Equation", "Foundations", (1, 2), ("1.3", "2.1.4", "C.2.4"), "Prior, likelihood, evidence, posterior, and normalization.", "foundation", True),
    ExtraTopicSpec("inverse_problem", "Inverse Problem", "Foundations", (1, 2), ("1.4", "2.1.4"), "Recovering hidden states from observations under a generative model.", "foundation", True),
    ExtraTopicSpec("generative_process_model", "Generative Process And Model", "Foundations", (2,), ("2.1.2", "2.1.3", "B.4"), "Contrasting the data-generating process with the agent's model.", "foundation", True),
    ExtraTopicSpec("precision_weighting", "Precision Weighting", "Foundations", (2, 5, 10), ("2.2", "5.1", "10.2"), "Variance, precision, and their control over belief updates.", "information", True),
    ExtraTopicSpec("hidden_state_estimation", "Hidden State Estimation", "Foundations", (2,), ("2.2", "2.3"), "Sequential inference over latent states.", "foundation", True, True),
    ExtraTopicSpec("multiple_samples", "Multiple Samples", "Statistical Estimation", (2,), ("2.3",), "How repeated observations tighten posterior beliefs.", "estimation", True),
    ExtraTopicSpec("mle_map", "MLE And MAP", "Statistical Estimation", (2,), ("2.5.1",), "Likelihood-only and prior-regularized point estimates.", "estimation", True),
    ExtraTopicSpec("gradient_descent", "Gradient Descent", "Statistical Estimation", (2, 3), ("2.5.2", "3.1"), "Iterative descent on a differentiable loss surface.", "estimation", True, True),
    ExtraTopicSpec("linear_regression", "Linear Regression", "Statistical Estimation", (3,), ("3.1", "3.2"), "Deterministic parameter estimation and residual geometry.", "estimation", True),
    ExtraTopicSpec("bayesian_linear_regression", "Bayesian Linear Regression", "Statistical Estimation", (3,), ("3.3",), "Posterior parameter uncertainty and predictive bands.", "estimation", True, True),
    ExtraTopicSpec("multivariate_gaussians", "Multivariate Gaussians", "Statistical Estimation", (3, 6), ("3.4", "C.10"), "Covariance geometry, entropy, and KL in multiple dimensions.", "information", True),
    ExtraTopicSpec("linear_gaussian_systems", "Linear Gaussian Systems", "Statistical Estimation", (3,), ("3.4",), "State-space prediction and filtering under linear Gaussian assumptions.", "continuous", True, True),
    ExtraTopicSpec("expectation_maximization", "Expectation Maximization", "Statistical Estimation", (3,), ("3.5",), "Alternating latent expectation and parameter maximization.", "estimation", True, True),
    ExtraTopicSpec("entropy", "Entropy", "Information And Variational Inference", (4, 14), ("C.10.4", "14.3"), "Discrete and differential entropy, including negative differential cases.", "information", True),
    ExtraTopicSpec("kl_divergence", "KL Divergence", "Information And Variational Inference", (4,), ("4.1", "C.10.5"), "Asymmetric divergence and posterior approximation loss.", "information", True),
    ExtraTopicSpec("surprisal_evidence", "Surprisal And Evidence", "Information And Variational Inference", (4,), ("4.2", "4.3"), "Evidence, negative log evidence, and bound gaps.", "variational", True),
    ExtraTopicSpec("variational_free_energy", "Variational Free Energy", "Information And Variational Inference", (4,), ("4.2", "4.3", "D.1"), "Energy, entropy, KL, and surprisal decompositions of VFE.", "variational", True, True),
    ExtraTopicSpec("mean_field_variational_inference", "Mean-Field Variational Inference", "Information And Variational Inference", (4,), ("4.5", "4.6"), "Factorized approximations and coordinate updates.", "variational", True),
    ExtraTopicSpec("cavi", "CAVI", "Information And Variational Inference", (4, 12), ("4.5", "12.4"), "Coordinate-ascent updates as repeated local message refinement.", "factor_graph", True, True),
    ExtraTopicSpec("model_comparison", "Model Comparison", "Information And Variational Inference", (4,), ("4.4", "C.11.1"), "Evidence and Bayes-factor style comparison across models.", "variational", True),
    ExtraTopicSpec("predictive_coding", "Predictive Coding", "Predictive Coding And Continuous Dynamics", (5,), ("5.1", "5.2"), "Prediction errors and recognition dynamics.", "continuous", True, True),
    ExtraTopicSpec("hierarchical_predictive_coding", "Hierarchical Predictive Coding", "Predictive Coding And Continuous Dynamics", (5, 8), ("5.4", "8.3"), "Layered prediction-error propagation.", "continuous", True, True),
    ExtraTopicSpec("generalized_filtering", "Generalized Filtering", "Predictive Coding And Continuous Dynamics", (6,), ("6.1", "6.2"), "Dynamic state inference with generalized filtering.", "continuous", True, True),
    ExtraTopicSpec("generalized_coordinates", "Generalized Coordinates", "Predictive Coding And Continuous Dynamics", (6,), ("6.3", "6.5", "6.6"), "State, velocity, and higher-order embedding orders.", "continuous", True),
    ExtraTopicSpec("active_generalized_filtering", "Active Generalized Filtering", "Active Inference Core", (7,), ("7.2", "7.4", "7.5"), "Action and perception as coupled free-energy descent.", "continuous", True, True),
    ExtraTopicSpec("learning_attention", "Learning And Attention", "Active Inference Core", (8,), ("8.1",), "Learning first- and second-order parameters through precision.", "learning", True, True),
    ExtraTopicSpec("hierarchical_message_passing", "Hierarchical Message Passing", "Active Inference Core", (8, 12), ("8.5", "12.5", "12.7"), "Forward and backward messages across hierarchical layers.", "factor_graph", True, True),
    ExtraTopicSpec("pomdp_arrays", "POMDP Arrays", "Discrete POMDP Active Inference", (9, 10), ("9.1", "B.10"), "D, A, B, C, and E arrays as discrete generative-model components.", "pomdp", True),
    ExtraTopicSpec("discrete_belief_filtering", "Discrete Belief Filtering", "Discrete POMDP Active Inference", (9,), ("9.2",), "Dynamic categorical belief updates over time.", "pomdp", True, True),
    ExtraTopicSpec("discrete_vfe", "Discrete VFE", "Discrete POMDP Active Inference", (9,), ("9.3",), "Discrete free energy for hidden-state estimation.", "pomdp", True),
    ExtraTopicSpec("gridworld_control", "Gridworld Control", "Discrete POMDP Active Inference", (9,), ("9.4", "9.5"), "Planning as inference in controllable grid-world transitions.", "application", True, True),
    ExtraTopicSpec("expected_free_energy", "Expected Free Energy", "Discrete POMDP Active Inference", (9,), ("9.5", "D.3"), "Risk, ambiguity, and epistemic value in policy scoring.", "extension", True),
    ExtraTopicSpec("exploration_exploitation", "Exploration And Exploitation", "Discrete POMDP Active Inference", (9, 10), ("9.6", "10.1"), "Policy choice as a tradeoff between information and preference satisfaction.", "pomdp", True, True),
    ExtraTopicSpec("dirichlet_learning", "Dirichlet Learning", "Learning And Depth", (10,), ("10.1",), "Pseudocount accumulation for POMDP parameter learning.", "learning", True, True),
    ExtraTopicSpec("policy_precision_habits", "Policy Precision And Habits", "Learning And Depth", (10,), ("10.2",), "Policy precision and baseline habits in action selection.", "pomdp", True, True),
    ExtraTopicSpec("factorial_depth", "Factorial Depth", "Learning And Depth", (10,), ("10.3", "12.6"), "Multiple state factors and observation modalities.", "factor_graph", True),
    ExtraTopicSpec("hierarchical_depth", "Hierarchical Depth", "Learning And Depth", (10, 12), ("10.4", "12.6"), "Nested policies and slower contextual layers.", "factor_graph", True),
    ExtraTopicSpec("free_energy_variants", "Free-Energy Variants", "Part III Extensions", (11,), ("11.1", "11.1.1", "11.1.2", "11.1.3", "11.1.4", "11.1.5", "11.1.6", "11.1.7", "11.1.8", "D.4"), "FEF, OFE, PFE, FEEF, generalized, Bethe, and Renyi forms.", "extension", True),
    ExtraTopicSpec("sophisticated_inference", "Sophisticated Inference", "Part III Extensions", (11,), ("11.2.1",), "Planning with beliefs over future belief updates.", "pomdp", True),
    ExtraTopicSpec("inductive_planning", "Inductive Planning", "Part III Extensions", (11,), ("11.2.2",), "Policy search that reuses substructure across paths.", "application", True),
    ExtraTopicSpec("state_preferences", "State Preferences", "Part III Extensions", (11,), ("11.2.3", "11.2.5"), "Preferences over states and time-dependent preference schedules.", "pomdp", True),
    ExtraTopicSpec("preference_habit_learning", "Preference And Habit Learning", "Part III Extensions", (11,), ("11.2.4",), "Learning preference and habit-style pseudocounts for C and E.", "learning"),
    ExtraTopicSpec("parameter_uncertainty", "Parameter Uncertainty", "Part III Extensions", (11,), ("11.2.6", "11.2.7"), "Forgetting rates and uncertainty on learned parameters.", "learning", True),
    ExtraTopicSpec("backward_smoothing", "Backward Smoothing", "Part III Extensions", (11, 12), ("11.2.9", "12.3"), "Backward messages that refine earlier state beliefs.", "factor_graph", True, True),
    ExtraTopicSpec("hybrid_generative_models", "Hybrid Generative Models", "Part III Extensions", (11, 12), ("11.3", "12.6"), "Continuous and discrete state-space components in one model.", "application", True),
    ExtraTopicSpec("tree_policy_search", "Tree Policy Search", "Part III Extensions", (11,), ("11.2.8", "11.4"), "Path-based policy computation, tree optimization, and receding search.", "application", True, True),
    ExtraTopicSpec("structure_learning", "Structure Learning", "Part III Extensions", (11,), ("11.5",), "Comparing candidate model structures through evidence-like scores.", "variational", True),
    ExtraTopicSpec("factor_graphs", "Factor Graphs", "Factor Graphs And Applications", (12,), ("12.1", "12.5", "B.12"), "Forney factor graphs as model diagrams for message passing.", "factor_graph", True),
    ExtraTopicSpec("belief_propagation", "Belief Propagation", "Factor Graphs And Applications", (12,), ("12.2", "12.3"), "Sum-product messages for state-space models.", "factor_graph", True, True),
    ExtraTopicSpec("variational_message_passing", "Variational Message Passing", "Factor Graphs And Applications", (12,), ("12.4", "12.4.1"), "Mean-field and marginal updates expressed as local messages.", "factor_graph", True),
    ExtraTopicSpec("robotics_navigation", "Robotics Navigation", "Factor Graphs And Applications", (13,), ("13.1", "13.2"), "Navigation and control as preference-seeking active inference.", "application", True, True),
    ExtraTopicSpec("social_robotics", "Social Robotics", "Factor Graphs And Applications", (13,), ("13.3",), "Belief updates over another agent's hidden intention.", "pomdp", True),
    ExtraTopicSpec("robotics_theory", "Robotics Theory", "Factor Graphs And Applications", (13,), ("13.4",), "Theory-facing robotics themes connecting control and active inference.", "application"),
    ExtraTopicSpec("ergodic_density", "Ergodic Density", "Thermodynamic/FEP Bridge", (14,), ("14.1", "14.2"), "Long-run occupancy as a density over viable states.", "ergodic", True, True),
    ExtraTopicSpec("fep_entropy_bounds", "FEP Entropy Bounds", "Thermodynamic/FEP Bridge", (14,), ("14.3",), "Entropy and VFE bounds for self-organizing systems.", "ergodic", True),
    ExtraTopicSpec("temperature", "Temperature", "Thermodynamic/FEP Bridge", (4, 14), ("D.3", "14.3"), "Temperature-scaled canonical probabilities and U - T S.", "thermo", True),
    ExtraTopicSpec("enthalpy", "Enthalpy", "Thermodynamic/FEP Bridge", (4, 14), ("D.4",), "H = U + pV and G = H - T S as explicit analogy-layer quantities.", "thermo", True),
    ExtraTopicSpec("bayesian_mechanics_bridge", "Bayesian Mechanics Bridge", "Thermodynamic/FEP Bridge", (14,), ("14.1", "14.4"), "A careful bridge between active inference, FEP, and Bayesian mechanics.", "ergodic", True),
    ExtraTopicSpec("active_inference_history", "Active Inference History", "Appendix A", (14,), ("A.1", "A.1.1", "A.1.2", "A.1.3", "A.1.4", "A.1.5", "A.1.6", "A.1.7"), "Historical lineage from neuroimaging through discrete active inference.", "history"),
    ExtraTopicSpec("active_inference_future", "Active Inference Future", "Appendix A", (14,), ("A.2", "A.2.5"), "Future-facing active inference research directions.", "history"),
    ExtraTopicSpec("deep_generative_models", "Deep Generative Models", "Appendix A", (14,), ("A.2.1",), "Deep learning and generative-model context for active inference.", "history"),
    ExtraTopicSpec("cybernetics_control", "Cybernetics And Control", "Appendix A", (14,), ("A.2.2",), "Cybernetics and control-theory context for active inference.", "history"),
    ExtraTopicSpec("information_theory_lineage", "Information Theory Lineage", "Appendix A", (14,), ("A.2.3",), "Information-theory context for active inference and FEP.", "history"),
    ExtraTopicSpec("reinforcement_learning_lineage", "Reinforcement Learning Lineage", "Appendix A", (14,), ("A.2.4",), "Reinforcement-learning context for active inference.", "history"),
    ExtraTopicSpec("appendix_notation_model_setup", "Notation And Model Setup", "Appendices", (2, 9, 12), ("B.1", "B.2", "B.3", "B.4", "B.5", "B.6", "B.7", "B.8", "B.9", "B.10", "B.11", "B.12"), "Appendix B notation, dimensions, and model-setup conventions.", "appendix_math"),
    ExtraTopicSpec("appendix_math_fundamentals", "Mathematical Fundamentals", "Appendices", (2, 3, 4), ("C.1", "C.1.1", "C.1.2", "C.1.3", "C.1.4", "C.2", "C.2.1", "C.2.2", "C.2.2.1", "C.2.2.2", "C.2.2.3", "C.2.2.4", "C.2.2.5", "C.2.3", "C.2.4", "C.2.5", "C.2.6", "C.3", "C.3.1", "C.3.2", "C.4", "C.4.1", "C.4.2", "C.4.3", "C.4.4", "C.5", "C.5.1", "C.5.2", "C.5.3", "C.5.4", "C.6", "C.6.1", "C.6.2", "C.7", "C.7.1", "C.7.2", "C.7.3", "C.8", "C.8.1", "C.8.2", "C.8.3", "C.10", "C.10.1", "C.10.2", "C.10.3", "C.10.4", "C.10.5", "C.10.6", "C.10.7", "C.12", "C.13", "C.13.1", "C.13.2"), "Appendix C mathematical fundamentals used across the companion.", "appendix_math"),
    ExtraTopicSpec("colored_noise", "Colored Noise", "Appendices", (6, 8), ("C.9",), "Smooth colored-noise covariance and trajectory intuition.", "noise"),
    ExtraTopicSpec("bayesian_model_selection", "Bayesian Model Selection", "Appendices", (4, 11), ("C.11", "C.11.1", "C.11.2", "C.11.3", "C.11.4"), "Bayes factors, model averaging, reduction, and expansion.", "model_comparison"),
    ExtraTopicSpec("appendix_free_energy_forms", "Appendix Free-Energy Forms", "Appendices", (4, 9, 11), ("D.1", "D.2", "D.3", "D.3.1", "D.3.2", "D.3.3", "D.3.4", "D.4"), "Appendix D static, dynamic, expected, and variant free-energy forms.", "extension"),
)


def extra_topic_slugs() -> tuple[str, ...]:
    """Return all extras topic slugs in registry order."""
    return tuple(spec.slug for spec in EXTRA_TOPICS)


def extra_topic_spec(slug: str) -> ExtraTopicSpec:
    """Return the registry spec for one extras topic slug."""
    for spec in EXTRA_TOPICS:
        if spec.slug == slug:
            return spec
    raise KeyError(f"Unknown extras topic: {slug!r}")


def extra_topics_by_family() -> dict[str, list[ExtraTopicSpec]]:
    """Group registered extras topics by curriculum family."""
    grouped: dict[str, list[ExtraTopicSpec]] = {}
    for spec in EXTRA_TOPICS:
        grouped.setdefault(spec.family, []).append(spec)
    return grouped


def _topic_index(slug: str) -> int:
    """Return a stable one-based index for deterministic parameterization."""
    return extra_topic_slugs().index(slug) + 1


def _gaussian(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """Evaluate a normalized one-dimensional Gaussian density."""
    return gaussian_pdf(x, mean, variance)


def _normalize_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normalize non-negative grid values to unit trapezoid mass."""
    return normalize_density(y, x)


def _foundation_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build a Bayesian-foundation demo with prior, likelihood, and posterior."""
    idx = _topic_index(spec.slug)
    x = np.linspace(-4.0, 4.0, 400)
    prior_mean = -1.2 + 0.08 * (idx % 7)
    likelihood_mean = 0.8 + 0.06 * (idx % 5)
    prior_var = 1.2 + 0.08 * (idx % 4)
    likelihood_var = 0.35 + 0.04 * (idx % 6)
    prior = _gaussian(x, prior_mean, prior_var)
    likelihood = _gaussian(x, likelihood_mean, likelihood_var)
    posterior = _normalize_grid(x, prior * likelihood)
    evidence = np.trapezoid(prior * likelihood, x)
    if mode == "simulate":
        precisions = np.linspace(0.5, 8.0, 80)
        means = np.empty_like(precisions)
        variances = np.empty_like(precisions)
        for i, precision in enumerate(precisions):
            lv = 1.0 / precision
            post_precision = 1.0 / prior_var + 1.0 / lv
            variances[i] = 1.0 / post_precision
            means[i] = variances[i] * (prior_mean / prior_var + likelihood_mean / lv)
        return TopicDemo(
            spec,
            {
                "x": precisions,
                "primary": means,
                "secondary": variances,
                "tertiary": 1.0 / variances,
                "bar_y": np.array([prior_mean, likelihood_mean, means[-1]]),
            },
            {"x_label": "likelihood precision", "primary_label": "posterior mean", "secondary_label": "posterior variance", "scalar": float(evidence)},
            ("primary", "secondary", "tertiary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {
            "x": x,
            "primary": prior,
            "secondary": likelihood / np.max(likelihood),
            "tertiary": posterior,
            "bar_y": np.array([float(evidence), float(np.trapezoid(posterior, x)), float(x[np.argmax(posterior)])]),
        },
        {"x_label": "hidden state", "primary_label": "prior", "secondary_label": "scaled likelihood", "tertiary_label": "posterior", "scalar": float(evidence)},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _estimation_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build a parameter-estimation demo with losses and descent traces."""
    idx = _topic_index(spec.slug)
    theta = np.linspace(-3.0, 5.0, 360)
    target = 0.5 + 0.07 * (idx % 10)
    mle = (theta - target) ** 2 + 0.2 * np.sin(theta + idx / 7.0) ** 2
    prior_penalty = 0.18 * (theta + 0.8) ** 2
    map_loss = mle + prior_penalty
    step = 0.16
    path = np.empty(40)
    path[0] = -2.6
    for t in range(1, path.size):
        grad = 2.0 * (path[t - 1] - target) + 0.4 * np.sin(path[t - 1]) * np.cos(path[t - 1])
        path[t] = path[t - 1] - step * grad
    if mode == "simulate":
        rates = np.linspace(0.02, 0.35, 80)
        final_loss = []
        for rate in rates:
            value = -2.6
            for _ in range(35):
                grad = 2.0 * (value - target)
                value -= rate * grad
            final_loss.append((value - target) ** 2)
        return TopicDemo(
            spec,
            {"x": rates, "primary": np.asarray(final_loss), "secondary": rates * 0.0 + target, "bar_y": path[:8]},
            {"x_label": "learning rate", "primary_label": "final loss", "secondary_label": "target parameter"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {
            "x": theta,
            "primary": mle,
            "secondary": map_loss,
            "tertiary": np.interp(
                theta,
                np.linspace(theta.min(), theta.max(), path.size),
                np.linspace(float(mle.max()), float(mle.min()), path.size),
            ),
            "bar_y": path,
        },
        {"x_label": "parameter", "primary_label": "MLE loss", "secondary_label": "MAP loss", "tertiary_label": "descent trace"},
        ("primary", "secondary"),
        bar_key="bar_y",
    )


def _information_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build entropy, KL, and precision information-theory arrays."""
    idx = _topic_index(spec.slug)
    x = np.linspace(-4.0, 4.0, 500)
    p = _gaussian(x, -0.5 + 0.04 * idx, 1.0)
    q = _gaussian(x, 0.7, 0.4 + 0.02 * (idx % 7))
    p = _normalize_grid(x, p)
    q = _normalize_grid(x, q)
    entropy_terms = np.zeros_like(q)
    mask = q > 0.0
    entropy_terms[mask] = -q[mask] * np.log(q[mask])
    kl_terms = np.zeros_like(q)
    mask = (q > 0.0) & (p > 0.0)
    kl_terms[mask] = q[mask] * (np.log(q[mask]) - np.log(p[mask]))
    variances = np.linspace(0.02, 3.0, 120)
    entropy_curve = 0.5 * np.log(2.0 * np.pi * np.e * variances)
    if mode == "simulate":
        temperature = np.linspace(0.15, 4.0, 120)
        energies = np.array([0.0, 1.0, 2.0, 3.0])
        entropy = np.array([boltzmann_entropy(canonical_probabilities(energies, temperature=t)) for t in temperature])
        return TopicDemo(
            spec,
            {"x": temperature, "primary": entropy, "secondary": 1.0 / temperature, "bar_y": canonical_probabilities(energies, temperature=1.0)},
            {"x_label": "temperature", "primary_label": "canonical entropy", "secondary_label": "inverse temperature"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": x, "primary": q, "secondary": p, "tertiary": entropy_terms, "quaternary": kl_terms, "bar_y": entropy_curve},
        {"x_label": "state", "primary_label": "q(x)", "secondary_label": "p(x)", "tertiary_label": "entropy density", "kl": float(np.trapezoid(kl_terms, x))},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _variational_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build variational free-energy and evidence-bound teaching arrays."""
    idx = _topic_index(spec.slug)
    means = np.linspace(-2.5, 2.5, 180)
    evidence = -1.3 - 0.02 * (idx % 5)
    kl = 0.25 * (means - 0.4) ** 2 + 0.08
    vfe = -evidence + kl
    gap = vfe + evidence
    if mode == "simulate":
        variances = np.linspace(0.08, 2.5, 120)
        entropy_curve = 0.5 * np.log(2.0 * np.pi * np.e * variances)
        energy_curve = 0.7 / variances + 0.15 * variances
        return TopicDemo(
            spec,
            {"x": variances, "primary": energy_curve, "secondary": entropy_curve, "tertiary": energy_curve - entropy_curve, "bar_y": np.array([float(np.min(vfe)), float(np.min(gap)), -evidence])},
            {"x_label": "q variance", "primary_label": "energy", "secondary_label": "entropy", "tertiary_label": "free energy"},
            ("primary", "secondary", "tertiary"),
            bar_key="bar_y",
        )
    belief = GaussianBelief(mu=0.4, var=0.6)
    bound_gap = float(np.min(gap))
    return TopicDemo(
        spec,
        {"x": means, "primary": vfe, "secondary": gap, "tertiary": np.full_like(means, -evidence), "bar_y": np.array([belief.mu, belief.var, bound_gap])},
        {"x_label": "q mean", "primary_label": "VFE", "secondary_label": "bound gap", "tertiary_label": "surprisal", "bound_gap": float(bound_gap)},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _continuous_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build continuous-state dynamics, prediction-error, and filtering traces."""
    idx = _topic_index(spec.slug)
    t = np.linspace(0.0, 12.0, 360)
    truth = np.sin(0.8 * t) + 0.25 * np.sin(2.1 * t + idx / 5.0)
    prediction = np.sin(0.8 * t - 0.35) * np.exp(-0.02 * t)
    error = truth - prediction
    belief = prediction + 0.55 * (1.0 - np.exp(-0.45 * t)) * error
    if mode == "simulate":
        precision = np.linspace(0.2, 8.0, 120)
        error_energy = 1.0 / (precision + 0.4) + 0.04 * precision
        return TopicDemo(
            spec,
            {"x": precision, "primary": error_energy, "secondary": 1.0 / precision, "bar_y": np.array([np.mean(np.abs(error)), np.mean(np.abs(truth - belief)), np.var(error)])},
            {"x_label": "precision", "primary_label": "prediction-error energy", "secondary_label": "variance"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": t, "primary": truth, "secondary": prediction, "tertiary": belief, "quaternary": error, "bar_y": np.array([np.mean(np.abs(error)), np.max(np.abs(error)), np.mean(np.abs(truth - belief))])},
        {"x_label": "time", "primary_label": "process", "secondary_label": "prediction", "tertiary_label": "belief"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _pomdp_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build categorical POMDP belief and policy-score arrays."""
    idx = _topic_index(spec.slug)
    states = np.arange(5, dtype=float)
    prior = np.array([0.45, 0.20, 0.15, 0.12, 0.08])
    likelihood = np.roll(np.array([0.05, 0.15, 0.55, 0.20, 0.05]), idx % 5)
    posterior = prior * likelihood
    posterior = posterior / posterior.sum()
    risk = np.array([1.8, 0.9, 1.2, 0.6]) + 0.03 * (idx % 4)
    ambiguity = np.array([0.2, 0.55, 0.35, 0.75])
    scores = risk + ambiguity
    if mode == "simulate":
        gamma = np.linspace(0.2, 8.0, 120)
        best_policy_probability = []
        for g in gamma:
            weights = np.exp(-g * (scores - np.min(scores)))
            q = weights / weights.sum()
            best_policy_probability.append(float(q[np.argmin(scores)]))
        return TopicDemo(
            spec,
            {"x": gamma, "primary": np.asarray(best_policy_probability), "secondary": 1.0 - np.asarray(best_policy_probability), "bar_y": scores},
            {"x_label": "policy precision", "primary_label": "best-policy probability", "secondary_label": "residual policy mass"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": states, "primary": prior, "secondary": likelihood, "tertiary": posterior, "bar_y": scores},
        {"x_label": "state", "primary_label": "prior D", "secondary_label": "likelihood A(o|s)", "tertiary_label": "posterior"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _learning_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build Dirichlet-style learning and attention curves."""
    idx = _topic_index(spec.slug)
    trials = np.arange(1, 80, dtype=float)
    counts_a = 1.0 + 0.72 * trials
    counts_b = 1.0 + 0.28 * trials
    expected_a = counts_a / (counts_a + counts_b)
    precision = 0.8 + 0.05 * np.sqrt(trials) + 0.01 * (idx % 5)
    if mode == "simulate":
        forgetting = np.linspace(0.0, 0.12, 100)
        retained = np.exp(-40.0 * forgetting)
        uncertainty = 1.0 / (1.0 + 20.0 * retained)
        return TopicDemo(
            spec,
            {"x": forgetting, "primary": retained, "secondary": uncertainty, "bar_y": np.array([expected_a[-1], precision[-1], uncertainty[-1]])},
            {"x_label": "forgetting rate", "primary_label": "retained evidence", "secondary_label": "parameter uncertainty"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": trials, "primary": expected_a, "secondary": 1.0 - expected_a, "tertiary": precision / np.max(precision), "bar_y": np.array([counts_a[-1], counts_b[-1], precision[-1]])},
        {"x_label": "trial", "primary_label": "learned A[good]", "secondary_label": "learned A[bad]", "tertiary_label": "scaled precision"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _extension_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build arrays for free-energy variants and planning extensions."""
    idx = _topic_index(spec.slug)
    policies = np.arange(6, dtype=float)
    risk = np.linspace(1.6, 0.35, policies.size) + 0.08 * np.sin(policies + idx)
    ambiguity = np.linspace(0.2, 0.9, policies.size)
    epistemic = np.linspace(0.85, 0.15, policies.size)
    table = free_energy_variant_table(risk, ambiguity, epistemic)
    renyi = np.array([renyi_bound(np.ones_like(risk), risk + ambiguity, alpha=a) for a in np.linspace(0.2, 1.8, policies.size) if not np.isclose(a, 1.0)])
    if mode == "simulate":
        alpha = np.linspace(0.2, 1.8, 120)
        probs = np.ones_like(risk) / risk.size
        bounds = np.array([renyi_bound(probs, risk + ambiguity, a) if not np.isclose(a, 1.0) else np.mean(risk + ambiguity) for a in alpha])
        return TopicDemo(
            spec,
            {"x": alpha, "primary": bounds, "secondary": np.full_like(bounds, np.mean(risk + ambiguity)), "bar_y": table["expected_free_energy"]},
            {"x_label": "Renyi alpha", "primary_label": "Renyi certainty equivalent", "secondary_label": "expected energy"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": policies, "primary": table["expected_free_energy"], "secondary": table["free_energy_of_future"], "tertiary": table["generalized_free_energy"], "bar_y": renyi},
        {"x_label": "policy", "primary_label": "EFE", "secondary_label": "FEF", "tertiary_label": "GFE"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _factor_graph_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build factor-graph message-passing arrays."""
    prior = np.array([0.65, 0.35])
    transition = np.array([[0.82, 0.28], [0.18, 0.72]])
    likelihoods = np.array([[0.9, 0.2], [0.55, 0.65], [0.25, 0.88], [0.4, 0.7]])
    beliefs = sum_product_chain(prior, transition, likelihoods)
    fac_msg = categorical_factor_message(transition, [prior], target_axis=0)
    vmp_msg = variational_message_update(np.log(transition + 1e-12), [prior], target_axis=0)
    if mode == "simulate":
        weights = np.linspace(0.05, 0.95, 100)
        posterior_state_1 = []
        for w in weights:
            msg = categorical_factor_message(transition, [np.array([w, 1.0 - w])], target_axis=0)
            posterior_state_1.append(msg[1])
        return TopicDemo(
            spec,
            {"x": weights, "primary": np.asarray(posterior_state_1), "secondary": 1.0 - np.asarray(posterior_state_1), "bar_y": vmp_msg},
            {"x_label": "incoming state-0 belief", "primary_label": "message to state 1", "secondary_label": "message to state 0"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": np.arange(beliefs.shape[0], dtype=float), "primary": beliefs[:, 0], "secondary": beliefs[:, 1], "tertiary": np.full(beliefs.shape[0], fac_msg[1]), "bar_y": vmp_msg},
        {"x_label": "time/message step", "primary_label": "belief state 0", "secondary_label": "belief state 1", "tertiary_label": "factor message"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _application_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build simple navigation and policy-search teaching arrays."""
    idx = _topic_index(spec.slug)
    t = np.linspace(0.0, 1.0, 120)
    start = np.array([0.0, 0.0])
    goal = np.array([1.0, 0.72 + 0.03 * (idx % 5)])
    x_path = start[0] + (goal[0] - start[0]) * t
    y_path = start[1] + (goal[1] - start[1]) * (1.0 - np.cos(np.pi * t)) / 2.0
    distance = np.sqrt((goal[0] - x_path) ** 2 + (goal[1] - y_path) ** 2)
    preference = np.exp(-4.0 * distance)
    if mode == "simulate":
        horizon = np.arange(1, 11, dtype=float)
        cost = 1.0 / horizon + 0.025 * horizon
        return TopicDemo(
            spec,
            {"x": horizon, "primary": cost, "secondary": 1.0 / horizon, "bar_y": np.array([goal[0], goal[1], cost.min()])},
            {"x_label": "planning horizon", "primary_label": "tree-search cost", "secondary_label": "lookahead value"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": x_path, "primary": y_path, "secondary": preference, "tertiary": distance, "bar_y": np.array([goal[0], goal[1], preference[-1]])},
        {"x_label": "x position", "primary_label": "y position", "secondary_label": "preference satisfaction", "tertiary_label": "distance to goal"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _thermo_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build thermodynamic bridge arrays for temperature and enthalpy topics."""
    idx = _topic_index(spec.slug)
    temperature = np.linspace(0.2, 4.0, 160)
    energies = np.array([0.0, 0.7, 1.5, 2.8])
    expected = np.empty_like(temperature)
    entropy = np.empty_like(temperature)
    helmholtz = np.empty_like(temperature)
    for i, temp in enumerate(temperature):
        probs = canonical_probabilities(energies, temperature=temp)
        expected[i] = expected_energy(probs, energies)
        entropy[i] = boltzmann_entropy(probs)
        helmholtz[i] = helmholtz_free_energy(expected[i], entropy[i], temp)
    if mode == "simulate":
        pressure = np.linspace(0.0, 2.0, 120)
        volume = 0.5 + 0.05 * (idx % 4)
        enthalpy_curve = expected[len(expected) // 2] + pressure * volume
        gibbs_curve = enthalpy_curve - entropy[len(entropy) // 2]
        return TopicDemo(
            spec,
            {"x": pressure, "primary": enthalpy_curve, "secondary": gibbs_curve, "bar_y": canonical_probabilities(energies, temperature=1.0)},
            {"x_label": "analogy pressure", "primary_label": "enthalpy H", "secondary_label": "Gibbs-like G"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": temperature, "primary": expected, "secondary": entropy, "tertiary": helmholtz, "bar_y": canonical_probabilities(energies, temperature=1.0)},
        {"x_label": "temperature", "primary_label": "U", "secondary_label": "S", "tertiary_label": "U - T S"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _ergodic_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build ergodic density and entropy-bound teaching arrays."""
    idx = _topic_index(spec.slug)
    trajectory = ergodic_ou_trajectory(n_steps=520, drift=0.05 + 0.002 * (idx % 8))
    centers, density = ergodic_density(trajectory, bins=90)
    entropy = density_entropy(centers, density)
    upper = entropy + 0.35 + 0.02 * (idx % 5)
    bound = entropy_upper_bound_from_vfe(entropy, upper)
    if mode == "simulate":
        drift = np.linspace(0.02, 0.16, 80)
        entropy_curve = []
        for value in drift:
            xs = ergodic_ou_trajectory(n_steps=260, drift=float(value))
            c, d = ergodic_density(xs, bins=50)
            entropy_curve.append(density_entropy(c, d))
        entropy_curve = np.asarray(entropy_curve)
        return TopicDemo(
            spec,
            {"x": drift, "primary": entropy_curve, "secondary": entropy_curve + bound.gap, "bar_y": np.array([bound.entropy, bound.upper_bound, bound.gap])},
            {"x_label": "attractor drift", "primary_label": "ergodic entropy", "secondary_label": "VFE-like upper bound"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": centers, "primary": density, "secondary": np.full_like(centers, entropy), "tertiary": np.full_like(centers, upper), "bar_y": np.array([bound.entropy, bound.upper_bound, bound.gap])},
        {"x_label": "state", "primary_label": "ergodic density", "secondary_label": "entropy", "tertiary_label": "VFE-like bound"},
        ("primary", "secondary", "tertiary"),
        bar_key="bar_y",
    )


def _history_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build Appendix A historical/future context as source-spine timelines."""
    appendix_a = next(app for app in APPENDICES if app.letter == "A")
    x = np.arange(len(spec.sections), dtype=float)
    emphasis = np.linspace(0.35, 1.0, max(1, x.size))
    if spec.slug == "active_inference_history":
        primary = emphasis
        secondary = np.linspace(1.0, 0.45, x.size)
    elif spec.slug == "active_inference_future":
        primary = np.linspace(0.45, 1.0, x.size)
        secondary = np.linspace(0.35, 0.8, x.size)
    else:
        primary = np.array([0.65 + 0.05 * (_topic_index(spec.slug) % 5)])
        secondary = np.array([0.45 + 0.04 * (_topic_index(spec.slug) % 4)])
        x = np.array([0.0])
    if mode == "simulate":
        horizon = np.linspace(0.0, 1.0, 80)
        primary = primary[-1] * (0.5 + 0.5 * horizon)
        secondary = secondary[-1] * np.sqrt(horizon + 1e-9)
        x = horizon
    return TopicDemo(
        spec,
        {"x": x, "primary": primary, "secondary": secondary, "bar_y": np.array([len(appendix_a.sections), len(spec.sections), _topic_index(spec.slug)])},
        {"x_label": "source-spine position", "primary_label": "AIF emphasis", "secondary_label": "adjacent-field emphasis"},
        ("primary", "secondary"),
        bar_key="bar_y",
    )


def _appendix_math_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build Appendix B/C math-support arrays from reusable helpers."""
    x = np.linspace(0.01, 8.0, 220)
    gamma = gamma_pdf(x, shape=2.5, rate=1.1)
    categorical = maximum_entropy_distribution(4)
    dmean = dirichlet_mean(np.array([1.0, 2.0, 3.0, 4.0]))
    joint = np.array([[0.28, 0.12], [0.18, 0.42]])
    mi = mutual_information(joint)
    if mode == "simulate":
        concentration = np.linspace(1.0, 20.0, 120)
        entropy = np.log(concentration + 1.0)
        precision = concentration / np.max(concentration)
        return TopicDemo(
            spec,
            {"x": concentration, "primary": entropy, "secondary": precision, "bar_y": dmean},
            {"x_label": "concentration", "primary_label": "log concentration", "secondary_label": "scaled precision"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": x, "primary": gamma, "secondary": np.full_like(x, mi), "bar_y": np.concatenate([categorical, dmean])},
        {"x_label": "value", "primary_label": "Gamma density", "secondary_label": "mutual information"},
        ("primary", "secondary"),
        bar_key="bar_y",
    )


def _model_comparison_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build Appendix C.11 model-comparison arrays."""
    x = np.arange(5, dtype=float)
    log_evidence = np.array([-3.0, -1.6, -1.1, -1.4, -2.2])
    posterior = model_posterior(log_evidence)
    if mode == "simulate":
        complexity = np.linspace(0.0, 1.5, 120)
        selected = []
        for value in complexity:
            selected.append(np.argmax(model_posterior(log_evidence - value * x)))
        selected_arr = np.asarray(selected, dtype=float)
        return TopicDemo(
            spec,
            {"x": complexity, "primary": selected_arr, "secondary": np.max(posterior) - 0.05 * complexity, "bar_y": posterior},
            {"x_label": "complexity penalty", "primary_label": "selected model", "secondary_label": "best posterior"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": x, "primary": log_evidence, "secondary": posterior, "bar_y": posterior},
        {"x_label": "model", "primary_label": "log evidence", "secondary_label": "posterior"},
        ("primary", "secondary"),
        bar_key="bar_y",
    )


def _noise_demo(spec: ExtraTopicSpec, mode: str) -> TopicDemo:
    """Build Appendix C.9 smooth colored-noise arrays."""
    x = np.linspace(0.0, 6.0, 160)
    cov = squared_exponential_covariance(x, length_scale=0.65, variance=1.0)
    sample = sample_colored_noise(x, length_scale=0.65, variance=0.4, rng=np.random.default_rng(7))
    if mode == "simulate":
        length = np.linspace(0.15, 2.0, 120)
        smoothness = np.array(
            [np.mean(squared_exponential_covariance(x[:20], length_scale=scale)) for scale in length]
        )
        return TopicDemo(
            spec,
            {"x": length, "primary": smoothness, "secondary": 1.0 / length, "bar_y": np.diag(cov)[:8]},
            {"x_label": "length scale", "primary_label": "mean covariance", "secondary_label": "inverse length"},
            ("primary", "secondary"),
            bar_key="bar_y",
        )
    return TopicDemo(
        spec,
        {"x": x, "primary": sample, "secondary": cov[len(cov) // 2], "bar_y": np.linalg.eigvalsh(cov[:8, :8])},
        {"x_label": "time", "primary_label": "colored-noise sample", "secondary_label": "covariance row"},
        ("primary", "secondary"),
        bar_key="bar_y",
    )


def build_topic_demo(slug: str, *, mode: str = "visualize") -> TopicDemo:
    """Build deterministic numeric arrays for one topic and artifact mode."""
    spec = extra_topic_spec(slug)
    if mode not in {"visualize", "simulate"}:
        raise ValueError("mode must be 'visualize' or 'simulate'")
    builders = {
        "foundation": _foundation_demo,
        "estimation": _estimation_demo,
        "information": _information_demo,
        "variational": _variational_demo,
        "continuous": _continuous_demo,
        "pomdp": _pomdp_demo,
        "learning": _learning_demo,
        "extension": _extension_demo,
        "factor_graph": _factor_graph_demo,
        "application": _application_demo,
        "thermo": _thermo_demo,
        "ergodic": _ergodic_demo,
        "history": _history_demo,
        "appendix_math": _appendix_math_demo,
        "model_comparison": _model_comparison_demo,
        "noise": _noise_demo,
    }
    demo = builders[spec.demo_kind](spec, mode)
    metadata = {
        "topic": spec.slug,
        "title": spec.title,
        "family": spec.family,
        "chapters": list(spec.chapters),
        "sections": list(spec.sections),
        "summary": spec.summary,
        "demo_kind": spec.demo_kind,
        "mode": mode,
        "source_apis": list(spec.source_apis),
        **demo.metadata,
    }
    return TopicDemo(
        spec,
        demo.arrays,
        metadata,
        demo.line_keys,
        heatmap_key=demo.heatmap_key,
        bar_key=demo.bar_key,
    )


def _line_label(demo: TopicDemo, key: str) -> str:
    """Return a display label for one line key."""
    return str(demo.metadata.get(f"{key}_label", key.replace("_", " ")))


def _sections_caption(sections: tuple[str, ...]) -> str:
    """Return a compact section list for figure captions."""
    if len(sections) <= 8:
        return ", ".join(sections)
    head = ", ".join(sections[:4])
    return f"{head}, ... ({len(sections)} sections)"


def render_topic_figure(slug: str, *, mode: str = "visualize") -> tuple[plt.Figure, TopicDemo]:
    """Render a deterministic static figure for one extras topic."""
    demo = build_topic_demo(slug, mode=mode)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.7, 1.0)},
    )
    x = demo.arrays["x"]
    plotted = 0
    for key in demo.line_keys:
        y = demo.arrays[key]
        if y.shape == x.shape and np.all(np.isfinite(y)):
            color = list(COLORS.values())[plotted % len(COLORS)]
            axes[0].plot(x, y, lw=2.6, label=_line_label(demo, key), color=color)
            plotted += 1
    axes[0].set_title(demo.spec.title, loc="left")
    axes[0].set_xlabel(str(demo.metadata.get("x_label", "axis")))
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.3)
    if plotted:
        axes[0].legend(fontsize=10, framealpha=0.95)

    if demo.heatmap_key is not None:
        heatmap = demo.arrays[demo.heatmap_key]
        image = axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("State-space Surface", loc="left")
    elif demo.bar_key is not None:
        values = np.ravel(demo.arrays[demo.bar_key])
        axes[1].bar(np.arange(values.size), values, color=COLORS["posterior"])
        axes[1].set_title("Diagnostic Quantities", loc="left")
        axes[1].set_xlabel("quantity")
        axes[1].set_ylabel("value")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].axis("off")
    source = str(demo.metadata["source_apis"][0]).split(".")[-1]
    caption = (
        f"{demo.spec.family} | sections {_sections_caption(demo.spec.sections)}\n"
        f"source: {source} | {mode}"
    )
    annotate_stat_box(axes[1], caption, loc="lower left", fontsize=9, monospace=False)
    fig.suptitle(f"{demo.spec.summary}", fontsize=14)
    return fig, demo


_ANIMATION_KIND_BY_DEMO_KIND: dict[str, str] = {
    "foundation": "sequential Bayesian update",
    "estimation": "descent trace",
    "information": "temperature and information sweep",
    "variational": "variational bound descent",
    "continuous": "recognition dynamics",
    "pomdp": "categorical policy/belief dynamics",
    "learning": "parameter learning trajectory",
    "extension": "free-energy variant sweep",
    "factor_graph": "message-passing trajectory",
    "application": "control/path trajectory",
    "thermo": "thermodynamic potential sweep",
    "ergodic": "ergodic-density trajectory",
    "history": "source-spine context sweep",
    "appendix_math": "appendix math parameter sweep",
    "model_comparison": "model evidence sweep",
    "noise": "colored-noise smoothness sweep",
}


def _animation_base_demo(slug: str) -> TopicDemo:
    """Return the most dynamic deterministic demo available for animation."""
    spec = extra_topic_spec(slug)
    if spec.has_simulation:
        return build_topic_demo(slug, mode="simulate")
    return build_topic_demo(slug, mode="visualize")


def _topic_animation_traces(demo: TopicDemo, *, frames: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build family-specific finite trajectories from a deterministic topic demo."""
    x = np.asarray(demo.arrays["x"], dtype=float).reshape(-1)
    primary = np.asarray(demo.arrays[demo.line_keys[0]], dtype=float).reshape(-1)
    if primary.shape != x.shape:
        x = np.linspace(0.0, 1.0, primary.size)
    if len(demo.line_keys) > 1:
        baseline = np.asarray(demo.arrays[demo.line_keys[1]], dtype=float).reshape(-1)
        if baseline.shape != primary.shape:
            baseline = np.full_like(primary, float(np.nanmean(primary)))
    elif demo.bar_key is not None:
        baseline = np.interp(
            np.linspace(0.0, 1.0, primary.size),
            np.linspace(0.0, 1.0, np.ravel(demo.arrays[demo.bar_key]).size),
            np.ravel(demo.arrays[demo.bar_key]),
        )
    else:
        baseline = np.zeros_like(primary) + float(np.nanmean(primary[: max(1, primary.size // 8)]))

    weights = np.linspace(0.0, 1.0, frames)
    kind = demo.spec.demo_kind
    if kind in {"estimation", "variational", "continuous", "learning"}:
        shaped = 1.0 - np.exp(-4.0 * weights)
    elif kind in {"pomdp", "factor_graph", "extension"}:
        shaped = weights ** 0.65
    elif kind in {"thermo", "ergodic", "information"}:
        shaped = 0.5 - 0.5 * np.cos(np.pi * weights)
    else:
        shaped = weights
    traces = np.array([(1.0 - w) * baseline + w * primary for w in shaped])
    target = primary
    return x, target, traces, weights


def build_topic_animation(slug: str) -> tuple[FuncAnimation, dict[str, np.ndarray], dict[str, object]]:
    """Build a lightweight animation for a dynamic extras topic."""
    spec = extra_topic_spec(slug)
    base = _animation_base_demo(slug)
    x, target, traces, weights = _topic_animation_traces(base)
    frames = traces.shape[0]
    trajectory_kind = _ANIMATION_KIND_BY_DEMO_KIND[spec.demo_kind]
    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ymin = float(np.min([np.min(traces), np.min(target)]))
    ymax = float(np.max([np.max(traces), np.max(target)]))
    pad = 0.08 * max(1e-6, ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel(str(base.metadata.get("x_label", "axis")))
    ax.set_ylabel(_line_label(base, base.line_keys[0]))
    ax.set_title(f"{spec.title}: {trajectory_kind}", loc="left")
    ax.grid(alpha=0.3)
    target_line, = ax.plot(x, target, color=COLORS["neutral"], lw=1.5, ls="--", label="target")
    line, = ax.plot([], [], color=COLORS["likelihood"], lw=2.6, label="current")
    text = annotate_stat_box(ax, "", loc="upper left", fontsize=9, monospace=False)
    ax.legend(fontsize=10)

    def init() -> tuple[object, ...]:
        """Initialize animated artists."""
        line.set_data([], [])
        text.set_text("")
        return line, target_line, text

    def update(frame: int) -> tuple[object, ...]:
        """Update animated artists for one frame."""
        line.set_data(x, traces[frame])
        text.set_text(
            f"{trajectory_kind}\n"
            f"frame {frame + 1}/{frames}\n"
            f"source {base.metadata['source_apis'][0].split('.')[-1]}"
        )
        return line, target_line, text

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=80)
    raw = {"x": x, "target": target, "traces": traces, "weights": weights}
    metadata = {
        "topic": slug,
        "family": spec.family,
        "summary": spec.summary,
        "trajectory_kind": trajectory_kind,
        "source_apis": list(spec.source_apis),
    }
    anim._raw_data = raw  # type: ignore[attr-defined]
    anim._metadata = metadata  # type: ignore[attr-defined]
    return anim, raw, metadata


def _parse_topic_args(argv: Sequence[str] | None, description: str) -> argparse.Namespace:
    """Parse common command-line arguments for extras topic wrappers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save", action="store_true", help="Save output under output/figures/extras/<topic>.")
    return parser.parse_args(argv)


def topic_artifact_path(slug: str, stem: str, suffix: str) -> Path:
    """Return the output figure path for one extras artifact."""
    return ensure_dir(default_figure_dir() / "extras" / slug) / f"{stem}.{suffix}"


def main_visualize(slug: str, argv: Sequence[str] | None = None) -> int:
    """Run the static visualization CLI for one extras topic."""
    args = _parse_topic_args(argv, f"Visualize extras topic {slug}.")
    fig, demo = render_topic_figure(slug, mode="visualize")
    stem = f"visualize_{slug}"
    if args.save:
        figure = topic_artifact_path(slug, stem, "png")
        fig.savefig(figure, dpi=150)
        save_extra_data(
            slug,
            stem,
            demo.arrays,
            {"script": f"{stem}.py", "kind": "visualize", "family": demo.spec.family, "sections": list(demo.spec.sections), **demo.metadata},
            figures=[figure],
        )
        plt.close(fig)
    else:
        plt.show()
    return 0


def main_simulate(slug: str, argv: Sequence[str] | None = None) -> int:
    """Run the parameter-sweep simulation CLI for one extras topic."""
    args = _parse_topic_args(argv, f"Simulate extras topic {slug}.")
    fig, demo = render_topic_figure(slug, mode="simulate")
    stem = f"simulate_{slug}"
    if args.save:
        figure = topic_artifact_path(slug, stem, "png")
        fig.savefig(figure, dpi=150)
        save_extra_data(
            slug,
            stem,
            demo.arrays,
            {"script": f"{stem}.py", "kind": "simulate", "family": demo.spec.family, "sections": list(demo.spec.sections), **demo.metadata},
            figures=[figure],
        )
        plt.close(fig)
    else:
        plt.show()
    return 0


def main_animation(slug: str, argv: Sequence[str] | None = None) -> int:
    """Run the animation CLI for one extras topic."""
    args = _parse_topic_args(argv, f"Animate extras topic {slug}.")
    anim, raw, _metadata = build_topic_animation(slug)
    stem = f"animation_{slug}"
    if args.save:
        path = topic_artifact_path(slug, stem, "gif")
        save_animation(anim, path, fps=12, dpi=110)
    else:
        # Keep a reference to raw data for parity with saved mode and show.
        anim._raw_data = raw  # type: ignore[attr-defined]
        plt.show()
    return 0


def main_interactive(slug: str, argv: Sequence[str] | None = None) -> int:
    """Run the interactive CLI for one extras topic."""
    args = _parse_topic_args(argv, f"Interactively explore extras topic {slug}.")
    from .visualizations.interactive import interactive_topic_demo

    fig = interactive_topic_demo(slug)
    if args.save:
        print(f"interactive_{slug} is GUI-only; nothing to save.")
        plt.close(fig)
    else:
        plt.show()
    return 0


__all__ = [
    "ExtraTopicSpec",
    "TopicDemo",
    "EXTRA_TOPICS",
    "extra_topic_slugs",
    "extra_topic_spec",
    "extra_topics_by_family",
    "build_topic_demo",
    "render_topic_figure",
    "build_topic_animation",
    "topic_artifact_path",
    "main_visualize",
    "main_simulate",
    "main_animation",
    "main_interactive",
]
