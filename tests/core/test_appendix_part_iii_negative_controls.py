"""Negative controls for Appendix B-D and Part III helper validation branches."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.appendix_math import (
    bayes_posterior_from_likelihood,
    dirac_delta_grid,
    dirichlet_mean,
    dirichlet_variance,
    euler_integrate,
    gamma_pdf,
    joint_from_likelihood_prior,
    jensen_gap,
    marginalize,
    maximum_entropy_distribution,
    mutual_information,
    normalize_categorical,
)
from active_inference.core.ergodic import (
    EntropyBound,
    density_entropy,
    entropy_upper_bound_from_vfe,
    ergodic_density,
    ergodic_ou_trajectory,
)
from active_inference.core.factor_graph import (
    FactorGraphChain,
    active_inference_factor_messages,
    backward_smoothing_messages,
    categorical_factor_message,
    hybrid_model_bridge,
    learning_attention_message,
    marginal_message_passing,
    normalize_message,
    sum_product_chain,
    variational_message_update,
)
from active_inference.core.model_comparison import (
    bayesian_model_average,
    bayesian_model_expansion,
    bayesian_model_reduction,
    log_bayes_factor,
    model_posterior,
)
from active_inference.core.noise import (
    colored_noise_precision,
    finite_difference_derivative,
    sample_colored_noise,
    squared_exponential_covariance,
)
from active_inference.core.pomdp_extensions import (
    forget_dirichlet_counts,
    habit_prior_from_counts,
    path_based_policy_scores,
    sophisticated_policy_trace,
    structure_log_evidence,
    time_dependent_preferences,
    tree_policy_search,
    update_preference_counts,
)
from active_inference.estimators.pomdp_extensions import make_line_world


def test_appendix_math_rejects_invalid_probability_shapes_and_values() -> None:
    """Appendix math helpers reject impossible categorical and tensor inputs."""
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_categorical([])
    with pytest.raises(ValueError, match="finite"):
        normalize_categorical([np.nan, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        normalize_categorical([1.0, -0.1])
    with pytest.raises(ValueError, match="positive mass"):
        normalize_categorical([0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        joint_from_likelihood_prior(np.ones((2, 3)), [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        marginalize(np.array([[0.2, -0.1], [0.3, 0.6]]), axis=0)
    np.testing.assert_allclose(marginalize(np.ones((2, 3)), axis=-1), [1.0 / 3.0] * 3)
    with pytest.raises(ValueError, match="out of bounds"):
        marginalize(np.ones((2, 2)), axis=3)
    with pytest.raises(ValueError, match="shape"):
        bayes_posterior_from_likelihood(np.ones((2, 3)), [0.5, 0.5], 0)
    with pytest.raises(ValueError, match="out of bounds"):
        bayes_posterior_from_likelihood(np.eye(2), [0.5, 0.5], 3)


def test_appendix_math_rejects_invalid_distribution_and_dynamics_inputs() -> None:
    """Distribution, Jensen, Euler, and identity helpers reject invalid inputs."""
    with pytest.raises(ValueError, match="positive"):
        gamma_pdf([1.0], shape=0.0, rate=1.0)
    with pytest.raises(ValueError, match="positive 1-D"):
        dirichlet_mean([1.0, 0.0])
    with pytest.raises(ValueError, match="positive 1-D"):
        dirichlet_variance([[1.0, 2.0]])
    with pytest.raises(ValueError, match="2-D"):
        mutual_information([0.5, 0.5])
    with pytest.raises(ValueError, match="positive"):
        maximum_entropy_distribution(0)
    with pytest.raises(ValueError, match="dt"):
        euler_integrate(lambda _t, x: x, [1.0], dt=0.0, steps=2)
    with pytest.raises(ValueError, match="same shape"):
        euler_integrate(lambda _t, _x: np.array([1.0, 2.0]), [1.0], dt=0.1, steps=2)
    with pytest.raises(ValueError, match="share a shape"):
        jensen_gap([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        dirac_delta_grid([[0.0, 1.0]], 0.0)
    with pytest.raises(ValueError, match="positive spacing"):
        dirac_delta_grid([1.0, 1.0], 1.0)


def test_factor_graph_helpers_reject_axis_and_message_mistakes() -> None:
    """Factor-graph helpers reject malformed messages, axes, and factor shapes."""
    with pytest.raises(ValueError, match="empty"):
        normalize_message([])
    with pytest.raises(ValueError, match="finite"):
        normalize_message([np.inf])
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_message([[0.5, 0.5]])
    with pytest.raises(ValueError, match="non-negative"):
        normalize_message([0.5, -0.1])
    with pytest.raises(ValueError, match="positive mass"):
        normalize_message([0.0, 0.0])
    factor = np.ones((2, 2))
    with pytest.raises(ValueError, match="out of bounds"):
        categorical_factor_message(factor, [np.ones(2)], target_axis=3)
    with pytest.raises(ValueError, match="every non-target"):
        categorical_factor_message(factor, [], target_axis=0)
    with pytest.raises(ValueError, match="length"):
        categorical_factor_message(factor, [np.ones(3)], target_axis=0)
    with pytest.raises(ValueError, match="shape"):
        sum_product_chain([0.5, 0.5], np.eye(2), [0.5, 0.5])
    with pytest.raises(ValueError, match="state dimension"):
        sum_product_chain([0.5, 0.5], np.eye(2), np.ones((2, 3)))
    with pytest.raises(ValueError, match="T-1"):
        sum_product_chain([0.5, 0.5], np.ones((3, 2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="out of bounds"):
        variational_message_update(np.ones((2, 2)), [np.ones(2)], target_axis=5)
    with pytest.raises(ValueError, match="cover non-target"):
        variational_message_update(np.ones((2, 2)), [], target_axis=0)
    with pytest.raises(ValueError, match="length"):
        variational_message_update(np.ones((2, 2)), [np.ones(3)], target_axis=0)


def test_factor_graph_structures_reject_bad_shapes_and_nonfinite_precision() -> None:
    """Higher-level factor-graph containers and messages validate their inputs."""
    with pytest.raises(ValueError, match="transition"):
        FactorGraphChain(np.ones(2), np.ones((3, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="likelihoods"):
        FactorGraphChain(np.ones(2), np.eye(2), np.ones((2, 3)))
    with pytest.raises(ValueError, match="likelihoods"):
        backward_smoothing_messages(np.eye(2), [1.0, 1.0])
    with pytest.raises(ValueError, match="transitions"):
        backward_smoothing_messages(np.ones((3, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        hybrid_model_bridge([[1.0, 2.0]], [0.5, 0.5])
    with pytest.raises(ValueError, match="shape"):
        marginal_message_passing([np.ones((2, 3))], [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        learning_attention_message([1.0, 2.0], float("inf"))
    messages = active_inference_factor_messages(np.eye(2), np.eye(2), [0.7, 0.3])
    assert set(messages) == {"prior", "prediction", "observation", "posterior"}


def test_part_iii_pomdp_extensions_reject_invalid_inputs() -> None:
    """POMDP extension helpers reject malformed preferences, policies, and counts."""
    with pytest.raises(ValueError, match="shape"):
        time_dependent_preferences([1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        forget_dirichlet_counts([-1.0, 2.0], rate=0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        forget_dirichlet_counts([1.0, 2.0], rate=1.5)
    with pytest.raises(ValueError, match="1-D"):
        structure_log_evidence([[1.0]], complexity=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        structure_log_evidence([1.0, 2.0], complexity=-1.0)
    with pytest.raises(ValueError, match="non-negative 1-D"):
        update_preference_counts([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="learning_rate"):
        update_preference_counts([1.0, 1.0], [0], learning_rate=-0.1)
    with pytest.raises(ValueError, match="out of bounds"):
        update_preference_counts([1.0, 1.0], [2])
    with pytest.raises(ValueError, match="non-negative 1-D"):
        habit_prior_from_counts([-1.0, 1.0])
    with pytest.raises(ValueError, match="1-D"):
        path_based_policy_scores([[1.0, 2.0]], [[0]])
    with pytest.raises(ValueError, match="match"):
        path_based_policy_scores([1.0, 2.0], [[0]], terminal_costs=[0.0])
    with pytest.raises(ValueError, match="non-empty"):
        path_based_policy_scores([1.0, 2.0], [[]])
    with pytest.raises(ValueError, match="out of bounds"):
        path_based_policy_scores([1.0, 2.0], [[2]])
    model = make_line_world()
    with pytest.raises(ValueError, match="shape"):
        tree_policy_search(model, model.D, [0, 1], model.C)
    with pytest.raises(ValueError, match="non-empty"):
        sophisticated_policy_trace(model, model.D, [], model.C)


def test_ergodic_model_comparison_and_noise_negative_controls() -> None:
    """Appendix support modules reject non-finite and dimensionally invalid inputs."""
    with pytest.raises(ValueError, match="finite"):
        model_posterior([0.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        log_bayes_factor(0.0, np.inf)
    with pytest.raises(ValueError, match="share a shape"):
        bayesian_model_average([1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        bayesian_model_reduction(np.inf, complexity_delta=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        bayesian_model_expansion(0.0, accuracy_gain=1.0, complexity_cost=-0.1)
    with pytest.raises(ValueError, match="at least two"):
        squared_exponential_covariance([0.0], length_scale=1.0)
    with pytest.raises(ValueError, match="positive"):
        squared_exponential_covariance([0.0, 1.0], length_scale=0.0)
    with pytest.raises(ValueError, match="positive"):
        colored_noise_precision([0.0, 1.0], length_scale=1.0, jitter=0.0)
    with pytest.raises(ValueError, match="share a shape"):
        finite_difference_derivative([0.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        finite_difference_derivative([0.0, 1.0], [1.0, 1.0])
    assert sample_colored_noise([0.0, 1.0, 2.0], length_scale=1.0).shape == (3,)


def test_ergodic_helpers_reject_invalid_bounds_and_entropy_inputs() -> None:
    """Ergodic-density helpers reject invalid bounds, mass, and entropy bounds."""
    with pytest.raises(ValueError, match="finite"):
        EntropyBound(np.nan, 1.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        EntropyBound(1.0, 0.5, -1.0)
    x, density = ergodic_density([1.0, 1.0, 1.0], bins=4)
    assert x.shape == density.shape
    with pytest.raises(ValueError, match="bounds"):
        ergodic_density([0.0, 1.0], bounds=(1.0, 0.0))
    with pytest.raises(ValueError, match="same shape"):
        density_entropy([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        density_entropy([0.0, 1.0], [1.0, -1.0])
    with pytest.raises(ValueError, match="positive mass"):
        density_entropy([0.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        entropy_upper_bound_from_vfe(np.nan, 1.0)
    with pytest.raises(ValueError, match="greater than or equal"):
        entropy_upper_bound_from_vfe(2.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        ergodic_ou_trajectory(n_steps=0)
    with pytest.raises(ValueError, match="drift"):
        ergodic_ou_trajectory(drift=0.0)
    with pytest.raises(ValueError, match="diffusion"):
        ergodic_ou_trajectory(diffusion=-0.1)
