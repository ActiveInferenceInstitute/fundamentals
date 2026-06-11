"""Tests for Appendix B/C helper modules."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    bayes_factor,
    bayes_posterior_from_likelihood,
    bayesian_model_average,
    bayesian_model_expansion,
    bayesian_model_reduction,
    colored_noise_precision,
    dirac_delta_grid,
    dirichlet_mean,
    dirichlet_variance,
    euler_integrate,
    finite_difference_derivative,
    gamma_pdf,
    joint_from_likelihood_prior,
    jensen_gap,
    kronecker_delta,
    marginalize,
    maximum_entropy_distribution,
    model_posterior,
    mutual_information,
    normalize_categorical,
    sample_colored_noise,
    squared_exponential_covariance,
)


def test_column_oriented_likelihood_posterior_rejects_transpose_mistake() -> None:
    """Appendix B's A[o, s] convention must not silently behave like A.T."""
    likelihood = np.array([[0.9, 0.2], [0.1, 0.8]])
    prior = np.array([0.25, 0.75])
    posterior = bayes_posterior_from_likelihood(likelihood, prior, observation=1)
    expected = likelihood[1] * prior
    expected = expected / expected.sum()
    np.testing.assert_allclose(posterior, expected)
    assert not np.allclose(posterior, normalize_categorical(likelihood.T[1] * prior))


def test_probability_distribution_helpers_are_normalized() -> None:
    """Appendix C probability helpers produce finite normalized quantities."""
    probs = normalize_categorical([2.0, 3.0, 5.0])
    np.testing.assert_allclose(probs.sum(), 1.0)
    np.testing.assert_allclose(maximum_entropy_distribution(4), np.full(4, 0.25))
    np.testing.assert_allclose(dirichlet_mean([1.0, 2.0, 3.0]), [1 / 6, 2 / 6, 3 / 6])
    assert np.all(dirichlet_variance([1.0, 2.0, 3.0]) > 0.0)
    assert np.all(gamma_pdf(np.array([0.5, 1.0, 2.0]), shape=2.0, rate=1.0) > 0.0)


def test_joint_marginal_information_and_identities() -> None:
    """Joint tables, marginalization, information, and delta helpers are consistent."""
    likelihood = np.array([[0.8, 0.1], [0.2, 0.9]])
    prior = np.array([0.6, 0.4])
    joint = joint_from_likelihood_prior(likelihood, prior)
    np.testing.assert_allclose(joint.sum(), 1.0)
    np.testing.assert_allclose(marginalize(joint, axis=1), prior)
    assert mutual_information(joint) > 0.0
    assert jensen_gap([0.5, 0.5], [1.0, 3.0]) > 0.0
    assert kronecker_delta(2, 2) == 1
    assert kronecker_delta(2, 3) == 0
    grid = np.linspace(-1.0, 1.0, 11)
    assert np.trapezoid(dirac_delta_grid(grid, 0.1), grid) == pytest.approx(1.0, rel=0.25)


def test_euler_and_colored_noise_helpers_are_finite() -> None:
    """Dynamical-system and colored-noise helpers return stable finite arrays."""
    time, states = euler_integrate(lambda _t, x: -x, [1.0], dt=0.1, steps=10)
    assert time.shape == (11,)
    assert states[-1, 0] < states[0, 0]
    cov = squared_exponential_covariance(time, length_scale=0.5)
    precision = colored_noise_precision(time, length_scale=0.5)
    np.testing.assert_allclose(cov, cov.T)
    np.testing.assert_allclose(precision, precision.T, atol=1e-7)
    sample = sample_colored_noise(time, length_scale=0.5, rng=np.random.default_rng(4))
    derivative = finite_difference_derivative(sample, time)
    assert np.all(np.isfinite(derivative))


def test_model_comparison_helpers_have_expected_directionality() -> None:
    """Model posterior and Bayes-factor helpers prefer higher log evidence."""
    log_evidence = np.array([-3.0, -1.0, -2.0])
    posterior = model_posterior(log_evidence)
    assert int(np.argmax(posterior)) == 1
    assert bayes_factor(-1.0, -3.0) > 1.0
    assert bayesian_model_average([0.0, 10.0, 5.0], log_evidence) > 5.0
    assert bayesian_model_reduction(-10.0, complexity_delta=1.0) == pytest.approx(-9.0)
    assert bayesian_model_expansion(-10.0, accuracy_gain=2.0, complexity_cost=0.5) == pytest.approx(-8.5)
