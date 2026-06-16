"""Tests for colored-noise covariance and derivative helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.noise import (
    colored_noise_precision,
    finite_difference_derivative,
    sample_colored_noise,
    squared_exponential_covariance,
)


class TestColoredNoise:
    def test_squared_exponential_covariance_is_symmetric_with_variance_diagonal(self) -> None:
        time = np.linspace(0.0, 1.0, 5)
        cov = squared_exponential_covariance(time, length_scale=0.4, variance=2.0)
        np.testing.assert_allclose(cov, cov.T)
        np.testing.assert_allclose(np.diag(cov), np.full(time.size, 2.0))
        assert cov[0, -1] < cov[0, 1]

    def test_precision_is_symmetric_positive_definite(self) -> None:
        precision = colored_noise_precision([0.0, 0.5, 1.0], length_scale=0.3, variance=1.5)
        np.testing.assert_allclose(precision, precision.T)
        assert np.all(np.linalg.eigvalsh(precision) > 0.0)

    def test_sampling_is_reproducible_with_explicit_rng(self) -> None:
        time = np.linspace(0.0, 1.0, 4)
        a = sample_colored_noise(time, length_scale=0.25, rng=np.random.default_rng(42))
        b = sample_colored_noise(time, length_scale=0.25, rng=np.random.default_rng(42))
        np.testing.assert_allclose(a, b)

    def test_rejects_invalid_scales(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            squared_exponential_covariance([0.0, 1.0], length_scale=0.0)
        with pytest.raises(ValueError, match="positive"):
            colored_noise_precision([0.0, 1.0], length_scale=1.0, jitter=0.0)


class TestFiniteDifferenceDerivative:
    def test_derivative_matches_quadratic_interior(self) -> None:
        time = np.linspace(-1.0, 1.0, 9)
        derivative = finite_difference_derivative(time**2, time)
        np.testing.assert_allclose(derivative[1:-1], 2.0 * time[1:-1], atol=1e-12)

    def test_requires_matching_strictly_increasing_vectors(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            finite_difference_derivative([1.0, 2.0], [0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            finite_difference_derivative([1.0, 2.0], [0.0, 0.0])
