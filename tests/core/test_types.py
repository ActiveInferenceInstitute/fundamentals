"""Tests for ``core.types`` — defensive validators and shape conventions."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.types import (
    assert_cov,
    assert_probabilities,
)


class TestAssertCov:
    def test_accepts_valid_cov(self) -> None:
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        out = assert_cov(cov, dim=2)
        np.testing.assert_array_equal(out, cov)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            assert_cov(np.eye(2), dim=3)

    def test_rejects_non_symmetric(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            assert_cov(np.array([[1.0, 0.0], [0.5, 1.0]]), dim=2)

    def test_rejects_indefinite(self) -> None:
        with pytest.raises(ValueError, match="positive definite"):
            assert_cov(np.array([[1.0, 2.0], [2.0, 1.0]]), dim=2)

    def test_isotropic_passes(self) -> None:
        out = assert_cov(np.eye(5), dim=5, name="Σ_test")
        assert out.shape == (5, 5)


class TestAssertProbabilities:
    def test_accepts_valid_distribution(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        out = assert_probabilities(p)
        np.testing.assert_allclose(out, p)

    def test_rejects_negative_entries(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            assert_probabilities(np.array([0.3, -0.1, 0.8]))

    def test_rejects_non_unit_sum(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            assert_probabilities(np.array([0.1, 0.2, 0.3]))

    def test_clips_tiny_negatives(self) -> None:
        # Floating-point roundoff can produce -1e-12 entries; the helper
        # should clip them rather than raise.
        p = np.array([0.5, 0.5, -1e-12])
        # The sum is still ~1, and the first negative value is within tol.
        out = assert_probabilities(p, tol=1e-9)
        assert np.all(out >= 0)
