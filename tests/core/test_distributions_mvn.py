"""Tests for the multivariate Gaussian helpers in ``core.distributions``."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.distributions import (
    diagonal_cov,
    isotropic_cov,
    mahalanobis_squared,
    mvn_log_pdf,
    mvn_pdf,
    mvn_sample,
)


class TestCovHelpers:
    def test_isotropic_validates(self) -> None:
        with pytest.raises(ValueError):
            isotropic_cov(2, 0.0)
        with pytest.raises(ValueError):
            isotropic_cov(0, 1.0)

    def test_diagonal_validates(self) -> None:
        with pytest.raises(ValueError):
            diagonal_cov(np.array([1.0, -0.1]))

    def test_isotropic_returns_diagonal(self) -> None:
        cov = isotropic_cov(3, 0.7)
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(np.diag(cov), [0.7, 0.7, 0.7])
        off = cov - np.diag(np.diag(cov))
        assert np.max(np.abs(off)) == 0.0

    def test_diagonal_returns_diagonal(self) -> None:
        cov = diagonal_cov(np.array([0.1, 0.5, 1.0]))
        np.testing.assert_allclose(np.diag(cov), [0.1, 0.5, 1.0])
        off = cov - np.diag(np.diag(cov))
        assert np.max(np.abs(off)) == 0.0


class TestMVN:
    def test_pdf_integrates_to_one_2d(self) -> None:
        grid = np.linspace(-5, 5, 121)
        XX, YY = np.meshgrid(grid, grid)
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
        density = mvn_pdf(pts, np.zeros(2), np.eye(2)).reshape(XX.shape)
        marg_y = np.trapezoid(density, grid, axis=0)
        total = np.trapezoid(marg_y, grid)
        assert total == pytest.approx(1.0, rel=1e-3)

    def test_log_pdf_matches_pdf(self) -> None:
        rng = np.random.default_rng(0)
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        pts = rng.normal(size=(7, 2))
        np.testing.assert_allclose(
            np.exp(mvn_log_pdf(pts, np.zeros(2), cov)),
            mvn_pdf(pts, np.zeros(2), cov),
            rtol=1e-12,
        )

    def test_log_pdf_single_vector(self) -> None:
        # Single-vector path takes a different branch from the batched one.
        cov = np.diag([1.0, 4.0])
        out = mvn_log_pdf(np.array([1.0, 2.0]), np.array([0.0, 0.0]), cov)
        assert np.ndim(out) == 0  # scalar
        assert np.isfinite(out)

    def test_sampling_reproduces_moments(self) -> None:
        rng = np.random.default_rng(0)
        cov = np.array([[1.0, 0.4], [0.4, 0.5]])
        samples = mvn_sample(np.array([1.0, -1.0]), cov, n=20000, rng=rng)
        np.testing.assert_allclose(samples.mean(axis=0), [1.0, -1.0], atol=0.05)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), cov, atol=0.05)

    def test_invalid_cov_shape(self) -> None:
        with pytest.raises(ValueError):
            mvn_log_pdf(np.zeros(2), np.zeros(2), np.eye(3))

    def test_non_symmetric_cov_raises(self) -> None:
        bad = np.array([[1.0, 0.0], [0.5, 1.0]])
        with pytest.raises(ValueError):
            mvn_log_pdf(np.zeros(2), np.zeros(2), bad)


class TestMahalanobis:
    def test_aligned_with_axes(self) -> None:
        cov = np.diag([4.0, 1.0])
        x = np.array([2.0, 1.0])
        m = mahalanobis_squared(x, np.zeros(2), cov)
        # (2/2)^2 + (1/1)^2 = 2
        assert m == pytest.approx(2.0)

    def test_zero_distance_at_mean(self) -> None:
        m = mahalanobis_squared(np.array([1.0, -2.0]),
                                np.array([1.0, -2.0]),
                                np.eye(2))
        assert m == pytest.approx(0.0)

    def test_batched_input(self) -> None:
        rng = np.random.default_rng(0)
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        pts = rng.normal(size=(5, 2))
        d = mahalanobis_squared(pts, np.zeros(2), cov)
        assert d.shape == (5,)
        assert np.all(d >= 0)
