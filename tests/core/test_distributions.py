"""Tests for ``active_inference.core.distributions`` — univariate helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.distributions import (
    dirac_like_pdf,
    gaussian_log_pdf,
    gaussian_pdf,
    normalize_density,
    uniform_pdf,
)


class TestGaussian:
    def test_pdf_integrates_to_one(self) -> None:
        x = np.linspace(-10, 10, 4001)
        pdf = gaussian_pdf(x, mu=0.5, sigma2=1.5)
        assert np.trapezoid(pdf, x) == pytest.approx(1.0, rel=1e-6)

    def test_pdf_peak_location(self) -> None:
        x = np.linspace(-3, 7, 1001)
        pdf = gaussian_pdf(x, mu=2.0, sigma2=0.5)
        assert x[int(np.argmax(pdf))] == pytest.approx(2.0, abs=1e-2)

    def test_log_pdf_matches_pdf(self) -> None:
        x = np.linspace(-2, 5, 50)
        np.testing.assert_allclose(
            np.exp(gaussian_log_pdf(x, mu=1.2, sigma2=2.5)),
            gaussian_pdf(x, mu=1.2, sigma2=2.5),
            rtol=1e-12,
        )

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            gaussian_pdf(0.0, 0.0, -1.0)
        with pytest.raises(ValueError):
            gaussian_log_pdf(0.0, 0.0, 0.0)

    def test_broadcasting(self) -> None:
        x = np.linspace(0, 1, 5)
        mu = np.array([[0.0], [1.0]])
        out = gaussian_pdf(x, mu, sigma2=0.1)
        assert out.shape == (2, 5)


class TestUniform:
    def test_uniform_normalization(self) -> None:
        x = np.linspace(-1, 6, 4001)
        pdf = uniform_pdf(x, low=0.0, high=5.0)
        assert np.trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-3)

    def test_uniform_zero_outside(self) -> None:
        x = np.array([-0.1, 0.0, 2.5, 5.0, 5.1])
        pdf = uniform_pdf(x, 0.0, 5.0)
        assert pdf[0] == 0.0
        assert pdf[-1] == 0.0
        assert pdf[1:-1].min() > 0

    def test_uniform_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            uniform_pdf(0.0, 1.0, 1.0)


class TestDiracLike:
    def test_concentrates_on_location(self) -> None:
        x = np.linspace(-5, 5, 2001)
        pdf = dirac_like_pdf(x, location=1.5, epsilon=0.01)
        assert x[int(np.argmax(pdf))] == pytest.approx(1.5, abs=1e-2)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError):
            dirac_like_pdf(0.0, 0.0, epsilon=0.0)


class TestNormalize:
    def test_normalize_unit_mass(self) -> None:
        x = np.linspace(-3, 3, 1001)
        unnorm = np.exp(-x ** 2)
        norm = normalize_density(unnorm, x)
        assert np.trapezoid(norm, x) == pytest.approx(1.0, rel=1e-6)

    def test_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_density(np.zeros(5), np.linspace(0, 1, 5))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_density(np.ones(3), np.linspace(0, 1, 5))
