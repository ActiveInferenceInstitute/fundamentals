"""Tests for ``core.generative_model`` (univariate + multivariate)."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.distributions import isotropic_cov
from active_inference.core.generative_model import (
    LinearGaussianModel,
    LinearGaussianMVModel,
)


class TestLinearGaussianModel:
    def test_predict_mean_linear(self) -> None:
        m = LinearGaussianModel(beta0=1.0, beta1=2.0, sigma2_y=0.5,
                                m_x=0.0, s2_x=1.0)
        np.testing.assert_allclose(m.predict_mean(np.array([0.0, 1.0])),
                                   [1.0, 3.0])

    def test_predict_mean_nonlinear(self) -> None:
        m = LinearGaussianModel(
            beta0=0.0, beta1=1.0, sigma2_y=0.5,
            m_x=0.0, s2_x=1.0,
            psi=lambda x: x ** 2,
        )
        np.testing.assert_allclose(m.predict_mean(np.array([1.0, 2.0])),
                                   [1.0, 4.0])

    def test_log_likelihood_finite(self) -> None:
        m = LinearGaussianModel(beta0=0.0, beta1=1.0, sigma2_y=0.5,
                                m_x=0.0, s2_x=1.0)
        x_grid = np.linspace(-3, 3, 50)
        out = m.log_likelihood(0.0, x_grid)
        assert np.all(np.isfinite(out))

    def test_prior_normalized_gaussian(self) -> None:
        m = LinearGaussianModel(beta0=0.0, beta1=1.0, sigma2_y=1.0,
                                m_x=2.0, s2_x=0.5)
        x = np.linspace(-10, 14, 4001)
        prior = m.prior(x)
        assert np.trapezoid(prior, x) == pytest.approx(1.0, rel=1e-5)

    def test_uniform_prior_zero_outside(self) -> None:
        m = LinearGaussianModel(
            beta0=0.0, beta1=1.0, sigma2_y=1.0,
            prior_kind="uniform", uniform_low=0.0, uniform_high=5.0,
        )
        x = np.array([-1.0, 0.5, 6.0])
        prior = m.prior(x)
        assert prior[0] == 0.0
        assert prior[2] == 0.0
        assert prior[1] > 0.0

    def test_log_likelihood_batch_sums(self) -> None:
        m = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.5,
                                m_x=2.0, s2_x=1.0)
        x_grid = np.array([1.0, 2.0, 3.0])
        ys = np.array([5.0, 7.0, 9.0])
        batch = m.log_likelihood_batch(ys, x_grid)
        manual = sum(m.log_likelihood(float(y), x_grid) for y in ys)
        np.testing.assert_allclose(batch, manual)

    def test_likelihood_deterministic_branch(self) -> None:
        m = LinearGaussianModel(beta0=0.0, beta1=1.0, sigma2_y=1e-6,
                                m_x=0.0, s2_x=1.0)
        x_grid = np.linspace(-3, 3, 200)
        # The deterministic likelihood is a near-Dirac centered on x.
        out = m.likelihood_deterministic(2.0, x_grid)
        assert x_grid[int(np.argmax(out))] == pytest.approx(2.0, abs=5e-2)

    def test_invalid_sigma2_y_raises(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianModel(sigma2_y=0.0)

    def test_invalid_uniform_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianModel(prior_kind="uniform",
                                uniform_low=1.0, uniform_high=1.0)

    def test_unknown_prior_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            m = LinearGaussianModel(prior_kind="laplace")
            m.log_prior(np.array([0.0]))


class TestLinearGaussianMVModel:
    def test_predict_mean_single_vector(self) -> None:
        m = LinearGaussianMVModel(
            Theta=np.array([[1.0, 2.0], [3.0, 4.0]]),
            cov_y=np.eye(2),
            mx=np.zeros(2), cov_x=np.eye(2),
            b=np.array([0.5, -0.5]),
        )
        out = m.predict_mean(np.array([1.0, 1.0]))
        np.testing.assert_allclose(out, [3.5, 6.5])

    def test_predict_mean_batched(self) -> None:
        m = LinearGaussianMVModel(
            Theta=np.eye(2), cov_y=np.eye(2),
            mx=np.zeros(2), cov_x=np.eye(2),
        )
        out = m.predict_mean(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 4.0]])

    def test_log_likelihood_single(self) -> None:
        m = LinearGaussianMVModel(
            Theta=np.eye(2), cov_y=isotropic_cov(2, 1.0),
            mx=np.zeros(2), cov_x=np.eye(2),
        )
        ll = m.log_likelihood(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        # log N(0; 0, I) in 2D = -0.5*(2*log(2π))
        assert ll == pytest.approx(-np.log(2 * np.pi), rel=1e-10)

    def test_log_prior_finite(self) -> None:
        m = LinearGaussianMVModel(
            Theta=np.eye(2), cov_y=np.eye(2),
            mx=np.array([1.0, 2.0]), cov_x=np.eye(2),
        )
        out = m.log_prior(np.array([0.0, 0.0]))
        assert np.isfinite(out)

    def test_dimensions_validate(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianMVModel(
                Theta=np.eye(2), cov_y=np.eye(3),
                mx=np.zeros(2), cov_x=np.eye(2),
            )
        with pytest.raises(ValueError):
            LinearGaussianMVModel(
                Theta=np.eye(2), cov_y=np.eye(2),
                mx=np.zeros(3), cov_x=np.eye(2),
            )
