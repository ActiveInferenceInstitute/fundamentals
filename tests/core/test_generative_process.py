"""Tests for ``core.generative_process`` (univariate + multivariate)."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.distributions import diagonal_cov, isotropic_cov
from active_inference.core.generative_process import (
    GenerativeProcess,
    LinearGaussianMVProcess,
    LinearGaussianProcess,
)


class TestUnivariate:
    def test_mean_is_linear(self) -> None:
        proc = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.0)
        np.testing.assert_allclose(
            proc.mean(np.array([0.0, 1.0, 2.0])), [3.0, 5.0, 7.0]
        )

    def test_zero_noise_is_deterministic(self) -> None:
        proc = LinearGaussianProcess(
            beta0=1.0, beta1=1.0, sigma2_y=0.0,
            rng=np.random.default_rng(0),
        )
        s = proc.sample(2.0, n=5)
        assert np.all(s == 3.0)

    def test_noise_scales_correctly(self) -> None:
        rng = np.random.default_rng(42)
        proc = LinearGaussianProcess(beta0=0.0, beta1=1.0, sigma2_y=4.0, rng=rng)
        s = proc.sample(0.0, n=20000)
        assert s.std() == pytest.approx(2.0, rel=0.05)

    def test_nonlinear_psi(self) -> None:
        proc = LinearGaussianProcess(
            beta0=0.0, beta1=1.0, sigma2_y=0.0,
            nonlinear=lambda x: x ** 2,
        )
        np.testing.assert_allclose(
            proc.mean(np.array([1.0, 2.0, 3.0])), [1.0, 4.0, 9.0]
        )

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianProcess(sigma2_y=-1.0)


class TestGenericProcess:
    def test_default_rng_is_set(self) -> None:
        gp = GenerativeProcess(g_E=lambda x: x, sigma2_y=0.0)
        assert isinstance(gp.rng, np.random.Generator)

    def test_sample_shapes(self) -> None:
        gp = GenerativeProcess(g_E=lambda x: x, sigma2_y=0.0)
        s = gp.sample(2.0, n=3)
        np.testing.assert_allclose(s, [2.0, 2.0, 2.0])

    def test_kwargs_threaded_through_g(self) -> None:
        # theta is passed as keyword args.
        gp = GenerativeProcess(
            g_E=lambda x, slope=1.0, intercept=0.0: slope * x + intercept,
            theta={"slope": 2.0, "intercept": 1.0},
            sigma2_y=0.0,
        )
        np.testing.assert_allclose(gp.mean(3.0), 7.0)


class TestMultivariate:
    def test_mean_uses_mixing_matrix(self) -> None:
        Theta = np.array([[1.0, 2.0], [-1.0, 3.0]])
        proc = LinearGaussianMVProcess(
            Theta=Theta, cov_y=np.eye(2),
            b=np.array([0.5, -0.5]),
        )
        out = proc.mean(np.array([1.0, 1.0]))
        np.testing.assert_allclose(out, [3.5, 1.5])

    def test_mean_batched_input(self) -> None:
        Theta = np.array([[1.0, 0.5], [0.0, 1.0]])
        proc = LinearGaussianMVProcess(Theta=Theta, cov_y=np.eye(2))
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = proc.mean(X)
        # row 0: [1*1+0.5*2, 0*1+1*2] = [2, 2]
        # row 1: [1*3+0.5*4, 0*3+1*4] = [5, 4]
        np.testing.assert_allclose(out, [[2.0, 2.0], [5.0, 4.0]])

    def test_sample_shape_single_state(self) -> None:
        Theta = np.eye(2)
        proc = LinearGaussianMVProcess(
            Theta=Theta, cov_y=isotropic_cov(2, 0.1),
            rng=np.random.default_rng(0),
        )
        s = proc.sample(np.array([0.5, 0.5]), n=10)
        assert s.shape == (10, 2)

    def test_sample_shape_batched_state(self) -> None:
        Theta = np.eye(2)
        proc = LinearGaussianMVProcess(
            Theta=Theta, cov_y=isotropic_cov(2, 0.1),
            rng=np.random.default_rng(0),
        )
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        s = proc.sample(X, n=4)
        assert s.shape == (2, 4, 2)

    def test_invalid_theta_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianMVProcess(
                Theta=np.array([1.0, 2.0]), cov_y=np.eye(2),
            )

    def test_invalid_offset_length_raises(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianMVProcess(
                Theta=np.eye(2), cov_y=np.eye(2),
                b=np.zeros(3),
            )

    def test_diagonal_noise_uncorrelated(self) -> None:
        rng = np.random.default_rng(1)
        cov_y = diagonal_cov(np.array([0.5, 1.5]))
        proc = LinearGaussianMVProcess(
            Theta=np.eye(2), cov_y=cov_y, rng=rng,
        )
        samples = proc.sample(np.zeros(2), n=20000)
        emp = np.cov(samples, rowvar=False)
        # Off-diagonal should be ~0; diagonal should match cov_y.
        np.testing.assert_allclose(np.diag(emp), [0.5, 1.5], rtol=0.05)
        assert abs(emp[0, 1]) < 0.05
