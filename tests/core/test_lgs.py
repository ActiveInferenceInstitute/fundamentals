"""Tests for ``core.lgs`` — closed-form Linear Gaussian System inference."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.distributions import isotropic_cov
from active_inference.core.generative_process import LinearGaussianMVProcess
from active_inference.core.lgs import LGSPosterior, LinearGaussianSystem


@pytest.fixture
def lgs_identity() -> LinearGaussianSystem:
    """Sensor-fusion case: y = x + ω with isotropic noise."""
    return LinearGaussianSystem(
        Theta=np.eye(2),
        cov_y=isotropic_cov(2, 0.1),
        mx=np.array([0.5, 0.5]),
        cov_x=isotropic_cov(2, 1.0),
    )


class TestPosteriorBatch:
    def test_recovers_truth_at_high_n(self) -> None:
        rng = np.random.default_rng(1)
        Theta = np.eye(2)
        cov_y = isotropic_cov(2, 0.1)
        proc = LinearGaussianMVProcess(Theta=Theta, cov_y=cov_y, rng=rng)
        true_x = np.array([0.4, 0.6])
        Y = proc.sample(true_x, n=200).reshape(200, 2)

        lgs = LinearGaussianSystem(
            Theta=Theta, cov_y=cov_y,
            mx=np.array([0.5, 0.5]), cov_x=isotropic_cov(2, 1.0),
        )
        post = lgs.posterior_batch(Y)
        np.testing.assert_allclose(post.mean, true_x, atol=0.1)
        assert np.all(np.diag(post.cov) < 1.0)  # tighter than the prior

    def test_single_observation_matches_batch_of_one(self) -> None:
        rng = np.random.default_rng(2)
        Theta = np.array([[1.0, 0.5], [0.0, 1.0]])
        cov_y = isotropic_cov(2, 0.4)
        lgs = LinearGaussianSystem(
            Theta=Theta, cov_y=cov_y,
            mx=np.zeros(2), cov_x=np.eye(2) * 0.5,
        )
        y = rng.normal(size=2)
        single = lgs.posterior(y)
        batched = lgs.posterior_batch(y.reshape(1, 2))
        np.testing.assert_allclose(single.mean, batched.mean)
        np.testing.assert_allclose(single.cov, batched.cov)

    def test_offset_b_is_subtracted(self) -> None:
        # When b is non-zero, y = x + b + ω → posterior should still center on x.
        rng = np.random.default_rng(3)
        true_x = np.array([1.0, 2.0])
        b = np.array([0.5, -0.5])
        proc = LinearGaussianMVProcess(
            Theta=np.eye(2), cov_y=isotropic_cov(2, 0.05),
            b=b, rng=rng,
        )
        Y = proc.sample(true_x, n=500).reshape(500, 2)

        lgs = LinearGaussianSystem(
            Theta=np.eye(2), cov_y=isotropic_cov(2, 0.05),
            mx=true_x, cov_x=isotropic_cov(2, 0.5), b=b,
        )
        post = lgs.posterior_batch(Y)
        np.testing.assert_allclose(post.mean, true_x, atol=0.05)


class TestPosteriorObject:
    def test_std_matches_diag_sqrt(self) -> None:
        post = LGSPosterior(
            mean=np.zeros(2),
            cov=np.diag([0.04, 0.16]),
        )
        np.testing.assert_allclose(post.std(), [0.2, 0.4])

    def test_precision_matches_inverse(self) -> None:
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        post = LGSPosterior(mean=np.zeros(2), cov=cov)
        np.testing.assert_allclose(post.precision, np.linalg.inv(cov))

    def test_sample_recovers_moments(self) -> None:
        rng = np.random.default_rng(0)
        cov = np.array([[1.0, 0.3], [0.3, 0.5]])
        mean = np.array([1.0, -1.0])
        post = LGSPosterior(mean=mean, cov=cov)
        samples = post.sample(n=20000, rng=rng)
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.05)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), cov, atol=0.05)

    def test_summary_string(self) -> None:
        post = LGSPosterior(mean=np.array([1.0, 2.0]),
                            cov=np.diag([0.04, 0.09]))
        s = post.summary()
        assert "LGSPosterior" in s
        assert "mean" in s and "std" in s


class TestSampleObservations:
    def test_recovers_likelihood_mean(self) -> None:
        rng = np.random.default_rng(0)
        Theta = np.array([[1.0, 0.5], [0.0, 1.0]])
        cov_y = np.diag([0.05, 0.05])
        lgs = LinearGaussianSystem(
            Theta=Theta, cov_y=cov_y,
            mx=np.zeros(2), cov_x=np.eye(2),
            b=np.array([0.5, -0.5]),
        )
        x = np.array([1.0, 1.0])
        samples = lgs.sample_observations(x, n=20000, rng=rng)
        expected = Theta @ x + np.array([0.5, -0.5])
        np.testing.assert_allclose(samples.mean(axis=0), expected, atol=0.02)

    def test_invalid_batched_input(self) -> None:
        lgs = LinearGaussianSystem(
            Theta=np.eye(2), cov_y=np.eye(2),
            mx=np.zeros(2), cov_x=np.eye(2),
        )
        with pytest.raises(ValueError):
            lgs.sample_observations(np.array([[0.0, 0.0], [1.0, 1.0]]), n=5)


class TestValidation:
    def test_invalid_observation_length(self, lgs_identity: LinearGaussianSystem) -> None:
        with pytest.raises(ValueError):
            lgs_identity.posterior(np.zeros(3))

    def test_invalid_batch_shape(self, lgs_identity: LinearGaussianSystem) -> None:
        with pytest.raises(ValueError):
            lgs_identity.posterior_batch(np.zeros((5, 3)))

    def test_invalid_theta_dim(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianSystem(
                Theta=np.array([1.0, 2.0]),
                cov_y=np.eye(2), mx=np.zeros(2), cov_x=np.eye(2),
            )

    def test_invalid_cov_y_shape(self) -> None:
        with pytest.raises(ValueError):
            LinearGaussianSystem(
                Theta=np.eye(2), cov_y=np.eye(3),
                mx=np.zeros(2), cov_x=np.eye(2),
            )


class TestPredictiveMean:
    def test_predictive_mean_single_and_batched(self) -> None:
        lgs = LinearGaussianSystem(
            Theta=np.array([[1.0, 2.0]]),
            cov_y=np.eye(1),
            mx=np.zeros(2), cov_x=np.eye(2),
            b=np.array([0.5]),
        )
        np.testing.assert_allclose(lgs.predictive_mean(np.array([1.0, 1.0])),
                                   [3.5])
        out = lgs.predictive_mean(np.array([[1.0, 1.0], [2.0, 2.0]]))
        np.testing.assert_allclose(out, [[3.5], [6.5]])
