"""Tests for the cross-cutting Posterior protocol."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    BayesianLinearRegression,
    LinearGaussianSystem,
    Pipeline,
    has_credible_interval,
    has_mean_cov,
    isotropic_cov,
    posterior_mean,
    posterior_std,
    summarize_posterior,
)
from active_inference.core.posterior import Posterior


@pytest.fixture
def grid_posterior():
    pipe = Pipeline.linear_gaussian(
        beta0=3.0, beta1=2.0, sigma2_y=0.4, m_x=4.0, s2_x=1.0,
        rng=np.random.default_rng(0),
    )
    return pipe.run(x_star=2.0, n=20)


@pytest.fixture
def lgs_posterior():
    lgs = LinearGaussianSystem(
        Theta=np.eye(2), cov_y=isotropic_cov(2, 0.1),
        mx=np.zeros(2), cov_x=np.eye(2),
    )
    Y = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
    return lgs.posterior_batch(Y)


@pytest.fixture
def blr_posterior():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(scale=0.3, size=50)
    blr = BayesianLinearRegression(
        prior_mean=np.zeros(3), prior_cov=np.eye(3) * 4.0,
        sigma2_y=0.1,
    )
    return blr.fit(X, y)


class TestProtocolMembership:
    def test_all_three_implement_protocol(
        self, grid_posterior, lgs_posterior, blr_posterior
    ) -> None:
        assert isinstance(grid_posterior, Posterior)
        assert isinstance(lgs_posterior, Posterior)
        assert isinstance(blr_posterior, Posterior)


class TestCapabilities:
    def test_grid_has_credible_interval(self, grid_posterior) -> None:
        assert has_credible_interval(grid_posterior) is True
        assert has_mean_cov(grid_posterior) is False

    def test_gaussian_posteriors_carry_mean_cov(
        self, lgs_posterior, blr_posterior
    ) -> None:
        for p in (lgs_posterior, blr_posterior):
            assert has_credible_interval(p) is False
            assert has_mean_cov(p) is True


class TestUniformAccessors:
    def test_grid_mean_is_scalar(self, grid_posterior) -> None:
        assert isinstance(posterior_mean(grid_posterior), float)

    def test_gaussian_mean_is_vector(self, lgs_posterior, blr_posterior) -> None:
        for p in (lgs_posterior, blr_posterior):
            out = posterior_mean(p)
            assert isinstance(out, np.ndarray)
            assert out.ndim == 1

    def test_grid_std_is_scalar(self, grid_posterior) -> None:
        assert isinstance(posterior_std(grid_posterior), float)

    def test_gaussian_std_is_vector(self, lgs_posterior, blr_posterior) -> None:
        for p in (lgs_posterior, blr_posterior):
            s = posterior_std(p)
            assert isinstance(s, np.ndarray)
            assert s.ndim == 1
            assert np.all(s >= 0)

    def test_summarize_returns_string(
        self, grid_posterior, lgs_posterior, blr_posterior
    ) -> None:
        for p in (grid_posterior, lgs_posterior, blr_posterior):
            s = summarize_posterior(p)
            assert isinstance(s, str)
            assert len(s) > 10


class TestErrorPath:
    def test_unknown_object_raises(self) -> None:
        class NotAPosterior:
            pass

        with pytest.raises(AttributeError):
            posterior_mean(NotAPosterior())
        with pytest.raises(AttributeError):
            posterior_std(NotAPosterior())
