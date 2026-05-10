"""Tests for ``estimators.map`` — univariate hidden-state MAP."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.generative_process import LinearGaussianProcess
from active_inference.core.inference import GridBayesianInference
from active_inference.estimators.map import (
    map_analytic_linear,
    map_grad_x,
    map_loss,
)
from active_inference.estimators.mle import mle_analytic_linear


def _samples(n: int = 30, x_true: float = 2.0,
             sigma2_y: float = 0.5, seed: int = 0) -> np.ndarray:
    proc = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=sigma2_y,
        rng=np.random.default_rng(seed),
    )
    return proc.sample(x_true, n=n).flatten()


class TestAnalytic:
    def test_lies_between_mle_and_prior(self) -> None:
        ys = _samples(n=10, x_true=2.0)
        mle = mle_analytic_linear(ys, 3.0, 2.0)
        prior_mean = 4.0
        m = map_analytic_linear(
            ys, 3.0, 2.0, sigma2_y=0.5, m_x=prior_mean, s2_x=0.25,
        )
        assert min(mle, prior_mean) <= m <= max(mle, prior_mean)

    def test_matches_grid_mode(self) -> None:
        ys = _samples(n=30, x_true=2.0)
        m_an = map_analytic_linear(
            ys, 3.0, 2.0, sigma2_y=0.5, m_x=4.0, s2_x=0.25,
        )
        model = LinearGaussianModel(
            beta0=3.0, beta1=2.0, sigma2_y=0.5,
            m_x=4.0, s2_x=0.25,
        )
        res = GridBayesianInference(
            model, np.linspace(0.0, 5.0, 2001),
        ).infer(ys)
        assert m_an == pytest.approx(res.posterior_mode, abs=5e-3)


class TestLossLandscape:
    def test_loss_minimum_at_analytic(self) -> None:
        ys = _samples(n=30)
        m_an = map_analytic_linear(ys, 3.0, 2.0, 0.5, 4.0, 0.25)
        x_grid = np.linspace(0.0, 5.0, 1001)
        loss = map_loss(x_grid, ys, 3.0, 2.0, 0.5, 4.0, 0.25)
        assert x_grid[int(np.argmin(loss))] == pytest.approx(m_an, abs=5e-3)


class TestGradient:
    def test_gradient_zero_at_analytic_solution(self) -> None:
        ys = _samples(n=30)
        m_an = map_analytic_linear(ys, 3.0, 2.0, 0.5, 4.0, 0.25)
        g = map_grad_x(m_an, ys, 3.0, 2.0, 0.5, 4.0, 0.25)
        assert g == pytest.approx(0.0, abs=1e-9)

    def test_prior_pulls_estimate_toward_prior_mean(self) -> None:
        # Strong prior (small s2_x) should pull MAP further from MLE.
        ys = _samples(n=10, x_true=2.0)
        mle = mle_analytic_linear(ys, 3.0, 2.0)
        weak_prior = map_analytic_linear(ys, 3.0, 2.0, 0.5, 4.0, s2_x=10.0)
        strong_prior = map_analytic_linear(ys, 3.0, 2.0, 0.5, 4.0, s2_x=0.05)
        assert abs(strong_prior - 4.0) < abs(weak_prior - 4.0)
        assert abs(weak_prior - mle) < abs(strong_prior - mle)
