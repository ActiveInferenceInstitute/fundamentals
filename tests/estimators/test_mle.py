"""Tests for ``estimators.mle`` — univariate hidden-state MLE."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.generative_process import LinearGaussianProcess
from active_inference.estimators.mle import (
    mle_analytic_linear,
    mle_grad_x,
    mle_loss,
)


def _samples(n: int = 50, x_true: float = 2.0,
             sigma2_y: float = 0.5, seed: int = 0) -> np.ndarray:
    proc = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=sigma2_y,
        rng=np.random.default_rng(seed),
    )
    return proc.sample(x_true, n=n).flatten()


class TestAnalytic:
    def test_matches_inverse_of_mean(self) -> None:
        ys = np.array([5.0, 7.0, 9.0])  # mean = 7 → inverse = 2
        x = mle_analytic_linear(ys, beta0=3.0, beta1=2.0)
        assert x == pytest.approx(2.0)

    def test_zero_beta1_raises(self) -> None:
        with pytest.raises(ValueError):
            mle_analytic_linear(np.array([1.0]), beta0=0.0, beta1=0.0)


class TestLossLandscape:
    def test_minimum_at_analytic_solution(self) -> None:
        ys = _samples(n=100)
        x_star = mle_analytic_linear(ys, 3.0, 2.0)
        x_grid = np.linspace(0.0, 5.0, 1001)
        loss = mle_loss(x_grid, ys, 3.0, 2.0, sigma2_y=0.5)
        assert x_grid[int(np.argmin(loss))] == pytest.approx(x_star, abs=5e-3)

    def test_loss_with_psi_nonlinear(self) -> None:
        ys = _samples(n=20)
        x_grid = np.linspace(0.5, 4.0, 51)
        loss = mle_loss(
            x_grid, ys, beta0=0.0, beta1=1.0, sigma2_y=1.0,
            psi=lambda x: x ** 2,
        )
        assert np.all(np.isfinite(loss))


class TestGradient:
    def test_gradient_zero_at_analytic_solution(self) -> None:
        ys = _samples(n=100)
        x_star = mle_analytic_linear(ys, 3.0, 2.0)
        g = mle_grad_x(x_star, ys, 3.0, 2.0, sigma2_y=0.5)
        assert g == pytest.approx(0.0, abs=1e-9)

    def test_gradient_sign_matches_residual(self) -> None:
        # If x is too small, residuals (y - β0 - β1 x) are positive on average,
        # so the gradient of -log L w.r.t. x should be negative.
        ys = _samples(n=200, x_true=2.5)
        g_low = mle_grad_x(0.0, ys, 3.0, 2.0, sigma2_y=0.5)
        g_high = mle_grad_x(5.0, ys, 3.0, 2.0, sigma2_y=0.5)
        assert g_low < 0 < g_high
