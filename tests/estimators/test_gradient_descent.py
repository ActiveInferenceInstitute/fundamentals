"""Tests for ``estimators.gradient_descent`` — generic 1-D minimizer."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.generative_process import LinearGaussianProcess
from active_inference.estimators.gradient_descent import (
    GradientDescentResult,
    gradient_descent,
)
from active_inference.estimators.map import map_grad_x, map_loss
from active_inference.estimators.mle import (
    mle_analytic_linear,
    mle_grad_x,
    mle_loss,
)


def _samples(n: int = 100, x_true: float = 2.5,
             sigma2_y: float = 0.5, seed: int = 0) -> np.ndarray:
    proc = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=sigma2_y,
        rng=np.random.default_rng(seed),
    )
    return proc.sample(x_true, n=n).flatten()


class TestConvergence:
    def test_quadratic_minimum(self) -> None:
        result = gradient_descent(
            loss_fn=lambda x: (x - 3.0) ** 2,
            x0=0.0,
            learning_rate=0.1,
            max_iter=200,
        )
        assert result.x == pytest.approx(3.0, abs=1e-3)
        assert result.converged

    def test_mle_descent_matches_analytic(self) -> None:
        ys = _samples()
        analytic = mle_analytic_linear(ys, 3.0, 2.0)
        result = gradient_descent(
            loss_fn=lambda x: float(mle_loss(x, ys, 3.0, 2.0, 0.5)),
            grad_fn=lambda x: mle_grad_x(x, ys, 3.0, 2.0, 0.5),
            x0=4.5, learning_rate=1e-3, max_iter=5000,
        )
        assert result.x == pytest.approx(analytic, abs=5e-3)
        assert result.losses[-1] <= result.losses[0]

    def test_map_descent_matches_analytic(self) -> None:
        ys = _samples()
        from active_inference.estimators.map import map_analytic_linear
        m_an = map_analytic_linear(ys, 3.0, 2.0, 0.5, 4.0, 0.25)
        result = gradient_descent(
            loss_fn=lambda x: float(map_loss(x, ys, 3.0, 2.0, 0.5, 4.0, 0.25)),
            grad_fn=lambda x: map_grad_x(x, ys, 3.0, 2.0, 0.5, 4.0, 0.25),
            x0=0.0, learning_rate=1e-3, max_iter=5000,
        )
        assert result.x == pytest.approx(m_an, abs=5e-3)


class TestNumericalGradient:
    def test_finite_difference_fallback(self) -> None:
        # Without grad_fn the loop falls back to centered finite difference.
        result = gradient_descent(
            loss_fn=lambda x: (x - 1.5) ** 2,
            x0=0.0, learning_rate=0.1, max_iter=200,
        )
        assert result.x == pytest.approx(1.5, abs=1e-3)


class TestValidation:
    def test_invalid_lr(self) -> None:
        with pytest.raises(ValueError):
            gradient_descent(
                loss_fn=lambda x: x ** 2, x0=1.0, learning_rate=-1.0,
            )

    def test_zero_max_iter(self) -> None:
        with pytest.raises(ValueError):
            gradient_descent(
                loss_fn=lambda x: x ** 2, x0=1.0,
                learning_rate=0.1, max_iter=0,
            )


class TestResultDataclass:
    def test_history_recorded(self) -> None:
        result = gradient_descent(
            loss_fn=lambda x: (x - 1.0) ** 2,
            x0=0.0, learning_rate=0.1, max_iter=20,
            record_history=True,
        )
        assert isinstance(result, GradientDescentResult)
        assert result.history.size > 1
        assert result.losses.size == result.history.size

    def test_history_disabled(self) -> None:
        result = gradient_descent(
            loss_fn=lambda x: (x - 1.0) ** 2,
            x0=0.0, learning_rate=0.1, max_iter=20,
            record_history=False,
        )
        # Only the initial value is kept on the history when disabled.
        assert result.history.size == 1
