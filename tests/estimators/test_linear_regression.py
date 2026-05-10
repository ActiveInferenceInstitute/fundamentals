"""Unit tests for the linear regression estimators."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.estimators.linear_regression import (
    BayesianLinearRegression,
    add_intercept,
    gd_linear_regression,
    mle_linear_regression,
    squared_loss,
    squared_loss_grad,
)


def make_data(n: int = 200, c: int = 3, sigma: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=c + 1)
    X = rng.normal(size=(n, c))
    y = true[0] + X @ true[1:] + rng.normal(scale=sigma, size=n)
    return X, y, true


class TestAnalyticMLE:
    def test_recovers_true_coefficients_at_high_n(self) -> None:
        X, y, true = make_data(n=2000, c=4, sigma=0.05, seed=0)
        theta = mle_linear_regression(X, y)
        np.testing.assert_allclose(theta, true, atol=0.05)

    def test_intercept_off_means_no_bias_column(self) -> None:
        X, y, _ = make_data(n=200, c=2, seed=1)
        theta_with = mle_linear_regression(X, y, intercept=True)
        theta_wo = mle_linear_regression(add_intercept(X), y, intercept=False)
        np.testing.assert_allclose(theta_with, theta_wo, atol=1e-12)


class TestGradientDescent:
    def test_matches_analytic(self) -> None:
        X, y, _ = make_data(n=500, c=3, sigma=0.1, seed=2)
        analytic = mle_linear_regression(X, y)
        result = gd_linear_regression(
            X, y, learning_rate=1e-3, max_iter=5000,
            theta0=np.zeros(4),
        )
        np.testing.assert_allclose(result.theta, analytic, atol=5e-3)

    def test_l2_shrinks_estimate(self) -> None:
        X, y, _ = make_data(n=200, c=3, sigma=0.5, seed=3)
        zero_l2 = gd_linear_regression(
            X, y, learning_rate=1e-3, max_iter=3000, l2=0.0, theta0=np.zeros(4),
        )
        large_l2 = gd_linear_regression(
            X, y, learning_rate=1e-3, max_iter=3000, l2=200.0, theta0=np.zeros(4),
        )
        # Larger λ should yield a strictly shorter coefficient vector.
        assert np.linalg.norm(large_l2.theta) < np.linalg.norm(zero_l2.theta)

    def test_invalid_lr_raises(self) -> None:
        with pytest.raises(ValueError):
            gd_linear_regression(np.array([[1.0]]), np.array([1.0]),
                                 learning_rate=-1.0)

    def test_loss_grad_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(4)
        X = rng.normal(size=(20, 2))
        y = rng.normal(size=20)
        theta = rng.normal(size=3)
        analytic = squared_loss_grad(theta, X, y)
        numeric = np.empty_like(theta)
        eps = 1e-5
        for i in range(theta.size):
            t_plus = theta.copy(); t_plus[i] += eps
            t_minus = theta.copy(); t_minus[i] -= eps
            numeric[i] = (squared_loss(t_plus, X, y)
                          - squared_loss(t_minus, X, y)) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, atol=1e-4)


class TestBayesianLinearRegression:
    def test_posterior_concentrates_with_n(self) -> None:
        X, y, true = make_data(n=500, c=2, sigma=0.5, seed=5)
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(3),
            prior_cov=np.eye(3) * 4.0,
            sigma2_y=0.25,
        )
        post_small = blr.fit(X[:5], y[:5])
        post_large = blr.fit(X, y)
        # Determinant of covariance shrinks as more data is incorporated.
        assert np.linalg.det(post_large.cov) < np.linalg.det(post_small.cov)
        # Posterior mean recovers the truth.
        np.testing.assert_allclose(post_large.mean, true, atol=0.1)

    def test_predictive_variance_lower_bound(self) -> None:
        X, y, _ = make_data(n=200, c=2, sigma=0.5, seed=6)
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(3),
            prior_cov=np.eye(3),
            sigma2_y=0.25,
        )
        post = blr.fit(X, y)
        _, var = post.predictive(X, sigma2_y=blr.sigma2_y)
        # Predictive variance must always be at least σ²_y (the irreducible noise).
        assert np.all(var >= blr.sigma2_y - 1e-9)

    def test_sequential_yields_increasing_n(self) -> None:
        X, y, _ = make_data(n=10, c=2, seed=7)
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(3),
            prior_cov=np.eye(3),
            sigma2_y=0.25,
        )
        history = list(blr.fit_sequential(X, y))
        assert [i for i, _ in history] == list(range(1, 11))

    def test_invalid_sigma2_y_raises(self) -> None:
        with pytest.raises(ValueError):
            BayesianLinearRegression(
                prior_mean=np.zeros(2),
                prior_cov=np.eye(2),
                sigma2_y=0.0,
            )

    def test_predict_returns_mean_only(self) -> None:
        X, y, _ = make_data(n=50, c=2, seed=8)
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(3), prior_cov=np.eye(3),
            sigma2_y=0.25,
        )
        post = blr.fit(X, y)
        mean_pred = post.predict(X)
        full_mean, _ = post.predictive(X, sigma2_y=blr.sigma2_y)
        np.testing.assert_allclose(mean_pred, full_mean)

    def test_sample_recovers_posterior_mean(self) -> None:
        X, y, _ = make_data(n=200, c=2, seed=9)
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(3), prior_cov=np.eye(3) * 4.0,
            sigma2_y=0.25,
        )
        post = blr.fit(X, y)
        samples = post.sample(n=5000, rng=np.random.default_rng(0))
        np.testing.assert_allclose(samples.mean(axis=0), post.mean, atol=0.05)

    def test_summary_contains_mean_and_std(self) -> None:
        X, y, _ = make_data(n=20, c=1, seed=10)
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(2), prior_cov=np.eye(2),
            sigma2_y=0.5,
        )
        post = blr.fit(X, y)
        s = post.summary()
        assert "BLRPosterior" in s
        assert "mean" in s and "std" in s
