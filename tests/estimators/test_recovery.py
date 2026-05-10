"""Parameter-recovery and calibration tests for the estimator suite.

These tests run small simulation studies (≤ 200 trials) and assert
*statistical* properties — that bias shrinks with N, that posterior width
shrinks with N, and that empirical coverage matches the nominal credible
level. They are slower than the unit tests in this folder but still under
a few seconds each.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.diagnostics import calibration_curve
from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.generative_process import (
    LinearGaussianMVProcess,
    LinearGaussianProcess,
)
from active_inference.core.inference import GridBayesianInference
from active_inference.core.lgs import LinearGaussianSystem
from active_inference.estimators.linear_regression import (
    BayesianLinearRegression,
    mle_linear_regression,
)
from active_inference.estimators.map import map_analytic_linear
from active_inference.estimators.mle import mle_analytic_linear
from active_inference.utils.grids import make_grid


SEED = 0


def _make_proc(sigma2_y: float = 0.5, seed: int = SEED) -> LinearGaussianProcess:
    return LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=sigma2_y,
        rng=np.random.default_rng(seed),
    )


# ---------------------------------------------------------------------------
# Univariate hidden-state recovery
# ---------------------------------------------------------------------------


class TestHiddenStateRecovery:
    def test_mle_bias_shrinks_with_n(self) -> None:
        # Sweep N; absolute bias of the MLE around the true x* should drop.
        rng = np.random.default_rng(SEED)
        x_star = 2.5
        biases = []
        for n in (5, 50, 500):
            proc = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=1.0, rng=rng)
            errors = []
            for _ in range(60):
                ys = proc.sample(x_star, n=n).flatten()
                errors.append(mle_analytic_linear(ys, 3.0, 2.0) - x_star)
            biases.append(float(np.std(errors)))
        # std of the estimator should fall as 1/sqrt(N).
        assert biases[0] > biases[1] > biases[2]

    def test_grid_posterior_credible_interval_covers_truth(self) -> None:
        # 95% credible intervals should cover the truth ~95% of the time.
        rng = np.random.default_rng(SEED + 1)
        T = 200
        n_per_trial = 30
        truths = rng.uniform(1.0, 4.0, size=T)
        contains = np.empty(T, dtype=bool)
        x_grid = make_grid(0.0, 5.0, 500)
        for t in range(T):
            proc = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.5,
                                         rng=np.random.default_rng(SEED + 100 + t))
            ys = proc.sample(truths[t], n=n_per_trial).flatten()
            # Wide-prior model so the data dominates.
            model = LinearGaussianModel(
                beta0=3.0, beta1=2.0, sigma2_y=0.5,
                m_x=2.5, s2_x=4.0, prior_kind="gaussian",
            )
            res = GridBayesianInference(model, x_grid).infer(ys)
            lo, hi = res.credible_interval(0.95)
            contains[t] = lo <= truths[t] <= hi
        coverage = float(np.mean(contains))
        # Allow ±5% tolerance due to finite T = 200.
        assert 0.90 <= coverage <= 1.0


# ---------------------------------------------------------------------------
# MAP / prior shrinkage
# ---------------------------------------------------------------------------


class TestMAPShrinkage:
    def test_strong_prior_shrinks_toward_prior_mean(self) -> None:
        rng = np.random.default_rng(SEED + 2)
        x_star = 2.0
        prior_mean = 4.0
        proc = _make_proc(sigma2_y=0.5, seed=SEED + 2)
        ys = proc.sample(x_star, n=20).flatten()
        weak = map_analytic_linear(ys, 3.0, 2.0, 0.5, prior_mean, 10.0)
        strong = map_analytic_linear(ys, 3.0, 2.0, 0.5, prior_mean, 0.05)
        # The strong-prior MAP must be closer to the prior mean than the weak one.
        assert abs(strong - prior_mean) < abs(weak - prior_mean)


# ---------------------------------------------------------------------------
# Bayesian linear regression — recovery + calibration
# ---------------------------------------------------------------------------


class TestBLRRecovery:
    def test_posterior_concentrates_with_n(self) -> None:
        rng = np.random.default_rng(SEED + 3)
        true = np.array([1.0, -0.5, 0.7])
        widths = []
        for n in (30, 300, 1500):
            X = rng.normal(size=(n, 2))
            y = true[0] + X @ true[1:] + rng.normal(scale=0.3, size=n)
            blr = BayesianLinearRegression(
                prior_mean=np.zeros(3), prior_cov=np.eye(3) * 4.0,
                sigma2_y=0.09,
            )
            post = blr.fit(X, y)
            widths.append(float(np.linalg.norm(post.std())))
        # Posterior width must monotonically shrink with N.
        assert widths[0] > widths[1] > widths[2]
        # Final posterior mean is close to the truth.
        assert np.linalg.norm(post.mean - true) < 0.1

    def test_posterior_predictive_calibration(self) -> None:
        # Generate data with known noise, ask BLR for predictive intervals,
        # and check that empirical coverage matches the nominal level.
        from scipy.special import erfinv
        rng = np.random.default_rng(SEED + 4)
        true = np.array([0.5, 1.5, -0.2])
        sigma2_y = 0.25
        N_train, N_test = 500, 1000
        X_train = rng.normal(size=(N_train, 2))
        y_train = true[0] + X_train @ true[1:] + rng.normal(
            scale=np.sqrt(sigma2_y), size=N_train,
        )
        X_test = rng.normal(size=(N_test, 2))
        y_test = true[0] + X_test @ true[1:] + rng.normal(
            scale=np.sqrt(sigma2_y), size=N_test,
        )
        blr = BayesianLinearRegression(
            prior_mean=np.zeros(3), prior_cov=np.eye(3) * 4.0,
            sigma2_y=sigma2_y,
        )
        post = blr.fit(X_train, y_train)
        mean_pred, var_pred = post.predictive(X_test, sigma2_y=sigma2_y)
        std_pred = np.sqrt(var_pred)

        # Coverage check at the 90 % level.
        z90 = float(np.sqrt(2.0) * erfinv(0.9))
        lo = mean_pred - z90 * std_pred
        hi = mean_pred + z90 * std_pred
        coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
        assert 0.85 <= coverage <= 0.95


# ---------------------------------------------------------------------------
# LGS — multivariate sensor fusion calibration
# ---------------------------------------------------------------------------


class TestLGSCalibration:
    def test_lgs_credible_ellipse_covers_at_nominal_rate(self) -> None:
        # Repeat a 2-D sensor-fusion experiment and check the 95% Mahalanobis
        # credible region covers the truth ~95% of the time.
        rng = np.random.default_rng(SEED + 5)
        T = 400
        n_per_trial = 30
        Theta = np.eye(2)
        cov_y = 0.04 * np.eye(2)
        # 95% MVN credible region uses the chi-square(2) quantile.
        # χ²₂ at 0.95 = -2 ln(0.05) ≈ 5.991
        chi2_95 = float(-2.0 * np.log(0.05))
        contains = np.empty(T, dtype=bool)
        for t in range(T):
            x_star = rng.uniform(0.0, 1.0, size=2)
            proc = LinearGaussianMVProcess(
                Theta=Theta, cov_y=cov_y, rng=np.random.default_rng(SEED + 1000 + t),
            )
            Y = proc.sample(x_star, n=n_per_trial).reshape(n_per_trial, 2)
            lgs = LinearGaussianSystem(
                Theta=Theta, cov_y=cov_y,
                mx=np.array([0.5, 0.5]), cov_x=4.0 * np.eye(2),
            )
            post = lgs.posterior_batch(Y)
            diff = x_star - post.mean
            mahal = float(diff @ np.linalg.inv(post.cov) @ diff)
            contains[t] = mahal <= chi2_95
        coverage = float(np.mean(contains))
        # Allow a comfortable tolerance — T = 400 trials.
        assert 0.90 <= coverage <= 0.99


# ---------------------------------------------------------------------------
# OLS / lstsq sanity
# ---------------------------------------------------------------------------


class TestOLSRecovery:
    def test_ols_unbiased_at_high_n(self) -> None:
        rng = np.random.default_rng(SEED + 6)
        true = np.array([0.7, -1.3, 0.4, 0.9])
        N = 5000
        X = rng.normal(size=(N, 3))
        y = true[0] + X @ true[1:] + rng.normal(scale=0.5, size=N)
        theta = mle_linear_regression(X, y)
        # ‖θ̂ − θ*‖₂ should be small for N = 5000, σ = 0.5.
        assert np.linalg.norm(theta - true) < 0.1

    def test_ols_residual_distribution(self) -> None:
        # Standardized residuals should look standard-normal.
        rng = np.random.default_rng(SEED + 7)
        N = 1000
        X = rng.normal(size=(N, 2))
        y = 1.0 + X @ np.array([0.5, -0.5]) + rng.normal(scale=0.3, size=N)
        theta = mle_linear_regression(X, y)
        from active_inference.estimators.linear_regression import add_intercept
        residuals = y - add_intercept(X) @ theta
        # Mean ~ 0 and sample std close to the true noise std.
        assert abs(residuals.mean()) < 0.05
        assert abs(residuals.std() - 0.3) < 0.05
