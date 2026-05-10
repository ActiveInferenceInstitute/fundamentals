"""Unit tests for the factor-analysis EM helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import diagonal_cov, fit_factor_analysis, mvn_sample
from active_inference.estimators.em import (
    factor_analysis_e_step,
    factor_analysis_m_step,
    incomplete_log_likelihood,
)


def synthetic(n: int = 400, n_factors: int = 2,
              n_obs: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    Theta = rng.normal(scale=0.7, size=(n_obs, n_factors))
    diag = rng.uniform(0.05, 0.3, size=n_obs)
    X = mvn_sample(np.zeros(n_factors), np.eye(n_factors), n=n, rng=rng)
    noise = mvn_sample(np.zeros(n_obs), diagonal_cov(diag), n=n, rng=rng)
    Y = X @ Theta.T + noise
    return Y, Theta, diag


class TestEMConvergence:
    def test_log_likelihood_monotone(self) -> None:
        Y, *_ = synthetic(n=400, seed=0)
        result = fit_factor_analysis(
            Y, n_factors=2, max_iter=80, tol=0,
            rng=np.random.default_rng(0),
        )
        diffs = np.diff(result.log_likelihoods)
        assert np.all(diffs >= -1e-6), f"non-monotone: min Δ = {diffs.min():.2e}"

    def test_em_recovers_subspace(self) -> None:
        Y, true_Theta, _ = synthetic(n=2000, seed=1)
        result = fit_factor_analysis(
            Y, n_factors=2, max_iter=400, tol=1e-6,
            rng=np.random.default_rng(0),
        )
        # FA is identifiable up to rotation: compare column spaces, not raw cols.
        Q_true, _ = np.linalg.qr(true_Theta)
        Q_est, _ = np.linalg.qr(result.Theta)
        # Subspace alignment: the max singular value of Q_true^T Q_est is 1
        # iff the spaces coincide.
        sv = np.linalg.svd(Q_true.T @ Q_est, compute_uv=False)
        assert np.allclose(sv, 1.0, atol=0.05), f"sv = {sv}"


class TestSingleSteps:
    def test_e_step_shape(self) -> None:
        Y, *_ = synthetic(n=10, seed=2)
        Theta = np.zeros((Y.shape[1], 2))
        Theta[:, 0] = 1.0
        cov_y = np.eye(Y.shape[1])
        mu, cov = factor_analysis_e_step(Y, Theta, cov_y)
        assert mu.shape == (10, 2)
        assert cov.shape == (2, 2)

    def test_m_step_returns_diagonal_cov(self) -> None:
        Y, *_ = synthetic(n=20, seed=3)
        Theta = np.zeros((Y.shape[1], 2))
        Theta[:, 0] = 1.0
        cov_y = np.eye(Y.shape[1])
        mu, cov = factor_analysis_e_step(Y, Theta, cov_y)
        Theta_new, cov_y_new = factor_analysis_m_step(Y, mu, cov)
        # cov_y must remain diagonal in factor analysis.
        off = cov_y_new - np.diag(np.diag(cov_y_new))
        assert np.max(np.abs(off)) < 1e-10

    def test_incomplete_ll_finite(self) -> None:
        Y, true_Theta, true_diag = synthetic(n=100, seed=4)
        ll = incomplete_log_likelihood(
            Y, true_Theta, np.diag(true_diag),
        )
        assert np.isfinite(ll)


class TestResultHelpers:
    def test_predict_latent_shape(self) -> None:
        Y, *_ = synthetic(n=200, seed=5)
        result = fit_factor_analysis(
            Y, n_factors=2, max_iter=50, tol=0,
            rng=np.random.default_rng(0),
        )
        latent = result.predict_latent(Y[:10])
        assert latent.shape == (10, 2)

    def test_predict_latent_invalid_shape(self) -> None:
        Y, *_ = synthetic(n=50, seed=6)
        result = fit_factor_analysis(
            Y, n_factors=2, max_iter=20, tol=0,
            rng=np.random.default_rng(0),
        )
        with pytest.raises(ValueError):
            result.predict_latent(np.zeros((5, 99)))

    def test_reconstruct_close_to_input(self) -> None:
        Y, *_ = synthetic(n=200, seed=7)
        result = fit_factor_analysis(
            Y, n_factors=2, max_iter=200, tol=1e-6,
            rng=np.random.default_rng(0),
        )
        Y_hat = result.reconstruct(Y)
        rmse = float(np.sqrt(((Y - Y_hat) ** 2).mean()))
        assert rmse < float(np.std(Y))

    def test_summary_contains_iters(self) -> None:
        Y, *_ = synthetic(n=80, seed=8)
        result = fit_factor_analysis(
            Y, n_factors=2, max_iter=10, tol=0,
            rng=np.random.default_rng(0),
        )
        s = result.summary()
        assert "iters=" in s
        assert "converged=" in s
