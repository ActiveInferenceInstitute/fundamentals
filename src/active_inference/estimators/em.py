"""Expectation–Maximization for linear factor analysis.

Factor analysis is the linear-Gaussian system constrained so that

* the prior is standard normal: ``p(x) = N(x ; 0, I_C)``,
* the observation noise is diagonal: ``cov_y = diag(σ²_1, ..., σ²_D)``,
* the data is centered (so the offset ``b = 0``).

The EM algorithm alternates a Bayesian posterior update over the latent
states (E-step) with a maximum-likelihood update of the loadings ``Θ`` and
diagonal noise variances (M-step). This module exposes both a single-step
helper for didactic walk-throughs and a full :func:`fit_factor_analysis`
loop with convergence diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.distributions import mvn_log_pdf


@dataclass
class FactorAnalysisResult:
    """Final state of an EM run plus the iteration history."""

    Theta: np.ndarray
    cov_y: np.ndarray
    posterior_means: np.ndarray   # shape (N, C)
    posterior_cov: np.ndarray     # shape (C, C) — shared across samples in FA
    log_likelihoods: np.ndarray   # incomplete-data log-likelihood per iteration
    n_iterations: int
    converged: bool
    history: dict = field(default_factory=dict)

    def predict_latent(self, Y_new: np.ndarray) -> np.ndarray:
        """Posterior mean over the latent state for new observations.

        Uses the *trained* loadings and noise covariance — does **not** run
        further EM iterations. Returns shape ``(N, C)``.
        """
        Y_new = np.asarray(Y_new, dtype=float)
        if Y_new.ndim != 2 or Y_new.shape[1] != self.Theta.shape[0]:
            raise ValueError(
                f"Y_new must be (N, {self.Theta.shape[0]}), got {Y_new.shape}"
            )
        inner = self.Theta @ self.Theta.T + self.cov_y
        beta = self.Theta.T @ np.linalg.inv(inner)
        return Y_new @ beta.T

    def reconstruct(self, Y: np.ndarray) -> np.ndarray:
        """Reconstruct ``Y`` through the latent space: ``Y → x → ŷ = Θx``.

        Returns shape ``(N, D)``.
        """
        return self.predict_latent(Y) @ self.Theta.T

    def summary(self, ndigits: int = 4) -> str:
        """One-line readout of dimensions, iteration count, and final log-likelihood."""
        ll = round(float(self.log_likelihoods[-1]), ndigits) \
            if self.log_likelihoods.size else float("nan")
        return (
            f"FactorAnalysisResult(D={self.Theta.shape[0]}, "
            f"C={self.Theta.shape[1]}, iters={self.n_iterations}, "
            f"converged={self.converged}, final_LL={ll})"
        )


def factor_analysis_e_step(
    Y: np.ndarray,
    Theta: np.ndarray,
    cov_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """E-step: posterior mean for each row of ``Y`` and the (shared) covariance.

    Standard FA derivation (with prior ``N(0, I)``)::

        β = Θᵀ (Θ Θᵀ + cov_y)⁻¹
        μ_x|y = β y
        Σ_x|y = I - β Θ
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be (N, D)")
    D, C = Theta.shape
    inner = Theta @ Theta.T + cov_y                    # (D, D)
    beta = Theta.T @ np.linalg.inv(inner)              # (C, D)
    mu = Y @ beta.T                                    # (N, C)
    cov = np.eye(C) - beta @ Theta                     # (C, C)
    return mu, cov


def factor_analysis_m_step(
    Y: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """M-step: re-estimate ``Θ`` and the diagonal noise variances."""
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    # Expected outer product E[x x^T] summed over the batch.
    Exx = mu.T @ mu + n * cov                          # (C, C)
    Exy = Y.T @ mu                                     # (D, C)
    Theta_new = Exy @ np.linalg.inv(Exx)               # (D, C)

    # Diagonal noise update: residual variance per output dimension.
    residual = Y - mu @ Theta_new.T
    diag = (residual ** 2).sum(axis=0) / n + (Theta_new @ cov * Theta_new).sum(axis=1)
    diag = np.maximum(diag, 1e-12)                     # numerical floor
    cov_y_new = np.diag(diag)
    return Theta_new, cov_y_new


def incomplete_log_likelihood(
    Y: np.ndarray,
    Theta: np.ndarray,
    cov_y: np.ndarray,
) -> float:
    """Marginal log-likelihood ``log p(Y) = Σ_n log N(y_n ; 0, Θ Θᵀ + cov_y)``."""
    cov_marg = Theta @ Theta.T + cov_y
    log_terms = mvn_log_pdf(Y, np.zeros(cov_marg.shape[0]), cov_marg)
    return float(log_terms.sum())


def fit_factor_analysis(
    Y: np.ndarray,
    n_factors: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-5,
    rng: Optional[np.random.Generator] = None,
    Theta_init: Optional[np.ndarray] = None,
    cov_y_init: Optional[np.ndarray] = None,
) -> FactorAnalysisResult:
    """Iterate E and M steps until the marginal log-likelihood converges.

    Parameters
    ----------
    Y : np.ndarray, shape ``(N, D)``
        Observed (zero-centered) data.
    n_factors : int
        Latent dimensionality ``C``.
    max_iter, tol : int, float
        Convergence controls.
    rng : np.random.Generator, optional
        For random initialization.
    Theta_init, cov_y_init : np.ndarray, optional
        Override the default random initialization.

    Returns
    -------
    FactorAnalysisResult
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be (N, D)")
    n, d = Y.shape
    if rng is None:
        rng = np.random.default_rng()

    Theta = (
        np.asarray(Theta_init, dtype=float)
        if Theta_init is not None
        else rng.normal(scale=0.1, size=(d, n_factors))
    )
    cov_y = (
        np.asarray(cov_y_init, dtype=float)
        if cov_y_init is not None
        else np.diag(np.var(Y, axis=0) + 1e-3)
    )

    log_likelihoods = []
    history = {"Theta": [], "cov_y_diag": []}
    converged = False

    for k in range(max_iter):
        mu, cov = factor_analysis_e_step(Y, Theta, cov_y)
        Theta, cov_y = factor_analysis_m_step(Y, mu, cov)
        ll = incomplete_log_likelihood(Y, Theta, cov_y)
        log_likelihoods.append(ll)
        history["Theta"].append(Theta.copy())
        history["cov_y_diag"].append(np.diag(cov_y).copy())
        if k > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            converged = True
            break

    mu, cov = factor_analysis_e_step(Y, Theta, cov_y)
    return FactorAnalysisResult(
        Theta=Theta,
        cov_y=cov_y,
        posterior_means=mu,
        posterior_cov=cov,
        log_likelihoods=np.asarray(log_likelihoods),
        n_iterations=k + 1,
        converged=converged,
        history=history,
    )
