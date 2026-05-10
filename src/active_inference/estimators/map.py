"""Maximum a posteriori (MAP) estimators for the linear-Gaussian model.

The MAP adds a Gaussian prior penalty to the MLE objective. With both prior and
likelihood Gaussian, the closed-form solution is a precision-weighted average
of the data-driven estimate and the prior mean.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.distributions import gaussian_log_pdf

ArrayLike = np.ndarray


def map_analytic_linear(
    y_obs: ArrayLike,
    beta0: float,
    beta1: float,
    sigma2_y: float,
    m_x: float,
    s2_x: float,
) -> float:
    """Closed-form MAP estimate for ``x`` under linear-Gaussian likelihood and prior.

    Combining the Gaussian likelihood with a Gaussian prior gives a posterior
    whose mean (= mode) is::

        x_MAP = (beta1 / sigma2_y * sum(y_i - beta0) + m_x / s2_x)
              / (N * beta1**2 / sigma2_y + 1 / s2_x)

    where ``N`` is the number of observations.
    """
    if beta1 == 0:
        raise ValueError("beta1 must be non-zero")
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    n = y_obs.size
    num = beta1 / sigma2_y * (y_obs - beta0).sum() + m_x / s2_x
    den = n * (beta1 ** 2) / sigma2_y + 1.0 / s2_x
    return float(num / den)


def map_loss(
    x: ArrayLike,
    y_obs: ArrayLike,
    beta0: float,
    beta1: float,
    sigma2_y: float,
    m_x: float,
    s2_x: float,
    psi: Optional[Callable[[ArrayLike], ArrayLike]] = None,
) -> np.ndarray:
    """Negative log-posterior up to an additive constant (the MAP loss)."""
    x = np.asarray(x, dtype=float)
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    psi_x = psi(x) if psi is not None else x
    mu = beta0 + beta1 * psi_x
    log_lik = gaussian_log_pdf(
        y_obs.reshape(-1, *([1] * np.ndim(x))), mu, sigma2_y
    ).sum(axis=0)
    log_prior = gaussian_log_pdf(x, m_x, s2_x)
    return -(log_lik + log_prior)


def map_grad_x(
    x: float,
    y_obs: ArrayLike,
    beta0: float,
    beta1: float,
    sigma2_y: float,
    m_x: float,
    s2_x: float,
) -> float:
    """Gradient of :func:`map_loss` w.r.t. ``x`` for the linear-Gaussian case."""
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    residuals = y_obs - (beta0 + beta1 * x)
    grad_lik = -beta1 / sigma2_y * residuals.sum()
    grad_prior = (x - m_x) / s2_x
    return float(grad_lik + grad_prior)
