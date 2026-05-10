"""Maximum-likelihood estimation utilities for the linear-Gaussian model.

These helpers are intentionally tiny — they exist so that chapter orchestrators
can call a single named function rather than re-deriving the algebra inline.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.distributions import gaussian_log_pdf

ArrayLike = np.ndarray


def mle_analytic_linear(
    y_obs: ArrayLike,
    beta0: float,
    beta1: float,
) -> float:
    """Closed-form MLE of the hidden state under ``y = beta0 + beta1 * x + noise``.

    Setting the derivative of the log-likelihood w.r.t. ``x`` to zero yields
    ``x_MLE = (mean(y) - beta0) / beta1``.

    Parameters
    ----------
    y_obs : array-like
        Observations sampled from the generative process.
    beta0, beta1 : float
        Coefficients of the agent's linear generating function.

    Returns
    -------
    float
        The MLE of the hidden state.
    """
    if beta1 == 0:
        raise ValueError("beta1 must be non-zero for the inverse to exist")
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    return float((y_obs.mean() - beta0) / beta1)


def mle_loss(
    x: ArrayLike,
    y_obs: ArrayLike,
    beta0: float,
    beta1: float,
    sigma2_y: float,
    psi: Optional[Callable[[ArrayLike], ArrayLike]] = None,
) -> np.ndarray:
    """Negative log-likelihood of ``x`` (sum over i.i.d. observations).

    Returns a scalar when ``x`` is scalar and an array when ``x`` is an array.
    """
    x = np.asarray(x, dtype=float)
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    psi_x = psi(x) if psi is not None else x
    mu = beta0 + beta1 * psi_x  # broadcastable
    # gaussian_log_pdf broadcasts y[:, None] vs mu shape
    log_terms = gaussian_log_pdf(y_obs.reshape(-1, *([1] * np.ndim(x))), mu, sigma2_y)
    return -log_terms.sum(axis=0)


def mle_grad_x(
    x: float,
    y_obs: ArrayLike,
    beta0: float,
    beta1: float,
    sigma2_y: float,
) -> float:
    """Analytic gradient of the negative log-likelihood w.r.t. ``x`` (linear case).

    ``d(-ell)/dx = -beta1 / sigma2_y * sum_i (y_i - beta0 - beta1 * x)``
    """
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    residuals = y_obs - (beta0 + beta1 * x)
    return float(-beta1 / sigma2_y * residuals.sum())
