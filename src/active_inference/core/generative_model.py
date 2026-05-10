"""Generative models — the *agent's internal* representation of the environment.

A generative model factorizes as ``p(x, y) = p(y | x) p(x)``. The univariate
classes in this module expose:

1. ``log_likelihood(y, x_grid)`` — log p(y | x) evaluated on a grid of states.
2. ``log_prior(x_grid)`` — log p(x) on the same grid.
3. ``predict_mean(x_grid)`` — the mean of the likelihood, useful for plots.

The multivariate ``LinearGaussianMVModel`` adds vector-valued analogues used
in Chapter 3 (Bayesian linear regression, linear Gaussian systems, factor
analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .distributions import (
    dirac_like_pdf,
    gaussian_log_pdf,
    gaussian_pdf,
    mvn_log_pdf,
    uniform_pdf,
)

ArrayLike = np.ndarray


@dataclass
class GenerativeModel:
    """Generic agent-side generative model interface.

    Subclasses must override :meth:`log_likelihood` and :meth:`log_prior`.
    A default :meth:`predict_mean` is provided for the common case where the
    likelihood mean is itself a function the user supplies.
    """

    name: str = "GenerativeModel"

    def log_likelihood(self, y: float, x_grid: ArrayLike) -> np.ndarray:
        """``log p(y | x)`` evaluated on ``x_grid``. Override in subclasses."""
        raise NotImplementedError

    def log_prior(self, x_grid: ArrayLike) -> np.ndarray:
        """``log p(x)`` evaluated on ``x_grid``. Override in subclasses."""
        raise NotImplementedError

    def likelihood(self, y: float, x_grid: ArrayLike) -> np.ndarray:
        """Linear-space likelihood: ``exp(log_likelihood(y, x_grid))``."""
        return np.exp(self.log_likelihood(y, x_grid))

    def prior(self, x_grid: ArrayLike) -> np.ndarray:
        """Linear-space prior density: ``exp(log_prior(x_grid))``."""
        return np.exp(self.log_prior(x_grid))

    def predict_mean(self, x_grid: ArrayLike) -> np.ndarray:
        """Mean of the likelihood at each ``x``. Override in subclasses."""
        raise NotImplementedError


@dataclass
class LinearGaussianModel(GenerativeModel):
    """Generative model with Gaussian likelihood and Gaussian (or uniform) prior.

    Likelihood :  ``p(y | x) = N(y ; beta0 + beta1 * psi(x), sigma2_y)``
    Prior      :  ``p(x)     = N(x ; m_x, s2_x)`` (or uniform on ``[a, b]``)

    Parameters
    ----------
    beta0, beta1 : float
        Linear coefficients of the agent's generating function.
    sigma2_y : float
        Variance of the likelihood. Use a small positive number to approximate
        a deterministic likelihood (Dirac-like).
    m_x, s2_x : float
        Mean and variance of the Gaussian state prior.
    prior_kind : {"gaussian", "uniform"}
        Whether the prior is Gaussian or improper-uniform on ``[uniform_low,
        uniform_high]``.
    psi : callable, optional
        Optional nonlinear transform of ``x`` applied before the linear
        readout. Defaults to identity.
    """

    beta0: float = 0.0
    beta1: float = 1.0
    sigma2_y: float = 1.0
    m_x: float = 0.0
    s2_x: float = 1.0
    prior_kind: str = "gaussian"
    uniform_low: float = 0.0
    uniform_high: float = 1.0
    psi: Optional[Callable[[ArrayLike], ArrayLike]] = None
    name: str = "LinearGaussianModel"

    def __post_init__(self) -> None:
        if self.sigma2_y <= 0:
            raise ValueError("sigma2_y must be strictly positive")
        if self.prior_kind == "gaussian" and self.s2_x <= 0:
            raise ValueError("s2_x must be strictly positive for Gaussian prior")
        if self.prior_kind == "uniform" and self.uniform_high <= self.uniform_low:
            raise ValueError("uniform_high must be greater than uniform_low")

    def predict_mean(self, x_grid: ArrayLike) -> np.ndarray:
        """Linear (or `psi`-transformed) mean ``β₀ + β₁ ψ(x)`` on the grid."""
        x_grid = np.asarray(x_grid, dtype=float)
        psi = self.psi(x_grid) if self.psi is not None else x_grid
        return self.beta0 + self.beta1 * psi

    def log_likelihood(self, y: float, x_grid: ArrayLike) -> np.ndarray:
        """``log N(y ; predict_mean(x), σ²_y)`` evaluated across ``x_grid``."""
        mu = self.predict_mean(x_grid)
        return gaussian_log_pdf(y, mu, self.sigma2_y)

    def log_likelihood_batch(self, y_obs: ArrayLike, x_grid: ArrayLike) -> np.ndarray:
        """Sum of log-likelihoods over a batch of i.i.d. observations.

        Parameters
        ----------
        y_obs : array-like, shape ``(N,)``
            Independent observations from the generative process.
        x_grid : array-like, shape ``(G,)``
            Candidate hidden-state values.

        Returns
        -------
        np.ndarray, shape ``(G,)``
            ``sum_i log p(y_i | x)`` for each ``x`` on the grid.
        """
        y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
        x_grid = np.asarray(x_grid, dtype=float)
        mu = self.predict_mean(x_grid)               # shape (G,)
        # Broadcast (N, 1) vs (1, G) -> (N, G)
        log_terms = gaussian_log_pdf(y_obs[:, None], mu[None, :], self.sigma2_y)
        return log_terms.sum(axis=0)

    def log_prior(self, x_grid: ArrayLike) -> np.ndarray:
        """``log p(x)`` for the configured prior kind (Gaussian or uniform)."""
        x_grid = np.asarray(x_grid, dtype=float)
        if self.prior_kind == "gaussian":
            return gaussian_log_pdf(x_grid, self.m_x, self.s2_x)
        elif self.prior_kind == "uniform":
            density = uniform_pdf(x_grid, self.uniform_low, self.uniform_high)
            with np.errstate(divide="ignore"):
                return np.log(density)
        raise ValueError(f"Unknown prior_kind: {self.prior_kind!r}")

    def likelihood_deterministic(self, y: float, x_grid: ArrayLike) -> np.ndarray:
        """Dirac-like deterministic likelihood used in §2.1 Example 2.1."""
        mu = self.predict_mean(x_grid)
        return dirac_like_pdf(y, location=mu, epsilon=np.sqrt(self.sigma2_y))

    def __repr__(self) -> str:  # pragma: no cover
        psi = "linear" if self.psi is None else "nonlinear"
        return (
            f"LinearGaussianModel({psi}, beta0={self.beta0:g}, beta1={self.beta1:g}, "
            f"sigma2_y={self.sigma2_y:g}, prior_kind={self.prior_kind}, "
            f"m_x={self.m_x:g}, s2_x={self.s2_x:g})"
        )


# ---------------------------------------------------------------------------
# Multivariate model
# ---------------------------------------------------------------------------


@dataclass
class LinearGaussianMVModel:
    """Multivariate linear-Gaussian agent.

    Likelihood :  ``p(y | x) = N(y ; Theta x + b, cov_y)``
    Prior      :  ``p(x)     = N(x ; mx, cov_x)``  (Gaussian, multivariate)

    Attributes are stored as NumPy arrays with shape checks done at
    construction time so that downstream estimators can rely on them.
    """

    Theta: np.ndarray
    cov_y: np.ndarray
    mx: np.ndarray
    cov_x: np.ndarray
    b: Optional[np.ndarray] = None
    name: str = "LinearGaussianMVModel"

    def __post_init__(self) -> None:
        self.Theta = np.asarray(self.Theta, dtype=float)
        if self.Theta.ndim != 2:
            raise ValueError("Theta must be 2-D")
        d, c = self.Theta.shape
        self.cov_y = np.asarray(self.cov_y, dtype=float)
        if self.cov_y.shape != (d, d):
            raise ValueError(f"cov_y must be ({d},{d}), got {self.cov_y.shape}")
        self.mx = np.asarray(self.mx, dtype=float).reshape(-1)
        if self.mx.size != c:
            raise ValueError(f"mx must have length {c}, got {self.mx.size}")
        self.cov_x = np.asarray(self.cov_x, dtype=float)
        if self.cov_x.shape != (c, c):
            raise ValueError(f"cov_x must be ({c},{c}), got {self.cov_x.shape}")
        if self.b is None:
            self.b = np.zeros(d)
        else:
            self.b = np.asarray(self.b, dtype=float).reshape(-1)
            if self.b.size != d:
                raise ValueError(f"b must have length {d}, got {self.b.size}")

    @property
    def D(self) -> int:
        """Observation dimensionality (rows of ``Theta``)."""
        return self.Theta.shape[0]

    @property
    def C(self) -> int:
        """Latent (hidden-state) dimensionality (columns of ``Theta``)."""
        return self.Theta.shape[1]

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        """Multivariate mean ``Θ x + b`` (single vector or row-batched ``x``)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return self.Theta @ x + self.b
        return x @ self.Theta.T + self.b

    def log_likelihood(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``log p(y | x)`` — accepts vector or batched ``x``."""
        mu = self.predict_mean(x)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and mu.ndim == 1:
            return mvn_log_pdf(y, mu, self.cov_y)
        # Broadcast — treat each row of x as a candidate state.
        out = np.empty(mu.shape[0]) if mu.ndim == 2 else np.empty(())
        if mu.ndim == 2:
            for i in range(mu.shape[0]):
                out[i] = mvn_log_pdf(y, mu[i], self.cov_y)
            return out
        return mvn_log_pdf(y, mu, self.cov_y)

    def log_prior(self, x: np.ndarray) -> np.ndarray:
        """Multivariate ``log N(x ; mx, cov_x)`` for the Gaussian state prior."""
        return mvn_log_pdf(x, self.mx, self.cov_x)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LinearGaussianMVModel(D={self.D}, C={self.C}, "
            f"|cov_y|={np.linalg.det(self.cov_y):.3g})"
        )
