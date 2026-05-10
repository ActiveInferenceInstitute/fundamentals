"""Generative processes — the *environment* from which observations are sampled.

A :class:`GenerativeProcess` is a thin, sample-only object: given a true
external state ``x_star`` it produces noisy observations through a generating
function ``g_E`` and an observation-noise distribution. Concrete subclasses
specialize the generating function (linear, polynomial, multivariate, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .distributions import mvn_sample

ArrayLike = np.ndarray


@dataclass
class GenerativeProcess:
    """Abstract scalar generative process ``y = g_E(x*; theta*) + omega_y``.

    Parameters
    ----------
    g_E : callable
        Function ``g_E(x_star, **theta)`` returning the noise-free outcome.
    theta : dict
        Parameters passed as keyword arguments to ``g_E``.
    sigma2_y : float
        Variance of the additive zero-mean Gaussian observation noise.
    rng : np.random.Generator, optional
        Random number generator. A fresh default-rng is created if ``None``.
    """

    g_E: Callable[..., ArrayLike]
    theta: dict = field(default_factory=dict)
    sigma2_y: float = 1.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.sigma2_y < 0:
            raise ValueError("sigma2_y must be non-negative")
        if self.rng is None:
            self.rng = np.random.default_rng()

    def mean(self, x_star: ArrayLike) -> np.ndarray:
        """Noise-free generating function evaluated at ``x_star``."""
        return np.asarray(self.g_E(x_star, **self.theta), dtype=float)

    def sample(self, x_star: ArrayLike, n: int = 1) -> np.ndarray:
        """Draw ``n`` observations from the generative process at state ``x_star``.

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` if ``x_star`` is scalar, else broadcasts.
        """
        mu = self.mean(x_star)
        if self.sigma2_y == 0:
            return np.broadcast_to(mu, (n,) + np.shape(mu)).copy()
        sigma = np.sqrt(self.sigma2_y)
        noise = self.rng.normal(loc=0.0, scale=sigma, size=(n,) + np.shape(mu))
        return mu + noise


@dataclass
class LinearGaussianProcess(GenerativeProcess):
    """Specialization of :class:`GenerativeProcess` with ``g_E(x) = beta0 + beta1 * x``.

    A convenience subclass that handles the most common case in Chapters 1–5.
    """

    beta0: float = 0.0
    beta1: float = 1.0

    def __init__(
        self,
        beta0: float = 0.0,
        beta1: float = 1.0,
        sigma2_y: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        nonlinear: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    ) -> None:
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)
        self._nonlinear = nonlinear

        def g(x_star, beta0=self.beta0, beta1=self.beta1, nonlinear=self._nonlinear):
            x_star = np.asarray(x_star, dtype=float)
            psi = nonlinear(x_star) if nonlinear is not None else x_star
            return beta0 + beta1 * psi

        super().__init__(
            g_E=g,
            theta={},
            sigma2_y=sigma2_y,
            rng=rng,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nl = "linear" if self._nonlinear is None else "nonlinear"
        return (
            f"LinearGaussianProcess({nl}, beta0={self.beta0:g}, "
            f"beta1={self.beta1:g}, sigma2_y={self.sigma2_y:g})"
        )


# ---------------------------------------------------------------------------
# Multivariate linear-Gaussian process: y = Theta x* + b + omega
# ---------------------------------------------------------------------------


@dataclass
class LinearGaussianMVProcess:
    """Multivariate linear-Gaussian generative process.

    Generates observations ``y = Theta @ x_star + b + omega`` where ``omega``
    is drawn from a multivariate normal with covariance ``cov_y``.

    Parameters
    ----------
    Theta : np.ndarray, shape ``(D, C)``
        Mixing matrix mapping ``C``-dim states to ``D``-dim observations.
    b : np.ndarray, shape ``(D,)`` or ``None``
        Optional offset. ``None`` is treated as zero.
    cov_y : np.ndarray, shape ``(D, D)``
        Observation-noise covariance.
    rng : np.random.Generator, optional
        Random number generator.
    """

    Theta: np.ndarray
    cov_y: np.ndarray
    b: Optional[np.ndarray] = None
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self.Theta = np.asarray(self.Theta, dtype=float)
        if self.Theta.ndim != 2:
            raise ValueError("Theta must be a 2-D matrix")
        self.cov_y = np.asarray(self.cov_y, dtype=float)
        d = self.Theta.shape[0]
        if self.cov_y.shape != (d, d):
            raise ValueError(
                f"cov_y must be shape ({d}, {d}), got {self.cov_y.shape}"
            )
        if self.b is None:
            self.b = np.zeros(d)
        else:
            self.b = np.asarray(self.b, dtype=float).reshape(-1)
            if self.b.size != d:
                raise ValueError(f"b must have length {d}, got {self.b.size}")
        if self.rng is None:
            self.rng = np.random.default_rng()

    @property
    def D(self) -> int:
        """Observation dimensionality."""
        return self.Theta.shape[0]

    @property
    def C(self) -> int:
        """Latent (state) dimensionality."""
        return self.Theta.shape[1]

    def mean(self, x_star: np.ndarray) -> np.ndarray:
        """Noise-free generator at ``x_star``.

        Accepts either a single state vector of shape ``(C,)`` or a batch
        ``(N, C)`` and broadcasts the offset accordingly.
        """
        x_star = np.asarray(x_star, dtype=float)
        if x_star.ndim == 1:
            return self.Theta @ x_star + self.b
        return x_star @ self.Theta.T + self.b

    def sample(self, x_star: np.ndarray, n: int = 1) -> np.ndarray:
        """Draw ``n`` observations from ``N(Theta x_star + b, cov_y)``.

        Returns shape ``(n, D)``.
        """
        mu = self.mean(x_star)
        if mu.ndim == 1:
            return mvn_sample(mu, self.cov_y, n=n, rng=self.rng)
        # Per-row means: independent samples for each x_star row.
        out = np.empty((mu.shape[0], n, self.D))
        for i, m in enumerate(mu):
            out[i] = mvn_sample(m, self.cov_y, n=n, rng=self.rng)
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LinearGaussianMVProcess(Theta={self.Theta.shape}, "
            f"D={self.D}, C={self.C})"
        )
