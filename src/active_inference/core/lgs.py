"""Linear Gaussian Systems — multivariate hidden-state inference in closed form.

A linear Gaussian system specifies::

    p(x)     = N(x ; mx, Σx)
    p(y | x) = N(y ; Θ x + b, Σy)

When both densities are Gaussian, the posterior over the hidden state given
one or more observations is also Gaussian and has a closed-form mean and
covariance — no grid approximation needed.

This module provides:

* :class:`LinearGaussianSystem` — the model definition.
* :meth:`LinearGaussianSystem.posterior` — single-observation posterior.
* :meth:`LinearGaussianSystem.posterior_batch` — i.i.d. observations of a
  single hidden state (the "sensor fusion" pattern).

All updates use Cholesky-based solves rather than explicit matrix inverses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LGSPosterior:
    """Closed-form Gaussian posterior over the hidden state.

    Attributes
    ----------
    mean : np.ndarray, shape ``(C,)``
    cov : np.ndarray, shape ``(C, C)``
    """

    mean: np.ndarray
    cov: np.ndarray

    @property
    def precision(self) -> np.ndarray:
        """Inverse of the posterior covariance — the precision matrix ``Σ⁻¹``."""
        return np.linalg.inv(self.cov)

    def std(self) -> np.ndarray:
        """Per-component standard deviations from the diagonal of ``cov``."""
        return np.sqrt(np.diag(self.cov))

    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw ``n`` samples from this posterior MVN.

        Returns shape ``(n, C)``.
        """
        from .distributions import mvn_sample
        return mvn_sample(self.mean, self.cov, n=n, rng=rng)

    def summary(self, ndigits: int = 4) -> str:
        """Human-readable one-line summary."""
        m = np.round(self.mean, ndigits)
        s = np.round(self.std(), ndigits)
        return f"LGSPosterior(mean={m.tolist()}, std={s.tolist()})"


@dataclass
class LinearGaussianSystem:
    """Closed-form linear Gaussian system.

    Parameters
    ----------
    Theta : np.ndarray, shape ``(D, C)``
        Mixing / projection matrix.
    cov_y : np.ndarray, shape ``(D, D)``
        Likelihood covariance.
    mx : np.ndarray, shape ``(C,)``
        Prior mean.
    cov_x : np.ndarray, shape ``(C, C)``
        Prior covariance.
    b : np.ndarray, shape ``(D,)`` or ``None``
        Optional offset; ``None`` is treated as zero.
    """

    Theta: np.ndarray
    cov_y: np.ndarray
    mx: np.ndarray
    cov_x: np.ndarray
    b: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.Theta = np.asarray(self.Theta, dtype=float)
        if self.Theta.ndim != 2:
            raise ValueError("Theta must be 2-D")
        d, c = self.Theta.shape
        self.cov_y = np.asarray(self.cov_y, dtype=float)
        if self.cov_y.shape != (d, d):
            raise ValueError(f"cov_y must be ({d},{d})")
        self.mx = np.asarray(self.mx, dtype=float).reshape(-1)
        if self.mx.size != c:
            raise ValueError(f"mx must have length {c}")
        self.cov_x = np.asarray(self.cov_x, dtype=float)
        if self.cov_x.shape != (c, c):
            raise ValueError(f"cov_x must be ({c},{c})")
        if self.b is None:
            self.b = np.zeros(d)
        else:
            self.b = np.asarray(self.b, dtype=float).reshape(-1)
            if self.b.size != d:
                raise ValueError(f"b must have length {d}")

    @property
    def D(self) -> int:
        """Number of observation dimensions (rows of ``Theta``)."""
        return self.Theta.shape[0]

    @property
    def C(self) -> int:
        """Number of latent state dimensions (columns of ``Theta``)."""
        return self.Theta.shape[1]

    # ------------------------------------------------------------------
    # Posterior updates
    # ------------------------------------------------------------------

    def posterior(self, y: np.ndarray) -> LGSPosterior:
        """Posterior over ``x`` given a single observation ``y``."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != self.D:
            raise ValueError(f"y must have length {self.D}, got {y.size}")
        prec_x = np.linalg.inv(self.cov_x)
        prec_y = np.linalg.inv(self.cov_y)
        cov_post = np.linalg.inv(prec_x + self.Theta.T @ prec_y @ self.Theta)
        mean_post = cov_post @ (
            self.Theta.T @ prec_y @ (y - self.b) + prec_x @ self.mx
        )
        return LGSPosterior(mean=mean_post, cov=cov_post)

    def posterior_batch(self, Y: np.ndarray) -> LGSPosterior:
        """Posterior over a single ``x`` given ``N`` i.i.d. observations.

        Parameters
        ----------
        Y : np.ndarray, shape ``(N, D)``
            Each row is one independent draw of ``y`` produced by the same
            hidden state.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or Y.shape[1] != self.D:
            raise ValueError(f"Y must be (N, {self.D})")
        n = Y.shape[0]
        prec_x = np.linalg.inv(self.cov_x)
        prec_y = np.linalg.inv(self.cov_y)
        cov_post = np.linalg.inv(prec_x + n * (self.Theta.T @ prec_y @ self.Theta))
        mean_post = cov_post @ (
            self.Theta.T @ prec_y @ (Y - self.b).sum(axis=0) + prec_x @ self.mx
        )
        return LGSPosterior(mean=mean_post, cov=cov_post)

    def predictive_mean(self, x: np.ndarray) -> np.ndarray:
        """Forward pass — observation mean for a given hidden state ``x``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return self.Theta @ x + self.b
        return x @ self.Theta.T + self.b

    def sample_observations(
        self,
        x: np.ndarray,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample ``n`` noisy observations from the likelihood at hidden state ``x``.

        Returns shape ``(n, D)``.
        """
        from .distributions import mvn_sample
        mu = self.predictive_mean(x)
        if mu.ndim != 1:
            raise ValueError("x must be a single hidden-state vector")
        return mvn_sample(mu, self.cov_y, n=n, rng=rng)

    def __repr__(self) -> str:  # pragma: no cover
        return f"LinearGaussianSystem(D={self.D}, C={self.C})"
