"""Numerically stable probability density utilities used throughout the package.

Univariate helpers (``gaussian_pdf``, ``gaussian_log_pdf``, ``uniform_pdf``,
``dirac_like_pdf``) are fully broadcastable so a likelihood can be evaluated
across hundreds of candidate hidden-state values at once.

Multivariate helpers (``mvn_pdf``, ``mvn_log_pdf``, ``mvn_sample``,
``mahalanobis_squared``) cover the Chapter 3 / Chapter 6+ machinery: vector
inputs, full covariance matrices, and grid evaluation on a ``(G, D)`` array
of candidate states.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

_LOG_2PI = float(np.log(2.0 * np.pi))


def gaussian_pdf(x: ArrayLike, mu: ArrayLike, sigma2: ArrayLike) -> np.ndarray:
    """Univariate Gaussian density ``N(x ; mu, sigma2)``.

    Parameters
    ----------
    x, mu, sigma2 : array-like
        Broadcastable arrays. ``sigma2`` is the *variance*, not the standard
        deviation, matching the book's notation.

    Returns
    -------
    np.ndarray
        Density evaluated at ``x``.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    if np.any(sigma2 <= 0):
        raise ValueError("sigma2 must be strictly positive")
    coeff = 1.0 / np.sqrt(2.0 * np.pi * sigma2)
    return coeff * np.exp(-0.5 * (x - mu) ** 2 / sigma2)


def gaussian_log_pdf(x: ArrayLike, mu: ArrayLike, sigma2: ArrayLike) -> np.ndarray:
    """Log of :func:`gaussian_pdf`, computed without exponentiating first."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    if np.any(sigma2 <= 0):
        raise ValueError("sigma2 must be strictly positive")
    return -0.5 * (_LOG_2PI + np.log(sigma2) + (x - mu) ** 2 / sigma2)


def uniform_pdf(x: ArrayLike, low: float, high: float) -> np.ndarray:
    """Uniform density on ``[low, high]`` — zero elsewhere."""
    if high <= low:
        raise ValueError("high must be greater than low")
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    inside = (x >= low) & (x <= high)
    out[inside] = 1.0 / (high - low)
    return out


def dirac_like_pdf(
    x: ArrayLike,
    location: ArrayLike,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """A narrow Gaussian standing in for a Dirac-delta in grid computations.

    Useful for the deterministic-likelihood examples where the variance is
    effectively zero. ``epsilon`` is the standard deviation of the proxy.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return gaussian_pdf(x, mu=location, sigma2=epsilon ** 2)


def normalize_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Normalize an unnormalized density on ``grid`` so it integrates to 1.

    Uses the trapezoid rule, which is more accurate than a Riemann sum for the
    smooth densities encountered in this book.
    """
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if values.shape != grid.shape:
        raise ValueError("values and grid must share shape")
    mass = np.trapezoid(values, grid)
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("density has non-positive or non-finite mass")
    return values / mass


# ---------------------------------------------------------------------------
# Multivariate helpers
# ---------------------------------------------------------------------------


def _validate_cov(cov: np.ndarray, dim: int) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (dim, dim):
        raise ValueError(f"covariance must have shape ({dim}, {dim}), got {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("covariance matrix must be symmetric")
    return cov


def mahalanobis_squared(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Squared Mahalanobis distance ``(x - mu)^T Σ^{-1} (x - mu)``.

    Parameters
    ----------
    x : np.ndarray, shape ``(D,)`` or ``(N, D)``
    mu : np.ndarray, shape ``(D,)``
    cov : np.ndarray, shape ``(D, D)``

    Returns
    -------
    np.ndarray
        Scalar if ``x`` is 1-D, length-``N`` array if ``x`` is 2-D.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    cov = _validate_cov(cov, mu.size)
    diff = x - mu
    # Solve Σ z = diff so we never explicitly invert.
    z = np.linalg.solve(cov, diff.T if diff.ndim > 1 else diff).T
    if diff.ndim == 1:
        return float(diff @ z)
    return np.einsum("nd,nd->n", diff, z)


def mvn_log_pdf(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Log density of a multivariate normal distribution.

    Uses Cholesky decomposition for numerical stability — never inverts the
    covariance matrix directly. Accepts either a single vector or a batch of
    row-vectors.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = _validate_cov(cov, mu.size)
    x = np.asarray(x, dtype=float)
    d = mu.size
    L = np.linalg.cholesky(cov)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    diff = x - mu
    # Solve L z = diff so quad = z^T z = diff^T Σ^{-1} diff.
    if diff.ndim == 1:
        from scipy.linalg import solve_triangular  # local import keeps top clean
        z = solve_triangular(L, diff, lower=True)
        quad = float(z @ z)
    else:
        from scipy.linalg import solve_triangular
        z = solve_triangular(L, diff.T, lower=True)
        quad = np.sum(z ** 2, axis=0)
    return -0.5 * (d * _LOG_2PI + log_det + quad)


def mvn_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Density of a multivariate normal distribution."""
    return np.exp(mvn_log_pdf(x, mu, cov))


def mvn_sample(
    mu: np.ndarray,
    cov: np.ndarray,
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw ``n`` samples from a multivariate normal via Cholesky.

    Returns shape ``(n, D)``; if ``n == 1`` you can ``.squeeze(0)`` if you
    prefer a 1-D vector.
    """
    if rng is None:
        rng = np.random.default_rng()
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = _validate_cov(cov, mu.size)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, mu.size))
    return mu + z @ L.T


def isotropic_cov(dim: int, variance: float = 1.0) -> np.ndarray:
    """Return ``variance * I_d`` — the spherical / isotropic covariance matrix."""
    if variance <= 0:
        raise ValueError("variance must be positive")
    if dim < 1:
        raise ValueError("dim must be at least 1")
    return float(variance) * np.eye(int(dim))


def diagonal_cov(variances: np.ndarray) -> np.ndarray:
    """Construct a diagonal covariance from a 1-D vector of per-dimension variances."""
    variances = np.asarray(variances, dtype=float).reshape(-1)
    if np.any(variances <= 0):
        raise ValueError("all variances must be positive")
    return np.diag(variances)
