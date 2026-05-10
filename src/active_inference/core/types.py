"""Shape and type aliases used throughout the package.

These are deliberately *aliases* rather than newtypes — runtime
overhead is zero and they exist purely to clarify shapes at function
boundaries.

Convention:

- ``Vector``   : ``np.ndarray`` of shape ``(D,)``
- ``Matrix``   : ``np.ndarray`` of shape ``(R, C)``
- ``Grid1D``   : evenly-spaced ``np.ndarray`` of shape ``(G,)``
- ``DesignMatrix`` : data matrix ``(N, C)`` (no intercept) or ``(N, C+1)``
- ``CovMatrix``: square positive-definite ``np.ndarray`` of shape ``(D, D)``
- ``Probabilities`` : 1-D non-negative array summing (or integrating) to 1
- ``LogProb``  : a log-probability scalar or array
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Plain aliases — no runtime checking, just intent.
Vector = np.ndarray
Matrix = np.ndarray
Grid1D = np.ndarray
DesignMatrix = np.ndarray
CovMatrix = np.ndarray
Probabilities = np.ndarray
LogProb = Any  # may be a float or an ndarray; we let NumPy broadcast


def assert_cov(cov: np.ndarray, dim: int, *, name: str = "cov") -> np.ndarray:
    """Defensive check used by classes that take a covariance argument.

    Validates square shape, symmetry, and positive-definiteness via a
    Cholesky attempt. Raises ``ValueError`` with a helpful message on
    failure; otherwise returns ``cov`` unchanged.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (dim, dim):
        raise ValueError(
            f"{name} must be ({dim}, {dim}), got {cov.shape}"
        )
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite: {exc}") from exc
    return cov


def assert_probabilities(p: np.ndarray, *, name: str = "p",
                         tol: float = 1e-6) -> np.ndarray:
    """Validate a discrete probability vector: non-negative + sums to 1."""
    p = np.asarray(p, dtype=float)
    if np.any(p < -tol):
        raise ValueError(f"{name} has negative entries (min = {p.min():g})")
    s = float(p.sum())
    if abs(s - 1.0) > tol:
        raise ValueError(f"{name} must sum to 1, got {s:g}")
    return np.clip(p, 0.0, None)
