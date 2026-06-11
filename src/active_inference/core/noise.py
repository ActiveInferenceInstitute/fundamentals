"""Appendix C.9 colored-noise covariance and sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _finite_vector(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite one-dimensional vector."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a finite 1-D vector with at least two entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def squared_exponential_covariance(
    time: np.ndarray | Sequence[float],
    *,
    length_scale: float,
    variance: float = 1.0,
) -> np.ndarray:
    """Return a smooth colored-noise covariance over time points."""
    t = _finite_vector("time", time)
    ell = float(length_scale)
    var = float(variance)
    if ell <= 0.0 or var <= 0.0:
        raise ValueError("length_scale and variance must be positive")
    distances = t[:, None] - t[None, :]
    return var * np.exp(-0.5 * (distances / ell) ** 2)


def colored_noise_precision(
    time: np.ndarray | Sequence[float],
    *,
    length_scale: float,
    variance: float = 1.0,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Return the inverse covariance for a smooth colored-noise process."""
    cov = squared_exponential_covariance(time, length_scale=length_scale, variance=variance)
    if jitter <= 0.0:
        raise ValueError("jitter must be positive")
    precision = np.linalg.inv(cov + jitter * np.eye(cov.shape[0]))
    return 0.5 * (precision + precision.T)


def sample_colored_noise(
    time: np.ndarray | Sequence[float],
    *,
    length_scale: float,
    variance: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one smooth colored-noise trajectory from the covariance."""
    generator = np.random.default_rng(0) if rng is None else rng
    cov = squared_exponential_covariance(time, length_scale=length_scale, variance=variance)
    return generator.multivariate_normal(np.zeros(cov.shape[0]), cov)


def finite_difference_derivative(values: np.ndarray | Sequence[float], time: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a stable first finite-difference derivative on the supplied grid."""
    y = _finite_vector("values", values)
    t = _finite_vector("time", time)
    if y.shape != t.shape:
        raise ValueError("values and time must share a shape")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return np.gradient(y, t)


__all__ = [
    "colored_noise_precision",
    "finite_difference_derivative",
    "sample_colored_noise",
    "squared_exponential_covariance",
]
