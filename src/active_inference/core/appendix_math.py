"""Appendix B/C mathematical helpers used by source-spine demos and tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.special import gammaln


def _finite_array(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite floating array."""
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def normalize_categorical(probabilities: np.ndarray | Sequence[float]) -> np.ndarray:
    """Normalize a non-negative categorical vector or tensor along its last axis."""
    probs = _finite_array("probabilities", probabilities)
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")
    total = np.sum(probs, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("probabilities must have positive mass")
    return probs / total


def joint_from_likelihood_prior(
    likelihood: np.ndarray | Sequence[Sequence[float]],
    prior: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return ``P(o, s) = P(o | s) P(s)`` for column-oriented likelihoods."""
    a = _finite_array("likelihood", likelihood)
    d = normalize_categorical(prior)
    if a.ndim != 2 or d.ndim != 1 or a.shape[1] != d.size:
        raise ValueError("likelihood must have shape (O, S) and prior shape (S,)")
    return a * d.reshape(1, -1)


def marginalize(joint: np.ndarray | Sequence[float], axis: int) -> np.ndarray:
    """Marginalize a joint probability table over all axes except ``axis``."""
    table = _finite_array("joint", joint)
    if np.any(table < 0.0) or np.sum(table) <= 0.0:
        raise ValueError("joint must be a non-negative table with positive mass")
    normed = table / np.sum(table)
    target = int(axis)
    if target < 0:
        target += normed.ndim
    if target < 0 or target >= normed.ndim:
        raise ValueError("axis is out of bounds")
    axes = tuple(i for i in range(normed.ndim) if i != target)
    return np.sum(normed, axis=axes)


def bayes_posterior_from_likelihood(
    likelihood: np.ndarray | Sequence[Sequence[float]],
    prior: np.ndarray | Sequence[float],
    observation: int,
) -> np.ndarray:
    """Return ``P(s | o)`` using Appendix B's column-oriented ``A[o, s]`` convention."""
    a = _finite_array("likelihood", likelihood)
    d = normalize_categorical(prior)
    obs = int(observation)
    if a.ndim != 2 or a.shape[1] != d.size:
        raise ValueError("likelihood must have shape (O, S)")
    if obs < 0 or obs >= a.shape[0]:
        raise ValueError("observation is out of bounds")
    posterior = a[obs] * d
    return normalize_categorical(posterior)


def gamma_pdf(x: np.ndarray | Sequence[float], shape: float, rate: float) -> np.ndarray:
    """Evaluate a finite Gamma density using Appendix C's shape/rate parameterization."""
    values = _finite_array("x", x)
    if shape <= 0.0 or rate <= 0.0:
        raise ValueError("shape and rate must be positive")
    out = np.zeros_like(values, dtype=float)
    mask = values >= 0.0
    log_pdf = shape * np.log(rate) - gammaln(shape) + (shape - 1.0) * np.log(np.maximum(values[mask], 1e-300)) - rate * values[mask]
    out[mask] = np.exp(log_pdf)
    return out


def dirichlet_mean(alpha: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return the normalized component-wise mean of a Dirichlet distribution."""
    concentration = _finite_array("alpha", alpha)
    if concentration.ndim != 1 or np.any(concentration <= 0.0):
        raise ValueError("alpha must be a positive 1-D vector")
    return concentration / np.sum(concentration)


def dirichlet_variance(alpha: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return per-component variances for a positive Dirichlet concentration vector."""
    concentration = _finite_array("alpha", alpha)
    if concentration.ndim != 1 or np.any(concentration <= 0.0):
        raise ValueError("alpha must be a positive 1-D vector")
    total = float(np.sum(concentration))
    return concentration * (total - concentration) / (total**2 * (total + 1.0))


def mutual_information(joint: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Return mutual information for a two-dimensional joint probability table."""
    table = _finite_array("joint", joint)
    if table.ndim != 2 or np.any(table < 0.0) or np.sum(table) <= 0.0:
        raise ValueError("joint must be a non-negative 2-D table with positive mass")
    pxy = table / np.sum(table)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    mask = pxy > 0.0
    return float(np.sum(pxy[mask] * (np.log(pxy[mask]) - np.log((px @ py)[mask]))))


def maximum_entropy_distribution(n_outcomes: int) -> np.ndarray:
    """Return the maximum-entropy categorical distribution over ``n_outcomes``."""
    n = int(n_outcomes)
    if n <= 0:
        raise ValueError("n_outcomes must be positive")
    return np.ones(n, dtype=float) / n


def euler_integrate(
    flow: Callable[[float, np.ndarray], np.ndarray],
    initial: np.ndarray | Sequence[float],
    *,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a small initial-value problem with explicit Euler integration."""
    if dt <= 0.0 or steps < 1:
        raise ValueError("dt must be positive and steps must be at least one")
    state = _finite_array("initial", initial).reshape(-1)
    time = np.arange(steps + 1, dtype=float) * dt
    states = np.empty((steps + 1, state.size), dtype=float)
    states[0] = state
    for i in range(steps):
        derivative = _finite_array("flow", flow(float(time[i]), states[i])).reshape(-1)
        if derivative.shape != state.shape:
            raise ValueError("flow must return the same shape as initial")
        states[i + 1] = states[i] + dt * derivative
    return time, states


def jensen_gap(weights: np.ndarray | Sequence[float], values: np.ndarray | Sequence[float]) -> float:
    """Return ``log(E[X]) - E[log(X)]`` for weighted positive values."""
    probs = normalize_categorical(weights)
    vals = _finite_array("values", values)
    if probs.shape != vals.shape or np.any(vals <= 0.0):
        raise ValueError("weights and positive values must share a shape")
    return float(np.log(probs @ vals) - probs @ np.log(vals))


def kronecker_delta(i: int, j: int) -> int:
    """Return one when two integer indices match and zero otherwise."""
    return int(int(i) == int(j))


def dirac_delta_grid(grid: np.ndarray | Sequence[float], location: float) -> np.ndarray:
    """Return a grid-normalized spike at the nearest grid point."""
    x = _finite_array("grid", grid)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least two points")
    idx = int(np.argmin(np.abs(x - float(location))))
    y = np.zeros_like(x, dtype=float)
    spacing = float(np.mean(np.diff(np.sort(x))))
    if spacing <= 0.0:
        raise ValueError("grid must have positive spacing")
    y[idx] = 1.0 / spacing
    return y


__all__ = [
    "bayes_posterior_from_likelihood",
    "dirac_delta_grid",
    "dirichlet_mean",
    "dirichlet_variance",
    "euler_integrate",
    "gamma_pdf",
    "joint_from_likelihood_prior",
    "jensen_gap",
    "kronecker_delta",
    "marginalize",
    "maximum_entropy_distribution",
    "mutual_information",
    "normalize_categorical",
]
