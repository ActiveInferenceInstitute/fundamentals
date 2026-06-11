"""Appendix C.11 Bayesian model comparison and selection helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _finite_vector(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite one-dimensional vector."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _logsumexp(values: np.ndarray) -> float:
    """Return stable log-sum-exp for one vector."""
    shift = float(np.max(values))
    return float(shift + np.log(np.sum(np.exp(values - shift))))


def model_posterior(log_evidences: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return posterior model probabilities from log model evidences."""
    logs = _finite_vector("log_evidences", log_evidences)
    return np.exp(logs - _logsumexp(logs))


def log_bayes_factor(log_evidence_a: float, log_evidence_b: float) -> float:
    """Return log Bayes factor comparing model A to model B."""
    a = float(log_evidence_a)
    b = float(log_evidence_b)
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("log evidences must be finite")
    return a - b


def bayes_factor(log_evidence_a: float, log_evidence_b: float) -> float:
    """Return Bayes factor comparing model A to model B."""
    return float(np.exp(log_bayes_factor(log_evidence_a, log_evidence_b)))


def bayesian_model_average(
    estimates: np.ndarray | Sequence[float],
    log_evidences: np.ndarray | Sequence[float],
) -> float:
    """Return a scalar estimate averaged under posterior model probabilities."""
    values = _finite_vector("estimates", estimates)
    weights = model_posterior(log_evidences)
    if values.shape != weights.shape:
        raise ValueError("estimates and log_evidences must share a shape")
    return float(weights @ values)


def bayesian_model_reduction(
    full_log_evidence: float,
    *,
    complexity_delta: float,
    accuracy_delta: float = 0.0,
) -> float:
    """Score a reduced model from full evidence plus accuracy/complexity deltas."""
    full = float(full_log_evidence)
    complexity = float(complexity_delta)
    accuracy = float(accuracy_delta)
    if not np.isfinite(full + complexity + accuracy):
        raise ValueError("all arguments must be finite")
    return full + accuracy + complexity


def bayesian_model_expansion(
    base_log_evidence: float,
    *,
    accuracy_gain: float,
    complexity_cost: float,
) -> float:
    """Score an expanded model by adding accuracy gain and subtracting complexity cost."""
    base = float(base_log_evidence)
    gain = float(accuracy_gain)
    cost = float(complexity_cost)
    if not np.isfinite(base + gain + cost) or cost < 0.0:
        raise ValueError("arguments must be finite and complexity_cost non-negative")
    return base + gain - cost


__all__ = [
    "bayes_factor",
    "bayesian_model_average",
    "bayesian_model_expansion",
    "bayesian_model_reduction",
    "log_bayes_factor",
    "model_posterior",
]
