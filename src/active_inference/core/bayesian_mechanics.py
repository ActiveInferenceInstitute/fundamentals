"""Bayesian-mechanics and Markov-blanket helpers for Chapter 14 demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ergodic import density_entropy, entropy_upper_bound_from_vfe, ergodic_density
from .validators import require_finite_array


@dataclass(frozen=True)
class MarkovBlanketFlow:
    """Internal, blanket, and external state trajectories for a simple blanket flow."""

    time: np.ndarray
    internal: np.ndarray
    blanket: np.ndarray
    external: np.ndarray


def simulate_markov_blanket_flow(n_steps: int = 160) -> MarkovBlanketFlow:
    """Return deterministic coupled external, blanket, and internal trajectories."""
    if n_steps < 4:
        raise ValueError("n_steps must be at least 4")
    time = np.linspace(0.0, 8.0, n_steps)
    external = np.sin(time)
    blanket = 0.75 * np.sin(time - 0.35)
    internal = 0.55 * np.sin(time - 0.7)
    return MarkovBlanketFlow(time=time, internal=internal, blanket=blanket, external=external)


@dataclass(frozen=True)
class BayesianMechanicsSummary:
    """Entropy and bound summary for an ergodic density."""

    grid: np.ndarray
    density: np.ndarray
    entropy: float
    upper_bound: float
    gap: float


def bayesian_mechanics_summary(
    trajectory: np.ndarray | Sequence[float],
    *,
    bins: int = 80,
    vfe_margin: float = 0.5,
) -> BayesianMechanicsSummary:
    """Summarize an ergodic trajectory through entropy and VFE-like upper bound."""
    values = require_finite_array(trajectory, name="trajectory")
    if values.ndim != 1:
        raise ValueError("trajectory must be one-dimensional")
    grid, density = ergodic_density(values, bins=bins)
    entropy = density_entropy(grid, density)
    bound = entropy_upper_bound_from_vfe(entropy, entropy + float(vfe_margin))
    return BayesianMechanicsSummary(
        grid=grid,
        density=density,
        entropy=bound.entropy,
        upper_bound=bound.upper_bound,
        gap=bound.gap,
    )


def blanket_coupling_matrix(flow: MarkovBlanketFlow) -> np.ndarray:
    """Return the correlation matrix among external, blanket, and internal states."""
    stacked = np.vstack([flow.external, flow.blanket, flow.internal])
    return np.corrcoef(stacked)


def viability_indicator(states: np.ndarray | Sequence[float], lower: float, upper: float) -> np.ndarray:
    """Return a binary viability indicator for states inside a survival interval."""
    values = require_finite_array(states, name="states")
    lo = float(lower)
    hi = float(upper)
    if not np.isfinite(lo + hi) or lo >= hi:
        raise ValueError("lower and upper must be finite with lower < upper")
    return ((values >= lo) & (values <= hi)).astype(float)


def survival_probability(states: np.ndarray | Sequence[float], lower: float, upper: float) -> float:
    """Return the empirical occupancy probability of a viability interval."""
    return float(np.mean(viability_indicator(states, lower, upper)))


def entropy_vfe_bound_curve(
    entropy: np.ndarray | Sequence[float],
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return entropy and VFE-like upper-bound curves for Chapter 14 demos."""
    ent = require_finite_array(entropy, name="entropy")
    if ent.ndim != 1:
        raise ValueError("entropy must be one-dimensional")
    gap = float(margin)
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("margin must be finite and non-negative")
    return ent, ent + gap


def phase1_fep_bridge(
    sensory_entropy: float,
    variational_free_energy: float,
    viability: float,
) -> np.ndarray:
    """Return a compact Phase-I FEP bridge vector: entropy, VFE, viability gap."""
    entropy = float(sensory_entropy)
    vfe = float(variational_free_energy)
    alive = float(viability)
    if not np.isfinite(entropy + vfe + alive) or alive < 0.0 or alive > 1.0:
        raise ValueError("arguments must be finite and viability must lie in [0, 1]")
    return np.array([entropy, vfe, 1.0 - alive], dtype=float)


__all__ = [
    "BayesianMechanicsSummary",
    "MarkovBlanketFlow",
    "bayesian_mechanics_summary",
    "blanket_coupling_matrix",
    "entropy_vfe_bound_curve",
    "phase1_fep_bridge",
    "simulate_markov_blanket_flow",
    "survival_probability",
    "viability_indicator",
]
