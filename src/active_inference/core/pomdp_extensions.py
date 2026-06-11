"""Planning-extension helpers for Part III discrete active inference topics.

The functions here intentionally build on the Chapter 9/10 POMDP primitives.
They provide compact, testable teaching forms for sophisticated inference,
time-dependent preferences, parameter forgetting, tree policy search, and
structure learning without changing the established ``POMDPModel`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pomdp import POMDPModel, evaluate_policy, policy_posterior, predict_state, softmax
from .validators import require_finite_array, require_positive_scalar


def time_dependent_preferences(preferences: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return a validated ``(T, O)`` schedule of observation preferences.

    Rows are normalized through :func:`active_inference.core.pomdp.softmax` so callers can
    specify log-preferences at each future step while downstream EFE code receives
    probability vectors.
    """
    schedule = require_finite_array(preferences, name="preferences")
    if schedule.ndim != 2 or schedule.shape[0] == 0 or schedule.shape[1] == 0:
        raise ValueError("preferences must have shape (T, O)")
    return np.vstack([softmax(row) for row in schedule])


def forget_dirichlet_counts(
    counts: np.ndarray | Sequence[float],
    *,
    rate: float,
    floor: float = 1.0,
) -> np.ndarray:
    """Apply exponential forgetting to Dirichlet pseudocounts with a positive floor."""
    arr = require_finite_array(counts, name="counts")
    if np.any(arr < 0.0):
        raise ValueError("counts must be non-negative")
    if rate < 0.0 or rate > 1.0:
        raise ValueError("rate must lie in [0, 1]")
    floor_value = require_positive_scalar(floor, name="floor")
    return floor_value + (1.0 - rate) * np.maximum(arr - floor_value, 0.0)


def structure_log_evidence(scores: np.ndarray | Sequence[float], complexity: float = 0.0) -> np.ndarray:
    """Return simple structure log-evidence scores penalized by model complexity."""
    values = require_finite_array(scores, name="scores")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty 1-D vector")
    penalty = float(complexity)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("complexity must be finite and non-negative")
    return values - penalty * np.arange(values.size, dtype=float)


def structure_posterior(scores: np.ndarray | Sequence[float], complexity: float = 0.0) -> np.ndarray:
    """Return a posterior over candidate structures from penalized log evidence."""
    return softmax(structure_log_evidence(scores, complexity=complexity))


def update_preference_counts(
    counts: np.ndarray | Sequence[float],
    observations: Sequence[int],
    *,
    learning_rate: float = 1.0,
) -> np.ndarray:
    """Update preference pseudocounts for the Chapter 11 ``C`` learning demo."""
    arr = require_finite_array(counts, name="counts")
    if arr.ndim != 1 or np.any(arr < 0.0):
        raise ValueError("counts must be a non-negative 1-D vector")
    rate = float(learning_rate)
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("learning_rate must be finite and non-negative")
    updated = arr.astype(float, copy=True)
    for obs in observations:
        index = int(obs)
        if index < 0 or index >= updated.size:
            raise ValueError("observation index is out of bounds")
        updated[index] += rate
    return updated


def habit_prior_from_counts(counts: np.ndarray | Sequence[float], precision: float = 1.0) -> np.ndarray:
    """Return an ``E``-style habit prior from action/policy counts."""
    arr = require_finite_array(counts, name="counts")
    if arr.ndim != 1 or np.any(arr < 0.0):
        raise ValueError("counts must be a non-negative 1-D vector")
    gamma = require_positive_scalar(precision, name="precision")
    return softmax(gamma * np.log(arr + 1.0))


def path_based_policy_scores(
    transition_costs: np.ndarray | Sequence[float],
    policies: Sequence[Sequence[int]],
    terminal_costs: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Compute path-based policy costs by summing action costs along each path."""
    costs = require_finite_array(transition_costs, name="transition_costs")
    if costs.ndim != 1:
        raise ValueError("transition_costs must be a 1-D vector")
    terminal = np.zeros(costs.shape, dtype=float) if terminal_costs is None else require_finite_array(terminal_costs, name="terminal_costs")
    if terminal.shape != costs.shape:
        raise ValueError("terminal_costs must match transition_costs")
    scores = []
    for policy in policies:
        actions = np.asarray(policy, dtype=int)
        if actions.ndim != 1 or actions.size == 0:
            raise ValueError("each policy must be a non-empty action sequence")
        if np.any(actions < 0) or np.any(actions >= costs.size):
            raise ValueError("policy action index is out of bounds")
        scores.append(float(np.sum(costs[actions]) + terminal[actions[-1]]))
    return np.asarray(scores, dtype=float)


@dataclass(frozen=True)
class TreePolicySearchResult:
    """Result of a bounded tree policy search over action sequences."""

    policies: np.ndarray
    expected_free_energies: np.ndarray
    posterior: np.ndarray
    best_policy: np.ndarray


def tree_policy_search(
    model: POMDPModel,
    belief: np.ndarray | Sequence[float],
    policies: Sequence[Sequence[int]],
    preferences: np.ndarray | Sequence[float],
    *,
    gamma: float = 4.0,
) -> TreePolicySearchResult:
    """Score candidate action trees and return a posterior over policies."""
    policy_array = np.asarray(policies, dtype=int)
    if policy_array.ndim != 2 or policy_array.shape[0] == 0:
        raise ValueError("policies must have shape (P, horizon)")
    belief_arr = require_finite_array(belief, name="belief")
    pref_arr = require_finite_array(preferences, name="preferences")
    scores = np.array(
        [evaluate_policy(model, belief_arr, list(policy), pref_arr) for policy in policy_array],
        dtype=float,
    )
    posterior = policy_posterior(scores, gamma=gamma)
    return TreePolicySearchResult(
        policies=policy_array,
        expected_free_energies=scores,
        posterior=posterior,
        best_policy=policy_array[int(np.argmax(posterior))],
    )


@dataclass(frozen=True)
class SophisticatedInferenceTrace:
    """Belief and policy traces for one sophisticated-inference lookahead."""

    policy: np.ndarray
    beliefs: np.ndarray
    belief_entropies: np.ndarray
    expected_free_energy: float


def sophisticated_policy_trace(
    model: POMDPModel,
    belief: np.ndarray | Sequence[float],
    policy: Sequence[int],
    preferences: np.ndarray | Sequence[float],
) -> SophisticatedInferenceTrace:
    """Roll one policy forward, tracking future beliefs and entropy reduction."""
    current = require_finite_array(belief, name="belief")
    pref_arr = require_finite_array(preferences, name="preferences")
    policy_array = np.asarray(policy, dtype=int)
    if policy_array.ndim != 1 or policy_array.size == 0:
        raise ValueError("policy must be a non-empty 1-D action sequence")
    beliefs = []
    entropies = []
    for action in policy_array:
        current = predict_state(model, current, int(action))
        beliefs.append(current)
        positive = np.clip(current, 1e-12, 1.0)
        entropies.append(float(-positive @ np.log(positive)))
    return SophisticatedInferenceTrace(
        policy=policy_array,
        beliefs=np.asarray(beliefs),
        belief_entropies=np.asarray(entropies),
        expected_free_energy=evaluate_policy(
            model,
            require_finite_array(belief, name="belief"),
            list(policy_array),
            pref_arr,
        ),
    )


__all__ = [
    "SophisticatedInferenceTrace",
    "TreePolicySearchResult",
    "forget_dirichlet_counts",
    "habit_prior_from_counts",
    "path_based_policy_scores",
    "sophisticated_policy_trace",
    "structure_log_evidence",
    "structure_posterior",
    "time_dependent_preferences",
    "tree_policy_search",
    "update_preference_counts",
]
