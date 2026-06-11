"""Small Part III simulations built from the discrete POMDP extension helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.pomdp import POMDPModel, one_hot
from ..core.pomdp_extensions import (
    TreePolicySearchResult,
    forget_dirichlet_counts,
    habit_prior_from_counts,
    path_based_policy_scores,
    sophisticated_policy_trace,
    structure_posterior,
    time_dependent_preferences,
    tree_policy_search,
    update_preference_counts,
)


@dataclass(frozen=True)
class StatePreferenceResult:
    """Trace of normalized preferences and selected target states over time."""

    schedule: np.ndarray
    target_states: np.ndarray
    posterior: np.ndarray


def simulate_state_preference_schedule(
    log_preferences: np.ndarray | Sequence[Sequence[float]],
) -> StatePreferenceResult:
    """Normalize time-varying preferences and return their target-state trace."""
    schedule = time_dependent_preferences(log_preferences)
    targets = np.argmax(schedule, axis=1)
    return StatePreferenceResult(schedule=schedule, target_states=targets, posterior=schedule[-1])


@dataclass(frozen=True)
class ParameterForgettingResult:
    """Before/after Dirichlet pseudocounts plus uncertainty diagnostics for forgetting demos."""

    before: np.ndarray
    after: np.ndarray
    uncertainty_before: np.ndarray
    uncertainty_after: np.ndarray


def simulate_parameter_forgetting(
    counts: np.ndarray | Sequence[float],
    *,
    rate: float,
    floor: float = 1.0,
) -> ParameterForgettingResult:
    """Apply forgetting and expose the inverse-count uncertainty change."""
    before = np.asarray(counts, dtype=float)
    after = forget_dirichlet_counts(before, rate=rate, floor=floor)
    return ParameterForgettingResult(
        before=before,
        after=after,
        uncertainty_before=1.0 / np.maximum(before, 1e-12),
        uncertainty_after=1.0 / np.maximum(after, 1e-12),
    )


@dataclass(frozen=True)
class PreferenceHabitLearningResult:
    """Preference ``C`` counts and habit ``E`` prior after simple learning."""

    preference_counts: np.ndarray
    normalized_preferences: np.ndarray
    habit_prior: np.ndarray


def simulate_preference_habit_learning(
    observations: Sequence[int],
    actions: Sequence[int],
    *,
    n_outcomes: int = 3,
    n_actions: int = 2,
) -> PreferenceHabitLearningResult:
    """Learn simple C/E-style pseudocounts from observations and actions."""
    c_counts = update_preference_counts(np.ones(n_outcomes), observations)
    e_counts = update_preference_counts(np.ones(n_actions), actions)
    return PreferenceHabitLearningResult(
        preference_counts=c_counts,
        normalized_preferences=c_counts / np.sum(c_counts),
        habit_prior=habit_prior_from_counts(e_counts),
    )


@dataclass(frozen=True)
class PathPolicyResult:
    """Path-based policy scores and posterior for policy-pruning demos."""

    policies: np.ndarray
    scores: np.ndarray
    posterior: np.ndarray


def simulate_path_policy_computation() -> PathPolicyResult:
    """Return deterministic path-cost and posterior arrays for §11.2.8 demos."""
    policies = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=int)
    scores = path_based_policy_scores([0.4, 0.2], policies, terminal_costs=[0.3, 0.0])
    posterior = np.exp(-(scores - np.min(scores)))
    posterior = posterior / posterior.sum()
    return PathPolicyResult(policies=policies, scores=scores, posterior=posterior)


@dataclass(frozen=True)
class StructureLearningResult:
    """Candidate structure evidence, posterior probabilities, and selected model index."""

    evidence: np.ndarray
    posterior: np.ndarray
    selected_index: int


def simulate_structure_learning(
    evidence: np.ndarray | Sequence[float],
    *,
    complexity: float = 0.0,
) -> StructureLearningResult:
    """Return a posterior over candidate structures from penalized evidence."""
    posterior = structure_posterior(evidence, complexity=complexity)
    return StructureLearningResult(
        evidence=np.asarray(evidence, dtype=float),
        posterior=posterior,
        selected_index=int(np.argmax(posterior)),
    )


def make_line_world(n_states: int = 5, goal: int | None = None) -> POMDPModel:
    """Build a deterministic one-dimensional POMDP for Part III planning demos."""
    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    if goal is None:
        goal = n_states - 1
    A = np.eye(n_states)
    B = np.zeros((2, n_states, n_states), dtype=float)
    for state in range(n_states):
        B[0, max(0, state - 1), state] = 1.0
        B[1, min(n_states - 1, state + 1), state] = 1.0
    return POMDPModel(A=A, D=one_hot(0, n_states), B=B, C=one_hot(goal, n_states))


def simulate_sophisticated_planning(
    *,
    n_states: int = 5,
    horizon: int = 3,
    gamma: float = 4.0,
) -> tuple[TreePolicySearchResult, np.ndarray, np.ndarray]:
    """Run bounded tree search in a line world and return policy/belief diagnostics."""
    model = make_line_world(n_states=n_states)
    belief = model.D.copy()
    policies = np.array(np.meshgrid(*([np.arange(2)] * horizon))).T.reshape(-1, horizon)
    result = tree_policy_search(model, belief, policies, model.C, gamma=gamma)
    trace = sophisticated_policy_trace(model, belief, result.best_policy, model.C)
    return result, trace.beliefs, trace.belief_entropies


__all__ = [
    "ParameterForgettingResult",
    "PathPolicyResult",
    "PreferenceHabitLearningResult",
    "StatePreferenceResult",
    "StructureLearningResult",
    "make_line_world",
    "simulate_parameter_forgetting",
    "simulate_path_policy_computation",
    "simulate_preference_habit_learning",
    "simulate_sophisticated_planning",
    "simulate_state_preference_schedule",
    "simulate_structure_learning",
]
