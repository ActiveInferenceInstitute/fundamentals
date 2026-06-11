"""Tests for Chapter 11 POMDP extension helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    forget_dirichlet_counts,
    habit_prior_from_counts,
    make_line_world,
    path_based_policy_scores,
    policy_posterior,
    structure_posterior,
    time_dependent_preferences,
    tree_policy_search,
    update_preference_counts,
)
from active_inference.core.pomdp import POMDPModel, expected_observation


def test_time_dependent_preferences_softmax_each_row() -> None:
    raw = np.array([[0.0, 2.0, -1.0], [4.0, 0.0, -2.0]])
    schedule = time_dependent_preferences(raw)
    assert schedule.shape == raw.shape
    np.testing.assert_allclose(schedule.sum(axis=1), np.ones(2))
    assert int(np.argmax(schedule[0])) == 1
    assert int(np.argmax(schedule[1])) == 0
    assert not np.allclose(schedule, raw)


def test_policy_posterior_uses_negative_efe_sign() -> None:
    scores = np.array([0.5, 3.0])
    posterior = policy_posterior(scores, gamma=4.0)
    assert posterior[0] > 0.99
    assert posterior[0] > posterior[1]


def test_tree_search_prefers_goal_reaching_policy() -> None:
    model = make_line_world(n_states=5)
    policies = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    result = tree_policy_search(model, model.D, policies, model.C, gamma=4.0)
    assert result.best_policy.tolist() == [1, 1, 1, 1]
    assert result.expected_free_energies[0] < result.expected_free_energies[1]
    assert result.posterior[0] > result.posterior[1]


def test_expected_observation_uses_column_stochastic_a_orientation() -> None:
    A = np.array([[0.9, 0.2], [0.1, 0.8]])
    model = POMDPModel(A=A, D=np.array([0.5, 0.5]))
    belief = np.array([0.25, 0.75])
    expected = expected_observation(model, belief)
    np.testing.assert_allclose(expected, A @ belief)
    assert not np.allclose(expected, A.T @ belief)


def test_forgetting_keeps_floor_and_rejects_bad_rate() -> None:
    before = np.array([1.0, 6.0, 11.0])
    after = forget_dirichlet_counts(before, rate=0.2, floor=1.0)
    np.testing.assert_allclose(after, [1.0, 5.0, 9.0])
    assert np.all(after >= 1.0)
    with pytest.raises(ValueError):
        forget_dirichlet_counts(before, rate=1.5)


def test_structure_posterior_penalizes_complexity_by_index() -> None:
    posterior = structure_posterior([1.0, 1.1, 1.2], complexity=0.25)
    assert int(np.argmax(posterior)) == 0
    np.testing.assert_allclose(posterior.sum(), 1.0)


def test_preference_and_habit_learning_use_counts_not_raw_preferences() -> None:
    counts = update_preference_counts([1.0, 1.0, 1.0], [2, 2, 1], learning_rate=0.5)
    np.testing.assert_allclose(counts, [1.0, 1.5, 2.0])
    habit = habit_prior_from_counts([1.0, 4.0], precision=2.0)
    np.testing.assert_allclose(habit.sum(), 1.0)
    assert habit[1] > habit[0]


def test_path_based_policy_scores_reject_axis_and_action_errors() -> None:
    policies = [[0, 1, 1], [1, 1, 1]]
    scores = path_based_policy_scores([0.5, 0.1], policies, terminal_costs=[0.2, 0.0])
    assert scores[1] < scores[0]
    with pytest.raises(ValueError):
        path_based_policy_scores([[0.5, 0.1]], policies)
