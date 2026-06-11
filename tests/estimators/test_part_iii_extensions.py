"""Tests for Part III estimator-level demo helpers."""

from __future__ import annotations

import numpy as np

from active_inference import (
    robotics_theory_landscape,
    simulate_fault_tolerant_control,
    simulate_parameter_forgetting,
    simulate_path_policy_computation,
    simulate_preference_habit_learning,
    simulate_robot_navigation,
    simulate_social_inference,
    simulate_sophisticated_planning,
    simulate_state_preference_schedule,
    simulate_structure_learning,
)


def test_state_preference_schedule_tracks_time_varying_target() -> None:
    result = simulate_state_preference_schedule([[0.0, 2.0], [3.0, 0.0]])
    assert result.target_states.tolist() == [1, 0]
    np.testing.assert_allclose(result.schedule.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(result.posterior, result.schedule[-1])


def test_parameter_forgetting_increases_inverse_count_uncertainty() -> None:
    result = simulate_parameter_forgetting([1.0, 5.0, 9.0], rate=0.5)
    assert np.all(result.after <= result.before)
    assert np.all(result.uncertainty_after >= result.uncertainty_before)


def test_structure_learning_selects_penalized_evidence_winner() -> None:
    result = simulate_structure_learning([1.0, 1.8, 1.9], complexity=0.2)
    assert result.selected_index == 1
    np.testing.assert_allclose(result.posterior.sum(), 1.0)


def test_preference_habit_and_path_policy_estimators_are_normalized() -> None:
    learned = simulate_preference_habit_learning([2, 2, 1], [1, 1, 0], n_outcomes=3, n_actions=2)
    np.testing.assert_allclose(learned.normalized_preferences.sum(), 1.0)
    np.testing.assert_allclose(learned.habit_prior.sum(), 1.0)
    path = simulate_path_policy_computation()
    np.testing.assert_allclose(path.posterior.sum(), 1.0)
    assert int(np.argmax(path.posterior)) == int(np.argmin(path.scores))


def test_sophisticated_planning_reaches_rightward_goal_policy() -> None:
    result, beliefs, entropies = simulate_sophisticated_planning(n_states=5, horizon=4)
    assert result.best_policy.tolist() == [1, 1, 1, 1]
    assert int(np.argmax(beliefs[-1])) == 4
    assert np.all(np.isfinite(entropies))


def test_robot_navigation_distance_decreases_and_preference_increases() -> None:
    result = simulate_robot_navigation(n_steps=50)
    assert np.all(np.isfinite(result.path))
    assert result.distance[-1] < result.distance[0]
    assert result.preference[-1] > result.preference[0]
    assert np.all(np.diff(result.distance) <= 1e-12)


def test_fault_tolerant_control_and_theory_landscape_are_finite() -> None:
    result = simulate_fault_tolerant_control(n_steps=40)
    assert np.all(np.isfinite(result.actual))
    assert np.max(np.abs(result.error)) < 1e-12
    assert np.min(result.efficacy) < 1.0
    theory = robotics_theory_landscape()
    assert theory.themes.shape == theory.active_inference_weight.shape
    assert np.all(theory.active_inference_weight > 0.0)


def test_social_inference_beliefs_are_normalized_and_finite() -> None:
    result = simulate_social_inference([0, 1, 1, 1, 0, 1, 1], accuracy=0.82)
    assert result.final_intention == 1
    np.testing.assert_allclose(result.beliefs.sum(axis=1), np.ones(result.beliefs.shape[0]))
    assert np.all(np.isfinite(result.beliefs))
