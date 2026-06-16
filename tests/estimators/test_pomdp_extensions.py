"""Direct tests for estimator-level POMDP extension demos."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.estimators.pomdp_extensions import (
    make_line_world,
    simulate_parameter_forgetting,
    simulate_path_policy_computation,
    simulate_preference_habit_learning,
    simulate_sophisticated_planning,
    simulate_state_preference_schedule,
    simulate_structure_learning,
)


class TestPOMDPExtensionEstimators:
    def test_state_preferences_and_structure_learning_are_normalized(self) -> None:
        schedule = simulate_state_preference_schedule([[0.0, 2.0], [3.0, 0.0]])
        assert schedule.target_states.tolist() == [1, 0]
        np.testing.assert_allclose(schedule.schedule.sum(axis=1), np.ones(2))
        structure = simulate_structure_learning([0.0, 2.0, 1.0], complexity=0.1)
        np.testing.assert_allclose(structure.posterior.sum(), 1.0)
        assert structure.selected_index == 1

    def test_forgetting_habits_and_paths_have_expected_direction(self) -> None:
        forgetting = simulate_parameter_forgetting([1.0, 4.0, 8.0], rate=0.25)
        assert np.all(forgetting.after <= forgetting.before)
        assert np.all(forgetting.uncertainty_after >= forgetting.uncertainty_before)
        learned = simulate_preference_habit_learning([2, 2, 1], [1, 1, 0], n_outcomes=3, n_actions=2)
        np.testing.assert_allclose(learned.normalized_preferences.sum(), 1.0)
        np.testing.assert_allclose(learned.habit_prior.sum(), 1.0)
        path = simulate_path_policy_computation()
        assert int(np.argmax(path.posterior)) == int(np.argmin(path.scores))

    def test_line_world_and_sophisticated_planning_move_to_goal(self) -> None:
        model = make_line_world(n_states=4)
        assert model.A.shape == (4, 4)
        result, beliefs, entropies = simulate_sophisticated_planning(n_states=4, horizon=3)
        assert result.best_policy.tolist() == [1, 1, 1]
        assert int(np.argmax(beliefs[-1])) == 3
        assert np.all(np.isfinite(entropies))
        with pytest.raises(ValueError, match="n_states"):
            make_line_world(n_states=1)
