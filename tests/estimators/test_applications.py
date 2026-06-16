"""Direct tests for application-level Part III demo helpers."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.estimators.applications import (
    robotics_theory_landscape,
    simulate_fault_tolerant_control,
    simulate_robot_navigation,
    simulate_social_inference,
)


class TestNavigationAndControl:
    def test_robot_navigation_reaches_goal_and_prefers_it(self) -> None:
        result = simulate_robot_navigation(n_steps=25, goal=(2.0, 1.0))
        np.testing.assert_allclose(result.path[-1], result.goal)
        assert result.distance[-1] == pytest.approx(0.0)
        assert result.preference[-1] > result.preference[0]

    def test_fault_tolerant_control_compensates_after_efficacy_drop(self) -> None:
        result = simulate_fault_tolerant_control(n_steps=40, post_fault_efficacy=0.5)
        np.testing.assert_allclose(result.actual, result.desired)
        assert np.min(result.efficacy) == pytest.approx(0.5)
        with pytest.raises(ValueError, match="post_fault_efficacy"):
            simulate_fault_tolerant_control(post_fault_efficacy=0.0)


class TestSocialAndTheory:
    def test_social_inference_updates_binary_belief_trace(self) -> None:
        result = simulate_social_inference([1, 1, 0, 1], accuracy=0.8)
        np.testing.assert_allclose(result.beliefs.sum(axis=1), np.ones(result.beliefs.shape[0]))
        assert result.final_intention == int(np.argmax(result.beliefs[-1]))
        with pytest.raises(ValueError, match="binary"):
            simulate_social_inference([0, 2])

    def test_robotics_theory_landscape_shapes_align(self) -> None:
        result = robotics_theory_landscape()
        assert result.themes.shape == result.active_inference_weight.shape == result.control_weight.shape
        assert np.all(result.active_inference_weight > 0.0)
