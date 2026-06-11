"""Application-level Part III demos for robotics and social active inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NavigationResult:
    """A deterministic navigation trajectory with distance and preference traces."""

    path: np.ndarray
    goal: np.ndarray
    distance: np.ndarray
    preference: np.ndarray


def simulate_robot_navigation(
    *,
    n_steps: int = 80,
    goal: tuple[float, float] = (1.0, 0.75),
) -> NavigationResult:
    """Return a smooth navigation path with monotonically increasing goal preference."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    t = np.linspace(0.0, 1.0, n_steps)
    target = np.asarray(goal, dtype=float)
    path = np.column_stack(
        [
            target[0] * t,
            target[1] * (1.0 - np.cos(np.pi * t)) / 2.0,
        ]
    )
    distance = np.linalg.norm(target - path, axis=1)
    preference = np.exp(-4.0 * distance)
    return NavigationResult(path=path, goal=target, distance=distance, preference=preference)


@dataclass(frozen=True)
class FaultTolerantControlResult:
    """Control trace showing compensation after actuator efficacy drops."""

    time: np.ndarray
    desired: np.ndarray
    actual: np.ndarray
    efficacy: np.ndarray
    error: np.ndarray


def simulate_fault_tolerant_control(
    *,
    n_steps: int = 90,
    fault_time: float = 0.45,
    post_fault_efficacy: float = 0.55,
) -> FaultTolerantControlResult:
    """Return a deterministic fault-tolerant control trace for Chapter 13."""
    if n_steps < 3:
        raise ValueError("n_steps must be at least 3")
    if post_fault_efficacy <= 0.0 or post_fault_efficacy > 1.0:
        raise ValueError("post_fault_efficacy must lie in (0, 1]")
    time = np.linspace(0.0, 1.0, n_steps)
    desired = np.sin(np.pi * time)
    efficacy = np.where(time < fault_time, 1.0, post_fault_efficacy)
    compensation = 1.0 / efficacy
    actual = efficacy * compensation * desired
    error = desired - actual
    return FaultTolerantControlResult(time=time, desired=desired, actual=actual, efficacy=efficacy, error=error)


@dataclass(frozen=True)
class SocialInferenceResult:
    """Observation sequence and belief trace over another agent's hidden intention."""

    observations: np.ndarray
    beliefs: np.ndarray
    final_intention: int


def simulate_social_inference(
    observations: np.ndarray | list[int] | None = None,
    *,
    accuracy: float = 0.82,
) -> SocialInferenceResult:
    """Infer a binary hidden intention from communicative observations."""
    if observations is None:
        obs = np.array([0, 1, 1, 1, 0, 1, 1], dtype=int)
    else:
        obs = np.asarray(observations, dtype=int)
    if obs.ndim != 1 or obs.size == 0:
        raise ValueError("observations must be a non-empty 1-D array")
    if accuracy <= 0.5 or accuracy >= 1.0:
        raise ValueError("accuracy must lie in (0.5, 1)")
    likelihood = np.array([[accuracy, 1.0 - accuracy], [1.0 - accuracy, accuracy]])
    belief = np.array([0.5, 0.5])
    beliefs = [belief.copy()]
    for item in obs:
        if item not in (0, 1):
            raise ValueError("observations must be binary")
        belief = belief * likelihood[int(item)]
        belief = belief / belief.sum()
        beliefs.append(belief.copy())
    trace = np.asarray(beliefs)
    return SocialInferenceResult(
        observations=obs,
        beliefs=trace,
        final_intention=int(np.argmax(trace[-1])),
    )


@dataclass(frozen=True)
class RoboticsTheoryResult:
    """Relative emphasis of robotics themes discussed in Chapter 13."""

    themes: np.ndarray
    active_inference_weight: np.ndarray
    control_weight: np.ndarray


def robotics_theory_landscape() -> RoboticsTheoryResult:
    """Return deterministic weights for the Chapter 13 theory overview."""
    themes = np.arange(4, dtype=float)
    active = np.array([0.7, 0.82, 0.76, 0.68])
    control = np.array([0.62, 0.88, 0.55, 0.5])
    return RoboticsTheoryResult(themes=themes, active_inference_weight=active, control_weight=control)


__all__ = [
    "FaultTolerantControlResult",
    "NavigationResult",
    "RoboticsTheoryResult",
    "SocialInferenceResult",
    "robotics_theory_landscape",
    "simulate_fault_tolerant_control",
    "simulate_robot_navigation",
    "simulate_social_inference",
]
