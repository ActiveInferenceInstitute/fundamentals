"""Example 11.3 - learning C/E-style preferences, habits, and path costs."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    save_chapter_data,
    simulate_path_policy_computation,
    simulate_preference_habit_learning,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic arrays for §11.2.4 and §11.2.8."""
    learned = simulate_preference_habit_learning([2, 2, 1, 2, 0], [1, 1, 0, 1], n_outcomes=3, n_actions=2)
    paths = simulate_path_policy_computation()
    return {
        "outcome_index": np.arange(learned.preference_counts.size, dtype=float),
        "preference_counts": learned.preference_counts,
        "normalized_preferences": learned.normalized_preferences,
        "action_index": np.arange(learned.habit_prior.size, dtype=float),
        "habit_prior": learned.habit_prior,
        "policy_index": np.arange(paths.scores.size, dtype=float),
        "path_scores": paths.scores,
        "path_posterior": paths.posterior,
        "policies": paths.policies.astype(float),
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the preference/habit/path computation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(arrays["outcome_index"], arrays["normalized_preferences"])
    axes[0].set_title("Learned C preferences")
    axes[0].set_xlabel("outcome")
    axes[0].set_ylabel("probability")
    axes[1].bar(arrays["action_index"], arrays["habit_prior"], color="tab:orange")
    axes[1].set_title("Habit E prior")
    axes[1].set_xlabel("action")
    axes[2].plot(arrays["policy_index"], arrays["path_scores"], marker="o", label="path cost")
    axes[2].plot(arrays["policy_index"], arrays["path_posterior"], marker="s", label="posterior")
    axes[2].set_title("Path-based policies")
    axes[2].set_xlabel("policy")
    axes[2].legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_11")
        figure = fig_dir / "example_11_3_preference_habit_learning.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(11, "example_11_3_preference_habit_learning", arrays, {"script": "example_11_3_preference_habit_learning.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 11 preference/habit example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
