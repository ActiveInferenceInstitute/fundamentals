"""Example 13.1 - robotics navigation as preference satisfaction."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    save_chapter_data,
    simulate_robot_navigation,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic robot-navigation arrays."""
    result = simulate_robot_navigation(n_steps=90, goal=(1.0, 0.78))
    return {
        "x": result.path[:, 0],
        "y": result.path[:, 1],
        "goal": result.goal,
        "distance": result.distance,
        "preference": result.preference,
        "step": np.arange(result.distance.size, dtype=float),
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 13 navigation figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(arrays["x"], arrays["y"], marker=".", label="path")
    axes[0].scatter(arrays["goal"][0], arrays["goal"][1], color="black", label="goal")
    axes[0].set_title("Navigation trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[1].plot(arrays["step"], arrays["distance"], label="distance")
    axes[1].plot(arrays["step"], arrays["preference"], label="preference")
    axes[1].set_title("Goal evidence")
    axes[1].set_xlabel("step")
    axes[1].legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_13")
        figure = fig_dir / "example_13_1_robotics_navigation.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            13,
            "example_13_1_robotics_navigation",
            arrays,
            metadata={"script": "example_13_1_robotics_navigation.py"},
            figures=[figure],
        )
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
