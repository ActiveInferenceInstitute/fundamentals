"""Example 13.3 - social robotics as hidden-intention inference."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    save_chapter_data,
    simulate_social_inference,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic social-inference arrays."""
    result = simulate_social_inference()
    return {
        "time": np.arange(result.beliefs.shape[0], dtype=float),
        "belief_intention_0": result.beliefs[:, 0],
        "belief_intention_1": result.beliefs[:, 1],
        "observations": result.observations.astype(float),
        "final_intention": np.array([result.final_intention], dtype=float),
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 13 social-inference figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].step(arrays["time"], arrays["belief_intention_0"], where="post", label="intention 0")
    axes[0].step(arrays["time"], arrays["belief_intention_1"], where="post", label="intention 1")
    axes[0].set_title("Belief over hidden intention")
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("probability")
    axes[0].legend()
    axes[1].bar(np.arange(arrays["observations"].size), arrays["observations"])
    axes[1].set_title("Communicative observations")
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("observation")
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_13")
        figure = fig_dir / "example_13_3_social_robotics.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            13,
            "example_13_3_social_robotics",
            arrays,
            metadata={"script": "example_13_3_social_robotics.py"},
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
