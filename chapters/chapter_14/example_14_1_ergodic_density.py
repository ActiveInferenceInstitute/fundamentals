"""Example 14.1 - ergodic density and entropy bounds."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    bayesian_mechanics_summary,
    default_figure_dir,
    ensure_dir,
    ergodic_ou_trajectory,
    save_chapter_data,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build ergodic-density arrays and entropy-bound diagnostics."""
    trajectory = ergodic_ou_trajectory(n_steps=600, drift=0.06, diffusion=0.22)
    summary = bayesian_mechanics_summary(trajectory, bins=70, vfe_margin=0.4)
    return {
        "time": np.arange(trajectory.size, dtype=float),
        "trajectory": trajectory,
        "density_grid": summary.grid,
        "density": summary.density,
        "entropy_bound": np.array([summary.entropy, summary.upper_bound, summary.gap]),
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 14 ergodic-density figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(arrays["time"], arrays["trajectory"], color="black")
    axes[0].set_title("Ergodic trajectory")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("state")
    axes[1].plot(arrays["density_grid"], arrays["density"], color="black")
    axes[1].set_title("Occupancy density")
    axes[1].set_xlabel("state")
    axes[1].set_ylabel("density")
    axes[1].text(
        0.04,
        0.95,
        f"H={arrays['entropy_bound'][0]:.3f}\nupper={arrays['entropy_bound'][1]:.3f}",
        transform=axes[1].transAxes,
        va="top",
    )
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_14")
        figure = fig_dir / "example_14_1_ergodic_density.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            14,
            "example_14_1_ergodic_density",
            arrays,
            metadata={"script": "example_14_1_ergodic_density.py"},
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
