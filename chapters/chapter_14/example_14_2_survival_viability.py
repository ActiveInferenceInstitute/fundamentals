"""Example 14.2 - survival, viability, and ergodic occupancy."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import default_figure_dir, ensure_dir, ergodic_ou_trajectory, save_chapter_data, viability_indicator


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic survival and viability arrays for §14.2."""
    trajectory = ergodic_ou_trajectory(n_steps=400, drift=0.08)
    indicator = viability_indicator(trajectory, -1.2, 1.2)
    time = np.arange(trajectory.size, dtype=float)
    occupancy = np.cumsum(indicator) / (time + 1.0)
    return {"time": time, "trajectory": trajectory, "viability_indicator": indicator, "running_viability": occupancy}


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render state viability and running survival probability."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(arrays["time"], arrays["trajectory"], label="state")
    axes[0].fill_between(arrays["time"], -1.2, 1.2, alpha=0.15, label="viable interval")
    axes[0].set_title("State viability")
    axes[0].legend()
    axes[1].plot(arrays["time"], arrays["running_viability"], color="tab:green")
    axes[1].set_title("Running survival probability")
    axes[1].set_ylim(0.0, 1.05)
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_14")
        figure = fig_dir / "example_14_2_survival_viability.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(14, "example_14_2_survival_viability", arrays, {"script": "example_14_2_survival_viability.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 14 viability example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
