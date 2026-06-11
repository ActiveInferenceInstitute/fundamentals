"""Example 13.4 - robotics theory theme landscape."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import default_figure_dir, ensure_dir, robotics_theory_landscape, save_chapter_data


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic robotics-theory landscape arrays for §13.4."""
    result = robotics_theory_landscape()
    return {"themes": result.themes, "active_inference_weight": result.active_inference_weight, "control_weight": result.control_weight}


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render active-inference and control-theory emphasis curves."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(arrays["themes"], arrays["active_inference_weight"], marker="o", label="active inference")
    ax.plot(arrays["themes"], arrays["control_weight"], marker="s", label="control theory")
    ax.set_xlabel("theme")
    ax.set_ylabel("relative emphasis")
    ax.set_title("Robotics theory links")
    ax.legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_13")
        figure = fig_dir / "example_13_4_robotics_theory.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(13, "example_13_4_robotics_theory", arrays, {"script": "example_13_4_robotics_theory.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 13 theory example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
