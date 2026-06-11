"""Example 13.2 - goal-directed and fault-tolerant control."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import default_figure_dir, ensure_dir, save_chapter_data, simulate_fault_tolerant_control


def build_arrays() -> dict[str, np.ndarray]:
    """Build deterministic fault-tolerant-control arrays for §13.2."""
    result = simulate_fault_tolerant_control(n_steps=90)
    return {"time": result.time, "desired": result.desired, "actual": result.actual, "efficacy": result.efficacy, "error": result.error}


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render desired, actual, efficacy, and residual-error traces."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(arrays["time"], arrays["desired"], label="desired")
    axes[0].plot(arrays["time"], arrays["actual"], "--", label="actual")
    axes[0].set_title("Fault compensation")
    axes[0].legend()
    axes[1].plot(arrays["time"], arrays["efficacy"], label="actuator efficacy")
    axes[1].plot(arrays["time"], arrays["error"], label="tracking error")
    axes[1].set_title("Fault and residual error")
    axes[1].legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_13")
        figure = fig_dir / "example_13_2_fault_tolerant_control.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(13, "example_13_2_fault_tolerant_control", arrays, {"script": "example_13_2_fault_tolerant_control.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 13 control example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
