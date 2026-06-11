"""Example 12.4 - VMP messages and a hybrid discrete/continuous bridge."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    hybrid_model_bridge,
    save_chapter_data,
    variational_message_update,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build VMP and hybrid-message arrays."""
    log_factor = np.log(np.array([[0.85, 0.25], [0.15, 0.75]]) + 1e-12)
    incoming = np.array([0.35, 0.65])
    vmp = variational_message_update(log_factor, [incoming], target_axis=0)
    continuous = np.linspace(-1.0, 1.0, 7)
    joint = hybrid_model_bridge(continuous, vmp)
    return {
        "state_index": np.arange(vmp.size, dtype=float),
        "vmp_message": vmp,
        "continuous_state": continuous,
        "hybrid_joint": joint,
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 12 hybrid message bridge."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(arrays["state_index"], arrays["vmp_message"], color="0.2")
    axes[0].set_title("VMP categorical message")
    axes[0].set_xlabel("state")
    axes[0].set_ylabel("probability")
    im = axes[1].imshow(arrays["hybrid_joint"], aspect="auto", cmap="viridis")
    axes[1].set_title("Hybrid joint bridge")
    axes[1].set_xlabel("continuous feature")
    axes[1].set_ylabel("discrete state")
    fig.colorbar(im, ax=axes[1])
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_12")
        figure = fig_dir / "example_12_4_hybrid_message_bridge.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            12,
            "example_12_4_hybrid_message_bridge",
            arrays,
            metadata={"script": "example_12_4_hybrid_message_bridge.py"},
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
