"""Example 12.1 - forward/backward messages in a categorical factor graph."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    FactorGraphChain,
    backward_smoothing_messages,
    default_figure_dir,
    ensure_dir,
    forward_backward_smoothing,
    save_chapter_data,
    sum_product_chain,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build a categorical chain and its message-passing beliefs."""
    prior = np.array([0.7, 0.3])
    transition = np.array([[0.85, 0.2], [0.15, 0.8]])
    likelihoods = np.array([[0.9, 0.2], [0.6, 0.5], [0.2, 0.85], [0.35, 0.75]])
    chain = FactorGraphChain(prior=prior, transition=transition, likelihoods=likelihoods)
    forward = sum_product_chain(chain.prior, chain.transition, chain.likelihoods)
    backward = backward_smoothing_messages(chain.transition, chain.likelihoods)
    smoothed = forward_backward_smoothing(chain.prior, chain.transition, chain.likelihoods)
    return {
        "time": np.arange(likelihoods.shape[0], dtype=float),
        "forward_state_0": forward[:, 0],
        "forward_state_1": forward[:, 1],
        "backward_state_0": backward[:, 0],
        "backward_state_1": backward[:, 1],
        "smoothed_state_0": smoothed[:, 0],
        "smoothed_state_1": smoothed[:, 1],
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 12 message-passing figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(arrays["time"], arrays["forward_state_1"], marker="o", label="forward")
    axes[0].plot(arrays["time"], arrays["smoothed_state_1"], marker="s", label="smoothed")
    axes[0].set_title("State-1 belief")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("probability")
    axes[0].legend()
    axes[1].plot(arrays["time"], arrays["backward_state_0"], marker="o", label="state 0")
    axes[1].plot(arrays["time"], arrays["backward_state_1"], marker="s", label="state 1")
    axes[1].set_title("Backward messages")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("message")
    axes[1].legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_12")
        figure = fig_dir / "example_12_1_factor_graph_messages.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            12,
            "example_12_1_factor_graph_messages",
            arrays,
            metadata={"script": "example_12_1_factor_graph_messages.py"},
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
