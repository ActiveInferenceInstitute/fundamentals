"""Example 12.2 - sum-product belief propagation and backward smoothing."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import default_figure_dir, ensure_dir, forward_backward_smoothing, save_chapter_data, sum_product_chain


def build_arrays() -> dict[str, np.ndarray]:
    """Build forward and smoothed beliefs for §12.2-§12.3."""
    prior = np.array([0.6, 0.4])
    transition = np.array([[0.82, 0.22], [0.18, 0.78]])
    likelihoods = np.array([[0.8, 0.2], [0.45, 0.65], [0.25, 0.9], [0.35, 0.75]])
    forward = sum_product_chain(prior, transition, likelihoods)
    smoothed = forward_backward_smoothing(prior, transition, likelihoods)
    return {"time": np.arange(likelihoods.shape[0], dtype=float), "forward": forward, "smoothed": smoothed, "likelihoods": likelihoods}


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render forward and smoothed state beliefs."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(arrays["time"], arrays["forward"][:, 1], marker="o", label="forward q(s=1)")
    ax.plot(arrays["time"], arrays["smoothed"][:, 1], marker="s", label="smoothed q(s=1)")
    ax.set_xlabel("time")
    ax.set_ylabel("belief")
    ax.set_title("Belief propagation and backward smoothing")
    ax.legend()
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_12")
        figure = fig_dir / "example_12_2_belief_propagation_smoothing.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(12, "example_12_2_belief_propagation_smoothing", arrays, {"script": "example_12_2_belief_propagation_smoothing.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 12 smoothing example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
