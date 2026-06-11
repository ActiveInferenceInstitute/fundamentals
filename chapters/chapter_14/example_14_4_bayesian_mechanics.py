"""Example 14.4 - Markov-blanket flow and Bayesian-mechanics coupling."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    blanket_coupling_matrix,
    default_figure_dir,
    ensure_dir,
    save_chapter_data,
    simulate_markov_blanket_flow,
)


def build_arrays() -> dict[str, np.ndarray]:
    """Build Markov-blanket flow and coupling arrays."""
    flow = simulate_markov_blanket_flow(n_steps=180)
    coupling = blanket_coupling_matrix(flow)
    return {
        "time": flow.time,
        "external": flow.external,
        "blanket": flow.blanket,
        "internal": flow.internal,
        "coupling": coupling,
    }


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render the Chapter 14 Bayesian-mechanics figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(arrays["time"], arrays["external"], label="external")
    axes[0].plot(arrays["time"], arrays["blanket"], label="blanket")
    axes[0].plot(arrays["time"], arrays["internal"], label="internal")
    axes[0].set_title("Blanket-separated flow")
    axes[0].set_xlabel("time")
    axes[0].legend()
    im = axes[1].imshow(arrays["coupling"], vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1].set_title("Coupling matrix")
    axes[1].set_xticks([0, 1, 2], ["external", "blanket", "internal"])
    axes[1].set_yticks([0, 1, 2], ["external", "blanket", "internal"])
    fig.colorbar(im, ax=axes[1])
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_14")
        figure = fig_dir / "example_14_4_bayesian_mechanics.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(
            14,
            "example_14_4_bayesian_mechanics",
            arrays,
            metadata={"script": "example_14_4_bayesian_mechanics.py"},
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
