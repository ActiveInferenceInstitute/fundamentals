"""Example 12.3 - variational and marginal message passing."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import default_figure_dir, ensure_dir, marginal_message_passing, save_chapter_data, variational_message_update


def build_arrays() -> dict[str, np.ndarray]:
    """Build arrays for §12.4 and §12.4.1."""
    factor = np.array([[0.72, 0.28], [0.18, 0.82]])
    incoming = np.array([0.55, 0.45])
    marginal = marginal_message_passing([factor, factor], incoming)
    vmp = variational_message_update(np.log(factor + 1e-12), [incoming], target_axis=0)
    return {"step": np.arange(marginal.shape[0], dtype=float), "marginal": marginal, "vmp": vmp, "factor": factor}


def render(arrays: dict[str, np.ndarray], *, save: bool) -> None:
    """Render marginal and VMP message values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(arrays["step"], arrays["marginal"][:, 0], marker="o", label="state 0")
    axes[0].plot(arrays["step"], arrays["marginal"][:, 1], marker="s", label="state 1")
    axes[0].set_title("Marginal messages")
    axes[0].legend()
    axes[1].bar([0, 1], arrays["vmp"], color="tab:purple")
    axes[1].set_title("VMP update")
    if save:
        fig_dir = ensure_dir(default_figure_dir() / "chapter_12")
        figure = fig_dir / "example_12_3_vmp_marginal_messages.png"
        fig.savefig(figure, dpi=170)
        save_chapter_data(12, "example_12_3_vmp_marginal_messages", arrays, {"script": "example_12_3_vmp_marginal_messages.py"}, figures=[figure])
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments and render the Chapter 12 VMP example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    render(build_arrays(), save=args.save)


if __name__ == "__main__":
    main()
