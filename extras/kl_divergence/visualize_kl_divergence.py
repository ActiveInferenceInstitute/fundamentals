"""Extra topic: KL divergence on grids and Gaussian closed forms."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    gaussian_kl_univariate,
    gaussian_pdf,
    grid_kl_divergence,
    save_extra_data,
)
from active_inference.extra_topics import extra_topic_spec
from active_inference.visualizations import COLORS, save_or_show


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this extras orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Render or display the KL divergence topic figure."""
    args = parse_args()
    x_grid = np.linspace(-6.0, 8.0, 2001)
    mu_p, var_p = 0.0, 1.0
    mu_q, var_q = 1.5, 1.8
    p = gaussian_pdf(x_grid, mu_p, var_p)
    q = gaussian_pdf(x_grid, mu_q, var_q)

    kl_pq_grid = grid_kl_divergence(p, q, x_grid)
    kl_qp_grid = grid_kl_divergence(q, p, x_grid)
    kl_self = grid_kl_divergence(p, p, x_grid)
    kl_pq_closed = gaussian_kl_univariate(mu_p, var_p, mu_q, var_q)
    kl_qp_closed = gaussian_kl_univariate(mu_q, var_q, mu_p, var_p)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    axes[0].plot(x_grid, p, label="p = N(0, 1)", lw=2)
    axes[0].plot(x_grid, q, label="q = N(1.5, 1.8)", lw=2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("density")
    axes[0].set_title("Two densities on one support")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    labels = ["KL(p||p)", "KL(p||q)", "KL(q||p)"]
    values = [kl_self, kl_pq_grid, kl_qp_grid]
    axes[1].bar(
        labels,
        values,
        color=[COLORS["neutral"], COLORS["prior"], COLORS["likelihood"]],
    )
    axes[1].set_ylabel("nats")
    axes[1].set_title("KL is non-negative and asymmetric")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].text(
        0.02,
        0.95,
        f"grid vs closed KL(p||q): {kl_pq_grid - kl_pq_closed:+.2e}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )
    fig.suptitle("Kullback-Leibler divergence")

    if args.save:
        out = ensure_dir(default_figure_dir() / "extras" / "kl_divergence")
        figure = save_or_show(fig, out / "visualize_kl_divergence.png")
        save_extra_data(
            "kl_divergence",
            "visualize_kl_divergence",
            arrays={
                "x_grid": x_grid,
                "p": p,
                "q": q,
                "kl_grid": np.array([kl_self, kl_pq_grid, kl_qp_grid]),
                "kl_closed": np.array([0.0, kl_pq_closed, kl_qp_closed]),
            },
            metadata={
                "script": "visualize_kl_divergence.py",
                "source_apis": list(extra_topic_spec("kl_divergence").source_apis),
            },
            figures=[figure] if figure is not None else [],
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
