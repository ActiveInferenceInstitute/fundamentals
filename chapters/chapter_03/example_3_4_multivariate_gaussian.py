"""Example 3.4 — Anatomy of a multivariate normal distribution.

Run::

    python chapters/chapter_03/example_3_4_multivariate_gaussian.py [--save]

Renders four 2-D Gaussians side by side under different covariance
structures: spherical, axis-aligned anisotropic, positive correlation,
and negative correlation. Each panel shows samples, the 1-σ and 2-σ
ellipses, and the underlying contour density.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    get_logger,
    mvn_pdf,
    mvn_sample,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    confidence_ellipse,
    save_or_show,
)

LOG = get_logger("ch3.ex4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=3)
    return p.parse_args()


CASES = [
    ("spherical (Σ = I)", np.eye(2)),
    ("anisotropic", np.diag([0.5, 4.0])),
    ("positive corr.", np.array([[2.0, 1.5], [1.5, 2.0]])),
    ("negative corr.", np.array([[1.0, -0.7], [-0.7, 1.0]])),
]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True,
                             sharex=True, sharey=True)

    grid = np.linspace(-4, 4, 200)
    XX, YY = np.meshgrid(grid, grid)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    mean = np.zeros(2)

    for ax, (label, cov) in zip(axes, CASES):
        density = mvn_pdf(pts, mean, cov).reshape(XX.shape)
        ax.contour(XX, YY, density, levels=8, cmap="viridis", alpha=0.7)
        samples = mvn_sample(mean, cov, n=80, rng=rng)
        ax.scatter(samples[:, 0], samples[:, 1], s=12, color="black",
                   alpha=0.6)
        for n_std, alpha in zip((1, 2), (0.45, 0.18)):
            ax.add_patch(confidence_ellipse(
                mean, cov, n_std=n_std, fc="#d62728", ec="#d62728",
                alpha=alpha, lw=1.5,
            ))
        ax.set_title(f"{label}\nΣ = {np.round(cov, 2).tolist()}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")

    fig.supxlabel(r"$x^{(0)}$")
    fig.supylabel(r"$x^{(1)}$")
    LOG.info("Rendered %d covariance shapes.", len(CASES))

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "example_3_4_mvn.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
