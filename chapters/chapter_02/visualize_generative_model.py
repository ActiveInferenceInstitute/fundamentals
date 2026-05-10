"""§2.4 — Visualizing the joint density p(x, y) of an agent's generative model.

Run::

    python chapters/chapter_02/visualize_generative_model.py [--save]

Three views are produced:

1. A heatmap of ``p(x, y)`` with a uniform prior — full envelope of the
   generator.
2. A second heatmap with a Gaussian prior — the joint is sliced to a band
   around the prior mean.
3. A 3-D surface of the same joint with the marginal prior projected onto
   the ``y = const`` plane.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    get_logger,
    make_grid,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_joint_heatmap, save_or_show

LOG = get_logger("ch2.joint")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_grid = make_grid(0.0, 5.0, 200)
    y_grid = make_grid(-1.0, 14.0, 200)

    uniform_model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.4,
        prior_kind="uniform", uniform_low=0.0, uniform_high=5.0,
    )
    gaussian_model = LinearGaussianModel(
        beta0=3.0, beta1=2.0, sigma2_y=0.4,
        m_x=2.5, s2_x=0.5, prior_kind="gaussian",
    )

    x_uni, y_uni, joint_uni = GridBayesianInference(uniform_model, x_grid).joint_density(y_grid)
    x_g, y_g, joint_g = GridBayesianInference(gaussian_model, x_grid).joint_density(y_grid)

    fig_uni = plot_joint_heatmap(
        x_uni, y_uni, joint_uni,
        title="Joint p(x, y) — uniform prior",
    )
    fig_g = plot_joint_heatmap(
        x_g, y_g, joint_g,
        title="Joint p(x, y) — Gaussian prior (m_x=2.5, s2_x=0.5)",
    )

    # 3-D surface for the Gaussian-prior model.
    fig_3d = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax3d = fig_3d.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(x_g, y_g)
    ax3d.plot_surface(X, Y, joint_g, cmap="magma", lw=0, antialiased=True)
    # Marginal prior collapsed onto the back wall.
    prior_marg = np.trapezoid(joint_g, y_g, axis=0)
    ax3d.plot(x_g, np.full_like(x_g, y_g.max()), prior_marg, color="red", lw=2,
              label="prior p(x)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("p(x, y)")
    ax3d.set_title("Generative model surface")
    ax3d.legend()

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig_uni, out / "joint_uniform.png")
        save_or_show(fig_g, out / "joint_gaussian.png")
        save_or_show(fig_3d, out / "joint_surface.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
