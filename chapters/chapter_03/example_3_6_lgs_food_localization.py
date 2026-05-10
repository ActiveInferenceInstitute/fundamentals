"""Example 3.6 — Multivariate hidden-state inference via the LGS.

Run::

    python chapters/chapter_03/example_3_6_lgs_food_localization.py [--save]

An agent at the origin estimates the 2-D location of a food source from
``--n-samples`` noisy visual observations. With Θ = I and b = 0 the
likelihood reduces to ``y = x + ω`` and the LGS posterior is the classical
sensor-fusion update.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    LinearGaussianMVProcess,
    LinearGaussianSystem,
    diagonal_cov,
    get_logger,
    isotropic_cov,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    confidence_ellipse,
    save_or_show,
)

LOG = get_logger("ch3.ex6")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    true_x = np.array([0.4, 0.6])
    cov_y = diagonal_cov([0.07, 0.06])

    process = LinearGaussianMVProcess(
        Theta=np.eye(2), cov_y=cov_y, rng=rng,
    )
    Y = process.sample(true_x, n=args.n_samples).reshape(args.n_samples, 2)

    lgs = LinearGaussianSystem(
        Theta=np.eye(2),
        cov_y=cov_y,
        mx=np.array([0.5, 0.5]),
        cov_x=isotropic_cov(2, 0.5),
    )
    posterior = lgs.posterior_batch(Y)
    LOG.info("Posterior mean = %s, std = %s",
             np.round(posterior.mean, 3), np.round(posterior.std(), 3))
    LOG.info("True position = %s", true_x)

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)

    # Background: prior ellipse.
    for n_std, alpha in zip((1, 2), (0.18, 0.07)):
        ax.add_patch(confidence_ellipse(
            lgs.mx, lgs.cov_x, n_std=n_std, fc="#1f77b4", ec="#1f77b4",
            alpha=alpha, lw=1.0,
        ))

    # Posterior ellipse.
    for n_std, alpha in zip((1, 2), (0.5, 0.22)):
        ax.add_patch(confidence_ellipse(
            posterior.mean, posterior.cov, n_std=n_std,
            fc="#2ca02c", ec="#2ca02c", alpha=alpha, lw=1.5,
        ))

    ax.scatter(Y[:, 0], Y[:, 1], color="white", edgecolor="black", s=24,
               lw=0.6, label="noisy observations")
    ax.scatter(*posterior.mean, color="#2ca02c", s=70, marker="o",
               label=f"posterior mean ({posterior.mean[0]:.2f}, {posterior.mean[1]:.2f})")
    ax.scatter(*lgs.mx, color="#1f77b4", marker="s", s=70,
               label=f"prior mean ({lgs.mx[0]}, {lgs.mx[1]})")
    ax.scatter(*true_x, marker="x", color="red", s=120, lw=2,
               label=f"true food ({true_x[0]}, {true_x[1]})")
    ax.scatter(0, 0, marker="^", color="black", s=80, label="agent")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("horizontal position")
    ax.set_ylabel("vertical position")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("LGS sensor fusion · 2-D food localization")
    ax.legend(loc="lower right", fontsize=9)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "example_3_6_lgs.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
