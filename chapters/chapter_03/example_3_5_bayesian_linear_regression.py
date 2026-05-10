"""Example 3.5 — Bayesian linear regression.

Run::

    python chapters/chapter_03/example_3_5_bayesian_linear_regression.py [--save]

Visualises the prior over (β₀, β₁) and the posterior after seeing 1, 5, 15,
and 50 samples. Also shows the posterior-predictive band against the data.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    BayesianLinearRegression,
    LinearGaussianProcess,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import (
    confidence_ellipse,
    save_or_show,
)

LOG = get_logger("ch3.ex5")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--max-n", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=1.0, rng=rng)
    x_pool = rng.uniform(0, 5, size=args.max_n)
    y_pool = np.array([process.sample(float(x), n=1)[0] for x in x_pool])

    blr = BayesianLinearRegression(
        prior_mean=np.zeros(2),
        prior_cov=np.eye(2) * 4.0,
        sigma2_y=1.0,
    )

    snapshots = [0, 1, 5, 15, args.max_n]
    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 4),
                             constrained_layout=True,
                             sharex=True, sharey=True)
    grid = np.linspace(-4, 4, 100)
    XX, YY = np.meshgrid(grid, grid)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    truth = np.array([3.0, 2.0])

    final_post = None
    for ax, n in zip(axes, snapshots):
        if n == 0:
            mean = blr.prior_mean
            cov = blr.prior_cov
            label = "prior"
        else:
            post = blr.fit(x_pool[:n], y_pool[:n])
            mean = post.mean
            cov = post.cov
            label = f"posterior (N = {n})"
            if n == args.max_n:
                final_post = post
        diff = pts - mean
        prec = np.linalg.inv(cov)
        quad = np.einsum("ni,ij,nj->n", diff, prec, diff).reshape(XX.shape)
        density = np.exp(-0.5 * quad)
        ax.contourf(XX, YY, density, levels=10, cmap="Greens")
        for n_std, alpha in zip((1, 2), (0.45, 0.18)):
            ax.add_patch(confidence_ellipse(
                mean, cov, n_std=n_std, fc="none", ec="black",
                alpha=alpha + 0.4, lw=1.5,
            ))
        ax.scatter(*truth, marker="x", color="red", s=80, lw=2)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\hat{\beta}_0$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\hat{\beta}_1$")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)

    LOG.info("Final posterior mean = %s, std = %s",
             np.round(final_post.mean, 3), np.round(final_post.std(), 3))

    # Posterior predictive band figure.
    fig_pred, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    grid_x = np.linspace(0, 5, 200).reshape(-1, 1)
    mean_pred, var_pred = final_post.predictive(grid_x, sigma2_y=blr.sigma2_y)
    std_pred = np.sqrt(var_pred)
    ax.fill_between(grid_x.flatten(), mean_pred - 2 * std_pred,
                    mean_pred + 2 * std_pred, alpha=0.25, color="#2ca02c",
                    label="95% predictive")
    ax.plot(grid_x, mean_pred, color="#2ca02c", lw=2, label="predictive mean")
    ax.plot(grid_x, 3.0 + 2.0 * grid_x.flatten(), color="red", ls=":",
            label="true line")
    ax.scatter(x_pool, y_pool, s=14, color="black", alpha=0.6, label="data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior predictive distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "example_3_5_blr_posteriors.png")
        save_or_show(fig_pred, out / "example_3_5_blr_predictive.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
