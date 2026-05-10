"""§2.5.2 — Iterative MLE / MAP via gradient descent.

Run::

    python chapters/chapter_02/example_2_10_gradient_descent.py [--save]

The same closed-form solutions from Examples 2.8 / 2.9 are obtainable through
gradient descent on the negative log-likelihood / posterior. This script runs
both, plots the loss curve and iterate trajectory, and verifies that the
iterative answer matches the analytic answer.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from active_inference import (
    LinearGaussianProcess,
    get_logger,
    gradient_descent,
    map_analytic_linear,
    mle_analytic_linear,
)
from active_inference.estimators.map import map_grad_x, map_loss
from active_inference.estimators.mle import mle_grad_x, mle_loss
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_gradient_descent, save_or_show

LOG = get_logger("ch2.ex10")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--x-true", type=float, default=2.5)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Step size; the linear-Gaussian Hessian scales as "
                        "N * beta1**2 / sigma2_y, so lr * Hessian < 2 is required "
                        "for convergence.")
    p.add_argument("--max-iter", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    process = LinearGaussianProcess(beta0=3.0, beta1=2.0, sigma2_y=0.5, rng=rng)
    samples = process.sample(args.x_true, n=args.n_samples).flatten()
    LOG.info("Sample mean = %.3f, std = %.3f", samples.mean(), samples.std())

    # MLE descent
    mle_iter = gradient_descent(
        loss_fn=lambda x: float(mle_loss(x, samples, 3.0, 2.0, 0.5)),
        grad_fn=lambda x: mle_grad_x(x, samples, 3.0, 2.0, 0.5),
        x0=5.0,
        learning_rate=args.lr,
        max_iter=args.max_iter,
    )
    mle_closed = mle_analytic_linear(samples, 3.0, 2.0)
    LOG.info("MLE closed-form = %.4f, gradient descent = %.4f, true x* = %.3f",
             mle_closed, mle_iter.x, args.x_true)
    assert abs(mle_iter.x - mle_closed) < 5e-3

    # MAP descent
    map_iter = gradient_descent(
        loss_fn=lambda x: float(map_loss(
            x, samples, 3.0, 2.0, 0.5, 4.0, 0.25
        )),
        grad_fn=lambda x: map_grad_x(x, samples, 3.0, 2.0, 0.5, 4.0, 0.25),
        x0=4.0,
        learning_rate=args.lr,
        max_iter=args.max_iter,
    )
    map_closed = map_analytic_linear(samples, 3.0, 2.0, 0.5, 4.0, 0.25)
    LOG.info("MAP closed-form = %.4f, gradient descent = %.4f",
             map_closed, map_iter.x)
    assert abs(map_iter.x - map_closed) < 5e-3

    fig_mle = plot_gradient_descent(
        mle_iter.history, mle_iter.losses,
        truth=args.x_true,
        title=f"MLE gradient descent (converged after {mle_iter.n_iterations})",
    )
    fig_map = plot_gradient_descent(
        map_iter.history, map_iter.losses,
        truth=args.x_true,
        title=f"MAP gradient descent (converged after {map_iter.n_iterations})",
    )

    # Combined comparison plot showing both descents on the same axes.
    fig_cmp, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(mle_iter.history, color="#1f77b4", lw=2, label="MLE iterate")
    ax.plot(map_iter.history, color="#d62728", lw=2, label="MAP iterate")
    ax.axhline(args.x_true, color="black", ls=":", label="x*")
    ax.axhline(mle_closed, color="#1f77b4", ls="--", alpha=0.5,
               label=f"MLE closed = {mle_closed:.3f}")
    ax.axhline(map_closed, color="#d62728", ls="--", alpha=0.5,
               label=f"MAP closed = {map_closed:.3f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("hidden-state estimate")
    ax.set_title("MLE vs MAP descent")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_02")
        save_or_show(fig_mle, out / "example_2_10_mle_descent.png")
        save_or_show(fig_map, out / "example_2_10_map_descent.png")
        save_or_show(fig_cmp, out / "example_2_10_comparison.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
